"""Jev backend: criteria -> Noul questions -> gate.

No network and no API key. A fake transport returns canned response bodies, so these tests
pin the CONTRACT (payload shape, gate arithmetic, failure behaviour) rather than the service.
Everything here would still pass if TypeSafe changed its model; what it catches is us breaking
the mapping between autonima's criteria and Jev's questions.
"""

import pytest

from autonima.backends.jev import (
    JevClient,
    JevError,
    apply_gate,
    build_criteria_questions,
    normalise_mapping,
)
from autonima.screening.jev_client import JevScreeningClient, build_state

MAPPING = {
    "inclusion": {"I1": "fMRI study of healthy adults", "I2": "reports whole-brain results"},
    "exclusion": {"E1": "review or meta-analysis", "E2": "paediatric sample"},
}


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body, self.status_code, self.text = body, status_code, str(body)

    def json(self):
        return self._body


def transport_returning(answers, usage=None, status=200):
    """A transport that records the payload it was handed and replays a canned answer set."""
    seen = {}

    def _transport(endpoint, headers, payload, timeout):
        seen["endpoint"], seen["headers"] = endpoint, headers
        seen["payload"], seen["timeout"] = payload, timeout
        return FakeResponse(
            {"model": "jev-1.13.0", "answers": answers,
             "usage": usage or {"input_tokens": 100, "output_tokens": 10}},
            status_code=status,
        )

    return _transport, seen


def nouls(**kwargs):
    return {k: {"type": "noul", "noul": v} for k, v in kwargs.items()}


# --- question construction ----------------------------------------------------------------

def test_one_noul_question_per_criterion_keyed_by_criterion_id():
    q = build_criteria_questions(MAPPING, objective="Find fMRI studies")
    assert set(q) == {"I1", "I2", "E1", "E2"}
    assert all(v["type"] == "noul" for v in q.values())
    assert q["I1"]["instructions"]["criterion"] == "fMRI study of healthy adults"
    assert q["I1"]["instructions"]["review_objective"] == "Find fMRI studies"
    assert set(q["E1"]["criteria"]) == {"true", "false"}


def test_objective_is_omitted_when_absent():
    q = build_criteria_questions(MAPPING)
    assert "review_objective" not in q["I1"]["instructions"]


def test_dataclass_and_dict_mappings_are_equivalent():
    """ConfigManager hands over the dataclass; the executed YAML hands over the dict."""
    from autonima.utils.criteria import CriteriaMapping

    dc = CriteriaMapping(inclusion=MAPPING["inclusion"], exclusion=MAPPING["exclusion"])
    assert normalise_mapping(dc) == normalise_mapping(MAPPING)
    assert build_criteria_questions(dc) == build_criteria_questions(MAPPING)


def test_empty_mapping_normalises_rather_than_raising():
    assert normalise_mapping(None) == {"inclusion": {}, "exclusion": {}}


# --- the gate -----------------------------------------------------------------------------

def test_include_requires_all_inclusion_and_no_exclusion():
    d = apply_gate(nouls(I1=0.9, I2=0.8, E1=0.1, E2=0.05), MAPPING)
    assert d.include
    assert d.satisfied_inclusion_ids == ["I1", "I2"]
    assert d.fired_exclusion_ids == []


def test_one_unmet_inclusion_criterion_rejects():
    d = apply_gate(nouls(I1=0.9, I2=0.2, E1=0.1, E2=0.05), MAPPING)
    assert not d.include
    assert d.satisfied_inclusion_ids == ["I1"]


def test_one_fired_exclusion_criterion_rejects():
    d = apply_gate(nouls(I1=0.9, I2=0.9, E1=0.95, E2=0.05), MAPPING)
    assert not d.include
    assert d.fired_exclusion_ids == ["E1"]


def test_missing_answer_is_treated_as_unsatisfied_not_skipped():
    """A dropped question must never become a lenient decision."""
    d = apply_gate(nouls(I1=0.9, E1=0.1, E2=0.05), MAPPING)   # I2 absent
    assert not d.include
    assert "I2" not in d.satisfied_inclusion_ids

    d2 = apply_gate(nouls(I1=0.9, I2=0.9, E1=0.1), MAPPING)   # E2 absent
    assert not d2.include
    assert "E2" in d2.fired_exclusion_ids


def test_thresholds_are_tunable_after_the_fact():
    """The whole point: re-gate stored probabilities without another model call."""
    answers = nouls(I1=0.62, I2=0.58, E1=0.1, E2=0.05)
    assert apply_gate(answers, MAPPING, inclusion_threshold=0.5).include
    assert not apply_gate(answers, MAPPING, inclusion_threshold=0.75).include


def test_confidence_is_the_weakest_link_not_the_average():
    d = apply_gate(nouls(I1=1.0, I2=0.51, E1=0.0, E2=0.0), MAPPING)
    assert d.confidence == pytest.approx(0.02, abs=1e-6)
    assert d.closest_call.criterion_id == "I2"


def test_out_of_range_threshold_rejected():
    with pytest.raises(ValueError):
        apply_gate(nouls(I1=1.0), MAPPING, inclusion_threshold=1.5)


# --- transport ----------------------------------------------------------------------------

def test_payload_matches_the_documented_request_shape():
    transport, seen = transport_returning(nouls(I1=0.9, I2=0.9, E1=0.0, E2=0.0))
    client = JevClient(transport=transport, api_key="k")
    client.gate(state={"title": "t"}, criteria_mapping=MAPPING)

    payload = seen["payload"]
    assert set(payload) == {"state", "model", "questions"}
    assert payload["model"] == "jev-latest"
    assert set(payload["questions"]) == {"I1", "I2", "E1", "E2"}
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["endpoint"].endswith("/v1/systemone")


def test_all_criteria_go_in_a_single_call():
    """Jev evaluates questions in parallel; issuing one call per criterion would waste that."""
    calls = []

    def counting(endpoint, headers, payload, timeout):
        calls.append(payload)
        return FakeResponse({"model": "m", "answers": nouls(I1=1, I2=1, E1=0, E2=0),
                             "usage": {}})

    JevClient(transport=counting, api_key="k").gate({"title": "t"}, MAPPING)
    assert len(calls) == 1


def test_client_errors_are_not_retried():
    attempts = []

    def failing(endpoint, headers, payload, timeout):
        attempts.append(1)
        return FakeResponse({"error": "bad schema"}, status_code=422)

    with pytest.raises(JevError):
        JevClient(transport=failing, api_key="k", max_retries=3).evaluate(
            "s", {"q": {"type": "noul", "instructions": "x"}})
    assert len(attempts) == 1, "422 is our bug; retrying it burns the rate limit"


def test_missing_answers_key_is_an_error():
    transport, _ = transport_returning({})
    bad = lambda e, h, p, t: FakeResponse({"model": "m", "usage": {}})
    with pytest.raises(JevError):
        JevClient(transport=bad, api_key="k").evaluate(
            "s", {"q": {"type": "noul", "instructions": "x"}})


def test_no_key_and_no_transport_fails_loudly(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError):
        JevClient()


def test_extra_question_colliding_with_a_criterion_id_is_refused():
    transport, _ = transport_returning(nouls(I1=1, I2=1, E1=0, E2=0))
    with pytest.raises(JevError):
        JevClient(transport=transport, api_key="k").gate(
            "s", MAPPING, extra_questions={"I1": {"type": "noul", "instructions": "x"}})


# --- screening adapter --------------------------------------------------------------------

def test_abstract_screening_returns_the_existing_schema():
    transport, _ = transport_returning(nouls(I1=0.9, I2=0.85, E1=0.02, E2=0.01))
    out = JevScreeningClient(client=JevClient(transport=transport, api_key="k")) \
        .screen_abstract_structured({"title": "t"}, MAPPING)
    assert out.decision == "INCLUDED"
    assert out.inclusion_criteria_applied == ["I1", "I2"]
    assert out.exclusion_criteria_applied == []
    assert 0.0 <= out.confidence <= 1.0
    assert out.reason.startswith("[jev]")


def test_rejection_reason_names_the_deciding_criterion():
    transport, _ = transport_returning(nouls(I1=0.9, I2=0.9, E1=0.97, E2=0.0))
    out = JevScreeningClient(client=JevClient(transport=transport, api_key="k")) \
        .screen_abstract_structured({"title": "t"}, MAPPING)
    assert out.decision == "EXCLUDED"
    assert "E1" in out.reason and "exclusion triggered" in out.reason


def test_fulltext_incomplete_is_its_own_question():
    answers = nouls(I1=0.9, I2=0.9, E1=0.0, E2=0.0)
    answers["__fulltext_incomplete"] = {"type": "noul", "noul": 0.93}
    transport, seen = transport_returning(answers)
    out = JevScreeningClient(client=JevClient(transport=transport, api_key="k")) \
        .screen_fulltext_structured({"title": "t", "full_text": "abs only"}, MAPPING)
    assert out.fulltext_incomplete is True
    assert "__fulltext_incomplete" in seen["payload"]["questions"]
    assert out.decision == "INCLUDED"


def test_screening_without_criteria_is_refused():
    transport, _ = transport_returning({})
    client = JevScreeningClient(client=JevClient(transport=transport, api_key="k"))
    with pytest.raises(JevError):
        client.screen_abstract_structured({"title": "t"}, {"inclusion": {}, "exclusion": {}})


def test_build_state_keeps_fields_separate():
    class S:
        title, abstract, pmid = "T", "A", "123"
        full_text = "BODY"

    assert build_state(S(), "abstract") == {"title": "T", "abstract": "A", "pmid": "123"}
    ft = build_state(S(), "fulltext")
    assert ft["full_text"] == "BODY"


def test_build_state_marks_absent_full_text():
    class S:
        title, abstract, full_text = "T", "A", ""

    assert build_state(S(), "fulltext")["full_text"] == "NO FULL TEXT AVAILABLE"


# --- screener dispatch --------------------------------------------------------------------

def _screener(tmp_path, backend):
    from autonima.models.types import ScreeningConfig
    from autonima.screening.screener import LLMScreener

    stage = {
        "model": "jev-latest" if backend == "jev" else "gpt-4o-mini",
        "backend": backend,
        "objective": "fMRI studies of adults",
        "inclusion_criteria": list(MAPPING["inclusion"].values()),
        "exclusion_criteria": list(MAPPING["exclusion"].values()),
        "criteria_mapping": MAPPING,
    }
    cfg = ScreeningConfig(abstract=dict(stage), fulltext=dict(stage))
    return LLMScreener(cfg, output_dir=str(tmp_path)), cfg


def _study():
    from autonima.models.types import Study

    return Study(
        pmid="12345678",
        title="An fMRI study of adults",
        abstract="We scanned 30 adults.",
        authors=["A. Author"],
        journal="J. Neuro",
        publication_date="2020-01-01",
    )


def test_backend_jev_routes_through_the_jev_path(tmp_path, monkeypatch):
    """`backend: jev` must reach JevScreeningClient, not the chat client."""
    import autonima.screening.jev_client as jc

    seen = {}

    class FakeJev:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def screen_abstract_structured(self, state, criteria_mapping, objective=None,
                                       guidance=None):
            seen["state"], seen["mapping"], seen["objective"] = state, criteria_mapping, objective
            seen["guidance"] = guidance
            from autonima.screening.schema import AbstractScreeningOutput

            return AbstractScreeningOutput(
                decision="INCLUDED", confidence=0.8, reason="[jev] all criteria satisfied",
                inclusion_criteria_applied=["I1", "I2"], exclusion_criteria_applied=[],
            )

    monkeypatch.setattr(jc, "JevScreeningClient", FakeJev)

    screener, cfg = _screener(tmp_path, "jev")
    result = screener._screen_single_study(_study(), "abstract", cfg.abstract)

    assert seen["objective"] == "fMRI studies of adults"
    assert seen["mapping"] == MAPPING
    assert seen["state"]["title"] == "An fMRI study of adults"
    assert seen["init"]["model"] == "jev-latest"
    assert result.reason.startswith("[jev]")


def test_thresholds_reach_the_client_from_stage_config(tmp_path, monkeypatch):
    import autonima.screening.jev_client as jc

    seen = {}

    class FakeJev:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def screen_abstract_structured(self, *a, **k):
            from autonima.screening.schema import AbstractScreeningOutput

            return AbstractScreeningOutput(decision="EXCLUDED", confidence=0.1, reason="[jev] x")

    monkeypatch.setattr(jc, "JevScreeningClient", FakeJev)
    screener, cfg = _screener(tmp_path, "jev")
    cfg.abstract["inclusion_threshold"] = 0.8
    cfg.abstract["exclusion_threshold"] = 0.3
    screener._screen_single_study(_study(), "abstract", cfg.abstract)

    assert seen["inclusion_threshold"] == 0.8
    assert seen["exclusion_threshold"] == 0.3


def test_default_backend_is_unchanged(tmp_path):
    """Absent `backend`, nothing about the existing path may change."""
    screener, cfg = _screener(tmp_path, "openai")
    assert cfg.abstract.get("backend") == "openai"
    # the jev branch keys off the string, so anything else falls through to the chat client
    assert str(cfg.abstract.get("backend", "openai")).lower() != "jev"


def test_adapter_accepts_the_criteria_mapping_dataclass():
    """ConfigManager hands the screener the dataclass, not a dict.

    Regression: the first full-project run failed on all 1,253 abstracts with
    "'CriteriaMapping' object has no attribute 'get'". The low-level helpers normalised, the
    adapter's own guard did not, and every adapter test here passed a dict.
    """
    from autonima.utils.criteria import CriteriaMapping

    dc = CriteriaMapping(inclusion=MAPPING["inclusion"], exclusion=MAPPING["exclusion"])
    transport, seen = transport_returning(nouls(I1=0.9, I2=0.9, E1=0.0, E2=0.0))
    out = JevScreeningClient(client=JevClient(transport=transport, api_key="k")) \
        .screen_abstract_structured({"title": "t"}, dc)
    assert out.decision == "INCLUDED"
    assert set(seen["payload"]["questions"]) == {"I1", "I2", "E1", "E2"}


def test_adapter_rejects_an_empty_dataclass_mapping():
    from autonima.utils.criteria import CriteriaMapping

    transport, _ = transport_returning({})
    with pytest.raises(JevError):
        JevScreeningClient(client=JevClient(transport=transport, api_key="k")) \
            .screen_abstract_structured({"title": "t"}, CriteriaMapping())


def test_usage_is_recorded_under_the_pipeline_stage_names():
    """execution.py snapshots with llm_usage.snapshot("abstract"/"fulltext"/"annotation").

    Regression: the first full run recorded under "jev_screening", which never appears in
    execution_progress.json, so the whole run reported zero cost.
    """
    from autonima.llm import usage as llm_usage

    llm_usage.reset()
    transport, _ = transport_returning(
        nouls(I1=0.9, I2=0.9, E1=0.0, E2=0.0), usage={"input_tokens": 1000, "output_tokens": 10})
    client = JevScreeningClient(client=JevClient(transport=transport, api_key="k"))
    client.screen_abstract_structured({"title": "t"}, MAPPING)
    assert llm_usage.snapshot("abstract"), "nothing recorded under 'abstract'"
    assert not llm_usage.snapshot("jev_screening")

    llm_usage.reset()
    answers = nouls(I1=0.9, I2=0.9, E1=0.0, E2=0.0)
    answers["__fulltext_incomplete"] = {"type": "noul", "noul": 0.0}
    transport2, _ = transport_returning(answers, usage={"input_tokens": 9000, "output_tokens": 20})
    JevScreeningClient(client=JevClient(transport=transport2, api_key="k")) \
        .screen_fulltext_structured({"title": "t", "full_text": "x"}, MAPPING)
    snap = llm_usage.snapshot("fulltext")
    assert snap and snap["input_tokens"] == 9000
    assert snap["cost_usd"] is not None, "jev must be priced so cost is a number"


def test_probabilities_are_persisted_so_thresholds_can_be_swept_later():
    """The whole point of a calibrated backend: re-gate a finished run with no API calls.

    Regression: the first full project run stored only the DECIDING criteria, inside the
    `reason` string, so the threshold sweep the backend exists to enable was impossible
    without re-running.
    """
    from autonima.backends.jev import apply_gate

    probs = dict(I1=0.62, I2=0.58, E1=0.10, E2=0.05)
    transport, _ = transport_returning(nouls(**probs))
    out = JevScreeningClient(client=JevClient(transport=transport, api_key="k")) \
        .screen_abstract_structured({"title": "t"}, MAPPING)

    assert out.criterion_probabilities == pytest.approx(probs)
    assert out.decision == "INCLUDED"

    # re-gate the STORED numbers at a stricter threshold, no transport involved
    stored = {k: {"type": "noul", "noul": v} for k, v in out.criterion_probabilities.items()}
    assert not apply_gate(stored, MAPPING, inclusion_threshold=0.75).include


def test_probabilities_survive_into_the_persisted_screening_result(tmp_path, monkeypatch):
    """ScreeningResult.to_dict() is what lands on disk; the vector must be in it."""
    import autonima.screening.jev_client as jc

    class FakeJev:
        def __init__(self, **kwargs):
            pass

        def screen_abstract_structured(self, state, criteria_mapping, objective=None,
                                       guidance=None):
            from autonima.screening.schema import AbstractScreeningOutput

            return AbstractScreeningOutput(
                decision="INCLUDED", confidence=0.4, reason="[jev] all criteria satisfied",
                inclusion_criteria_applied=["I1", "I2"],
                criterion_probabilities={"I1": 0.7, "I2": 0.66, "E1": 0.2, "E2": 0.1},
            )

    monkeypatch.setattr(jc, "JevScreeningClient", FakeJev)
    screener, cfg = _screener(tmp_path, "jev")
    result = screener._screen_single_study(_study(), "abstract", cfg.abstract)
    assert result.criterion_probabilities["I2"] == 0.66
    assert "criterion_probabilities" in result.to_dict()
    assert result.to_dict()["criterion_probabilities"]["E1"] == 0.2


def test_empty_criteria_refuses_rather_than_including_everything():
    """all([]) is True. An empty mapping must not wave every item through.

    Reachable in practice: a threshold sweep that mis-parses criterion IDs builds an empty
    mapping and would otherwise report 100% selection at every threshold.
    """
    with pytest.raises(JevError):
        apply_gate(nouls(I1=0.9), {"inclusion": {}, "exclusion": {}})


def test_screening_threshold_is_not_in_the_stage_hash(tmp_path):
    from autonima.models.types import ScreeningConfig
    from autonima.screening.screener import LLMScreener

    def sig(tau):
        stage = {"model": "jev-latest", "backend": "jev", "objective": "o",
                 "criteria_mapping": MAPPING, "inclusion_threshold": tau,
                 "exclusion_threshold": tau}
        s = LLMScreener(ScreeningConfig(abstract=dict(stage), fulltext=dict(stage)),
                        output_dir=str(tmp_path))
        return s._screening_cache_signature(_study(), "abstract", stage)["stage_hash"]

    assert sig(0.5) == sig(0.2)


def test_cached_screening_result_is_regated(tmp_path):
    from autonima.models.types import ScreeningConfig
    from autonima.screening.screener import LLMScreener

    cached = {"study_id": "1", "decision": "included_abstract", "screening_type": "abstract",
              "criterion_probabilities": {"I1": 0.62, "I2": 0.58, "E1": 0.0, "E2": 0.0}}
    stage = {"model": "jev-latest", "backend": "jev", "criteria_mapping": MAPPING,
             "inclusion_threshold": 0.75, "exclusion_threshold": 0.5}
    s = LLMScreener(ScreeningConfig(abstract=dict(stage), fulltext=dict(stage)),
                    output_dir=str(tmp_path))
    assert s._regate_cached(cached, stage)["decision"] == "excluded_abstract"
    loose = {**stage, "inclusion_threshold": 0.3}
    assert s._regate_cached(cached, loose)["decision"] == "included_abstract"


def test_execution_stage_hash_ignores_the_threshold():
    """The execution layer invalidates the artifact FILE, before the per-study cache is read.

    Regression: excluding thresholds from the per-study signature alone was not enough -- the
    screening stages splat their whole config block into the execution signature, so a
    threshold change wiped the results file and the per-study cache was never consulted.
    """
    from autonima.execution import stage_hashes

    base = {
        "search": {"database": "pubmed", "query": "q", "email": "a@b.c"},
        "screening": {
            "abstract": {"model": "jev-latest", "backend": "jev", "objective": "o",
                         "inclusion_criteria": ["c"], "inclusion_threshold": 0.5,
                         "exclusion_threshold": 0.5},
            "fulltext": {"model": "jev-latest", "backend": "jev", "objective": "o",
                         "inclusion_criteria": ["c"], "inclusion_threshold": 0.5,
                         "exclusion_threshold": 0.5},
        },
    }
    tuned = {**base, "screening": {
        k: {**v, "inclusion_threshold": 0.2, "exclusion_threshold": 0.2}
        for k, v in base["screening"].items()}}
    a, b = stage_hashes(base), stage_hashes(tuned)
    assert a["abstract"] == b["abstract"]
    assert a["fulltext"] == b["fulltext"]

    # something that changes what is asked must still invalidate
    changed = {**base, "screening": {
        k: {**v, "inclusion_criteria": ["different"]} for k, v in base["screening"].items()}}
    assert stage_hashes(changed)["abstract"] != a["abstract"]
