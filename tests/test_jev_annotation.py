"""Analysis-level selection through Jev. Fake transport; no network, no key."""

import pytest

from autonima.annotation.jev_client import (
    JevAnnotationClient,
    build_state,
    describe_analysis,
    mapping_for,
)
from autonima.annotation.schema import (
    AnalysisMetadata,
    AnnotationCriteriaConfig,
    StudyAnalysisGroup,
)
from autonima.backends.jev import JevClient

FIELDS = ["analysis_name", "analysis_description", "table_caption", "study_title",
          "study_fulltext"]


def criteria(name="reappraisal", with_mapping=True):
    kwargs = dict(
        name=name,
        description="Reappraisal versus maintain",
        inclusion_criteria=["contrast is reappraise > maintain", "whole-brain analysis"],
        exclusion_criteria=["ROI-only analysis"],
    )
    if with_mapping:
        kwargs["criteria_mapping"] = {
            "inclusion": {"I1": "contrast is reappraise > maintain",
                          "I2": "whole-brain analysis"},
            "exclusion": {"E1": "ROI-only analysis"},
        }
    return AnnotationCriteriaConfig(**kwargs)


def analysis(aid, name):
    return AnalysisMetadata(analysis_id=aid, study_id="S1", table_id="T1",
                            analysis_name=name, study_title="A study",
                            study_fulltext="Full article body.")


def group(n=2):
    return StudyAnalysisGroup(
        study_id="S1", study_title="A study", study_abstract="Abstract.",
        study_fulltext="Full article body.",
        analyses=[analysis(f"a{i}", f"analysis {i}") for i in range(n)],
    )


class FakeResponse:
    def __init__(self, body):
        self._body, self.status_code = body, 200

    def json(self):
        return self._body


def transport(prob_for):
    """Answers every question with a probability chosen by a callable over the question key."""
    calls = []

    def _t(endpoint, headers, payload, timeout):
        calls.append(payload)
        answers = {k: {"type": "noul", "noul": prob_for(k, q)}
                   for k, q in payload["questions"].items()}
        return FakeResponse({"model": "jev-1.13.0", "answers": answers,
                             "usage": {"input_tokens": 10, "output_tokens": 1}})

    return _t, calls


def client_with(transport_fn, **kwargs):
    return JevAnnotationClient(client=JevClient(transport=transport_fn, api_key="k"), **kwargs)


# --- mapping ------------------------------------------------------------------------------

def test_mapping_uses_the_config_when_present():
    m = mapping_for(criteria())
    assert set(m["inclusion"]) == {"I1", "I2"} and set(m["exclusion"]) == {"E1"}


def test_mapping_is_synthesised_when_the_config_omits_it():
    """Annotation criteria are not given IDs by ConfigManager, unlike screening criteria."""
    m = mapping_for(criteria(with_mapping=False))
    assert m["inclusion"] == {"I1": "contrast is reappraise > maintain",
                              "I2": "whole-brain analysis"}
    assert m["exclusion"] == {"E1": "ROI-only analysis"}


# --- question construction ------------------------------------------------------------------

def test_one_question_per_analysis_criterion_pair():
    t, calls = transport(lambda k, q: 1.0)
    client_with(t).make_decision(group(2), [criteria()], FIELDS)
    # 2 analyses x 3 criteria
    assert len(calls[0]["questions"]) == 6


def test_questions_scale_across_multiple_target_contrasts():
    t, calls = transport(lambda k, q: 1.0)
    client_with(t).make_decision(group(2), [criteria("a"), criteria("b")], FIELDS)
    assert len(calls[0]["questions"]) == 12  # 2 analyses x 2 targets x 3 criteria


def test_article_is_in_the_state_and_not_repeated_in_every_question():
    """Repeating a full text per question would multiply input tokens by the question count."""
    t, calls = transport(lambda k, q: 1.0)
    client_with(t).make_decision(group(2), [criteria()], FIELDS)
    payload = calls[0]
    assert payload["state"]["full_text"] == "Full article body."
    for q in payload["questions"].values():
        assert "study_fulltext" not in q["instructions"]["analysis"]


def test_question_carries_the_analysis_it_is_about():
    t, calls = transport(lambda k, q: 1.0)
    client_with(t).make_decision(group(2), [criteria()], FIELDS)
    ids = {q["instructions"]["analysis"]["analysis_id"]
           for q in calls[0]["questions"].values()}
    assert ids == {"a0", "a1"}


def test_keys_are_opaque_so_free_text_ids_cannot_collide():
    weird = StudyAnalysisGroup(study_id="S1", analyses=[
        AnalysisMetadata(analysis_id="a|b", study_id="S1", table_id="T"),
        AnalysisMetadata(analysis_id="a", study_id="S1", table_id="T"),
    ])
    t, calls = transport(lambda k, q: 1.0)
    out = client_with(t).make_decision(weird, [criteria()], FIELDS)
    assert all(k.startswith("q") for k in calls[0]["questions"])
    assert {d.analysis_id for d in out} == {"a|b", "a"}


# --- decisions ------------------------------------------------------------------------------

def test_one_decision_per_analysis_and_target():
    t, _ = transport(lambda k, q: 1.0)
    out = client_with(t).make_decision(group(3), [criteria("x"), criteria("y")], FIELDS)
    assert len(out) == 6
    assert {(d.analysis_id, d.annotation_name) for d in out} == {
        (f"a{i}", n) for i in range(3) for n in ("x", "y")}


def test_probabilities_are_routed_to_the_right_analysis():
    """The failure this guards: one analysis's probability attaching to another."""
    def prob(key, q):
        # only analysis a1 satisfies the inclusion criteria
        is_a1 = q["instructions"]["analysis"]["analysis_id"] == "a1"
        crit = q["instructions"]["criterion"]
        if crit == "ROI-only analysis":
            return 0.0
        return 0.95 if is_a1 else 0.05

    t, _ = transport(prob)
    out = {d.analysis_id: d for d in client_with(t).make_decision(group(2), [criteria()], FIELDS)}
    assert out["a1"].include is True
    assert out["a0"].include is False


def test_exclusion_criterion_rejects_the_analysis():
    def prob(key, q):
        return 0.99 if q["instructions"]["criterion"] == "ROI-only analysis" else 0.95

    t, _ = transport(prob)
    out = client_with(t).make_decision(group(1), [criteria()], FIELDS)
    assert out[0].include is False
    assert out[0].exclusion_criteria_applied == ["E1"]
    assert "exclusion triggered" in out[0].reasoning


def test_decision_carries_the_existing_schema_fields():
    t, _ = transport(lambda k, q: 0.9 if "ROI" not in q["instructions"]["criterion"] else 0.1)
    d = client_with(t).make_decision(group(1), [criteria()], FIELDS)[0]
    assert d.study_id == "S1" and d.annotation_name == "reappraisal"
    assert d.include is True
    assert d.inclusion_criteria_applied == ["I1", "I2"]
    assert d.reasoning.startswith("[jev]")
    assert 0.0 <= d.confidence <= 1.0


def test_thresholds_apply():
    t, _ = transport(lambda k, q: 0.6 if "ROI" not in q["instructions"]["criterion"] else 0.0)
    assert client_with(t, inclusion_threshold=0.5).make_decision(
        group(1), [criteria()], FIELDS)[0].include is True
    assert client_with(t, inclusion_threshold=0.8).make_decision(
        group(1), [criteria()], FIELDS)[0].include is False


def test_missing_answer_does_not_silently_include():
    def half_answering(endpoint, headers, payload, timeout):
        keys = list(payload["questions"])
        answers = {k: {"type": "noul", "noul": 1.0} for k in keys[:-1]}  # drop one
        return FakeResponse({"model": "m", "answers": answers, "usage": {}})

    out = client_with(half_answering).make_decision(group(1), [criteria()], FIELDS)
    assert out[0].include is False


# --- batching -------------------------------------------------------------------------------

def test_small_studies_still_go_in_one_call():
    t, calls = transport(lambda k, q: 1.0)
    client_with(t).make_decision(group(4), [criteria()], FIELDS)
    assert len(calls) == 1
    assert sum(len(c["questions"]) for c in calls) == 12


def test_many_analyses_are_split_to_respect_the_64k_request_budget():
    """Regression: a fixed question COUNT lost 63 of 128 studies to max_tokens_exceeded.

    The binding constraint is tokens -- the article in the state plus per-question analysis
    metadata -- so chunking has to be budget-driven.
    """
    from autonima.backends.jev import MAX_REQUEST_TOKENS, estimate_tokens

    big = StudyAnalysisGroup(
        study_id="S1", study_title="A study", study_fulltext="word " * 8000,
        analyses=[AnalysisMetadata(analysis_id=f"a{i}", study_id="S1", table_id="T1",
                                   analysis_name=f"analysis {i}",
                                   analysis_description="d" * 2000)
                  for i in range(40)],
    )
    t, calls = transport(lambda k, q: 1.0)
    out = client_with(t).make_decision(big, [criteria("x"), criteria("y")], FIELDS)
    assert len(calls) > 1, "a 40-analysis study must not go in one request"
    for payload in calls:
        total = estimate_tokens(payload["state"]) + sum(
            estimate_tokens({k: q}) for k, q in payload["questions"].items())
        assert total <= MAX_REQUEST_TOKENS, f"chunk of {total} tokens exceeds the budget"
    assert len(out) == 40 * 2


def test_an_oversized_article_is_trimmed_from_the_middle():
    """Head has title/abstract/methods, tail has results and tables. Cutting the tail to fit
    would systematically drop the evidence the criteria turn on."""
    from autonima.backends.jev import MAX_STATE_PLUS_QUESTION_TOKENS, estimate_tokens

    huge = StudyAnalysisGroup(
        study_id="S1", study_title="A study",
        study_fulltext="HEAD_MARKER " + ("filler " * 200000) + " TAIL_MARKER",
        analyses=[analysis("a0", "one")],
    )
    t, calls = transport(lambda k, q: 1.0)
    client_with(t).make_decision(huge, [criteria()], FIELDS)
    sent = calls[0]["state"]["full_text"]
    assert estimate_tokens(calls[0]["state"]) <= MAX_STATE_PLUS_QUESTION_TOKENS
    assert "HEAD_MARKER" in sent and "TAIL_MARKER" in sent
    assert "truncated" in sent


def test_chunking_does_not_change_the_decisions():
    prob = lambda k, q: 0.9 if "ROI" not in q["instructions"]["criterion"] else 0.1  # noqa: E731
    t1, _ = transport(prob)
    t2, _ = transport(prob)
    small = client_with(t1).make_decision(group(4), [criteria()], FIELDS)
    import autonima.backends.jev as jev
    old = jev.MAX_REQUEST_TOKENS
    jev.MAX_REQUEST_TOKENS = 1200          # force many chunks
    try:
        many = client_with(t2).make_decision(group(4), [criteria()], FIELDS)
    finally:
        jev.MAX_REQUEST_TOKENS = old
    assert [(d.analysis_id, d.include) for d in small] == \
           [(d.analysis_id, d.include) for d in many]


# --- edge cases -----------------------------------------------------------------------------

def test_no_criteria_returns_no_decisions():
    t, calls = transport(lambda k, q: 1.0)
    assert client_with(t).make_decision(group(1), [], FIELDS) == []
    assert calls == []


def test_no_analyses_returns_no_decisions():
    t, calls = transport(lambda k, q: 1.0)
    empty = StudyAnalysisGroup(study_id="S1", analyses=[])
    assert client_with(t).make_decision(empty, [criteria()], FIELDS) == []
    assert calls == []


def test_single_analysis_mode_uses_the_same_machinery():
    t, _ = transport(lambda k, q: 0.9 if "ROI" not in q["instructions"]["criterion"] else 0.1)
    out = client_with(t).make_decision(
        analysis("a7", "solo"), [criteria()], FIELDS, prompt_type="single_analysis")
    assert len(out) == 1 and out[0].analysis_id == "a7" and out[0].include is True


def test_metadata_fields_are_honoured():
    a = analysis("a0", "nm")
    a.analysis_description = "desc"
    a.table_caption = "cap"
    described = describe_analysis(a, ["analysis_name"])
    assert "analysis_description" not in described and described["analysis_name"] == "nm"


def test_build_state_omits_fulltext_when_not_requested():
    assert "full_text" not in build_state(group(1), ["analysis_name"])
    assert "full_text" in build_state(group(1), FIELDS)


# --- processor dispatch ---------------------------------------------------------------------

def test_processor_selects_the_jev_client_on_backend_jev():
    from autonima.annotation.processor import AnnotationProcessor
    from autonima.annotation.schema import AnnotationConfig

    cfg = AnnotationConfig(backend="jev", model="jev-latest", annotations=[criteria()])
    proc = AnnotationProcessor.__new__(AnnotationProcessor)
    import autonima.annotation.jev_client as jc

    captured = {}
    real = jc.JevAnnotationClient

    class Spy(real):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(client=JevClient(transport=transport(lambda k, q: 1.0)[0],
                                              api_key="k"),
                             **{k: v for k, v in kwargs.items() if k != "model"})

    jc.JevAnnotationClient = Spy
    try:
        AnnotationProcessor.__init__(proc, cfg)
        assert isinstance(proc.client, real)
        assert captured["model"] == "jev-latest"
    finally:
        jc.JevAnnotationClient = real


def test_processor_default_backend_is_the_chat_client():
    from autonima.annotation.client import AnnotationClient
    from autonima.annotation.processor import AnnotationProcessor
    from autonima.annotation.schema import AnnotationConfig

    proc = AnnotationProcessor(AnnotationConfig(annotations=[criteria()]))
    assert isinstance(proc.client, AnnotationClient)


def test_annotation_decisions_persist_the_probability_vector():
    t, _ = transport(lambda k, q: 0.9 if "ROI" not in q["instructions"]["criterion"] else 0.1)
    d = client_with(t).make_decision(group(1), [criteria()], FIELDS)[0]
    assert d.criterion_probabilities == {"I1": 0.9, "I2": 0.9, "E1": 0.1}


def test_stored_probabilities_can_be_re_gated_offline():
    from autonima.backends.jev import apply_gate
    from autonima.annotation.jev_client import mapping_for

    t, _ = transport(lambda k, q: 0.6 if "ROI" not in q["instructions"]["criterion"] else 0.0)
    d = client_with(t).make_decision(group(1), [criteria()], FIELDS)[0]
    stored = {k: {"type": "noul", "noul": v} for k, v in d.criterion_probabilities.items()}
    m = mapping_for(criteria())
    assert d.include is True
    assert not apply_gate(stored, m, inclusion_threshold=0.8).include


def test_config_loader_preserves_the_annotation_backend(tmp_path):
    """_load_annotation_config is an explicit allowlist, so new fields need adding to it.

    Regression: `backend: jev` parsed, validated and ran a whole project while the loader
    silently dropped it, so every annotation request went to OpenAI and 404'd on `jev-latest`.
    """
    import yaml
    from autonima.config import ConfigManager

    cfg = {
        "search": {"database": "pubmed", "query": "x", "email": "a@b.c"},
        "screening": {
            "abstract": {"objective": "o", "inclusion_criteria": ["c"]},
            "fulltext": {"objective": "o", "inclusion_criteria": ["c"]},
        },
        "annotation": {
            "model": "jev-latest", "backend": "jev",
            "inclusion_threshold": 0.7, "exclusion_threshold": 0.3,
            "model_params": {"reasoning_effort": "none"},
            "annotations": [],
        },
    }
    f = tmp_path / "c.yml"
    f.write_text(yaml.safe_dump(cfg))
    loaded = ConfigManager().load_from_file(str(f))
    assert loaded.annotation.backend == "jev"
    assert loaded.annotation.inclusion_threshold == 0.7
    assert loaded.annotation.exclusion_threshold == 0.3
    # pre-existing drop, fixed alongside: annotation model_params never reached the config
    assert loaded.annotation.model_params == {"reasoning_effort": "none"}


def test_every_annotation_config_field_is_reachable_from_yaml(tmp_path):
    """Guard the allowlist itself: a field on the model that the loader cannot set is a bug."""
    import yaml
    from autonima.annotation.schema import AnnotationConfig
    from autonima.config import ConfigManager

    ignore = {"annotations"}  # built separately from its own sub-dicts
    probe = {
        "model": "probe-model", "backend": "jev", "enabled": False,
        "prompt_type": "single_analysis", "create_all_included_annotations": False,
        "metadata_fields": ["analysis_name"], "inclusion_criteria": ["i"],
        "exclusion_criteria": ["e"], "inclusion_threshold": 0.9,
        "exclusion_threshold": 0.1, "model_params": {"k": "v"},
    }
    missing = set(AnnotationConfig.model_fields) - set(probe) - ignore
    assert not missing, f"AnnotationConfig gained fields with no loader coverage: {missing}"

    cfg = {"search": {"database": "pubmed", "query": "x", "email": "a@b.c"},
           "screening": {"abstract": {"objective": "o", "inclusion_criteria": ["c"]},
                         "fulltext": {"objective": "o", "inclusion_criteria": ["c"]}},
           "annotation": {**probe, "annotations": []}}
    f = tmp_path / "c.yml"
    f.write_text(yaml.safe_dump(cfg))
    loaded = ConfigManager().load_from_file(str(f))
    for key, want in probe.items():
        assert getattr(loaded.annotation, key) == want, f"{key} was dropped by the loader"
