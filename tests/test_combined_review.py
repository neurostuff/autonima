"""screening.fulltext.mode: combined -- full-text screening and analysis selection in one call."""

import asyncio
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from autonima.annotation.client import AnnotationClient
from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.schema import AnalysisMetadata, AnnotationCriteriaConfig, StudyAnalysisGroup
from autonima.config import ConfigManager, ConfigurationError
from autonima.documents import FolderRecordSource, attach_documents
from autonima.execution import stage_hashes
from autonima.models.types import Study, StudyStatus
from autonima.screening.combined import CombinedReview, combined_link
from autonima.screening.screener import LLMScreener


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_record(root, pmid):
    text = (f"Record {pmid}\n- {{{{analysis:a}}}}: reappraise > look, whole brain\n"
            f"- {{{{analysis:b}}}}: look > reappraise, region of interest\n")
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{pmid}.md").write_text(text, encoding="utf-8")
    (root / f"{pmid}.analyses.json").write_text(json.dumps({
        "document_sha256": _sha256(text),
        "analyses": [
            {"key": "a", "name": "reappraise > look", "table_id": "t", "points": [{"coordinates": [1, 2, 3]}]},
            {"key": "b", "name": "look > reappraise", "table_id": "t", "points": [{"coordinates": [4, 5, 6]}]},
        ],
        "tables": [{"table_id": "t", "table_label": "Table 1"}],
    }), encoding="utf-8")


CONFIG = {
    "search": {"database": "pubmed", "query": "x", "email": "a@b.c"},
    "screening": {
        "abstract": {"objective": "o", "inclusion_criteria": ["i"], "exclusion_criteria": ["e"]},
        "fulltext": {"objective": "Emotion regulation", "inclusion_criteria": ["Instructed regulation task"],
                     "exclusion_criteria": ["Review article"], "model": "gpt-test", "mode": "combined"},
    },
    "parsing": {"parse_coordinates": False},
    "annotation": {
        "enabled": True, "model": "gpt-annotation", "prompt_type": "multi_analysis",
        "create_all_included_annotations": False,
        "metadata_fields": ["study_title", "study_fulltext", "analysis_name"],
        "annotations": [{"name": "decrease", "inclusion_criteria": ["Regulate > look"]},
                        {"name": "look", "inclusion_criteria": ["Look > regulate"]}],
    },
}


def _config(tmp_path, **changes):
    root = tmp_path / "records"
    for pmid in ("111", "222"):
        _write_record(root, pmid)
    config = deepcopy(CONFIG)
    config["retrieval"] = {"records": {"enabled": True, "kind": "analyses", "root": str(root)}}
    config["output"] = {"directory": str(tmp_path / "run")}
    for path, value in changes.items():
        node = config
        *parents, last = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        if value is None:
            node.pop(last, None)
        else:
            node[last] = value
    return ConfigManager().load_from_dict(config)


def _studies(config):
    studies = [
        Study(pmid=p, title=f"Study {p}", abstract="An abstract.", authors=["A"], journal="J",
              publication_date="2020", status=StudyStatus.INCLUDED_ABSTRACT)
        for p in ("111", "222")
    ]
    attach_documents(studies, FolderRecordSource(config.documents.root), output_dir=config.output.directory)
    return studies


def _decision(name, include):
    return {"annotation_name": name, "include": include, "reasoning": "r", "inclusion_criteria_applied": [],
            "exclusion_criteria_applied": []}


def _answer(prompt):
    """111 is included with both analyses decided; 222 is excluded."""
    if '"study_id": "111"' in prompt:
        return {"study_id": "111",
                "study_screening": {"decision": "INCLUDED", "confidence": 0.9, "reason": "I2 met",
                                    "inclusion_criteria_applied": ["I2"], "exclusion_criteria_applied": []},
                "decisions": [
                    {"analysis_id": "111_analysis_0", "annotations": [_decision("decrease", True), _decision("look", False)]},
                    {"analysis_id": "111_analysis_1", "annotations": [_decision("decrease", False), _decision("look", True)]},
                ]}
    return {"study_id": "222",
            "study_screening": {"decision": "EXCLUDED", "confidence": 0.8, "reason": "E2 met",
                                "inclusion_criteria_applied": [], "exclusion_criteria_applied": ["E2"]},
            "decisions": []}


class FakeCompletions:
    def __init__(self, answer=_answer):
        self.answer, self.calls = answer, []

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        self.calls.append(kwargs)
        message = SimpleNamespace(function_call=SimpleNamespace(arguments=json.dumps(self.answer(prompt))), content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _wire(client, completions):
    client._client = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))


def _screen(config, studies, completions):
    screener = LLMScreener(config.screening, output_dir=config.output.directory)
    combined = CombinedReview(config.annotation, config.screening.fulltext, config.output.directory)
    _wire(combined.client, completions)
    screener.combined = combined
    try:
        results = asyncio.run(screener.screen_fulltexts(studies))
    finally:
        combined.flush()
    return {r.study_id: r for r in results}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_one_call_screens_the_study_and_selects_its_analyses(tmp_path):
    config = _config(tmp_path)
    studies = _studies(config)
    completions = FakeCompletions()

    results = _screen(config, studies, completions)

    assert len(completions.calls) == 2
    assert completions.calls[0]["model"] == "gpt-test"  # the full-text model makes the call
    assert results["111"].decision == StudyStatus.INCLUDED_FULLTEXT
    assert results["222"].decision == StudyStatus.EXCLUDED_FULLTEXT
    assert "combined" in results["111"].cache_signature
    prompt = completions.calls[0]["messages"][-1]["content"]
    assert "FULL-TEXT ELIGIBILITY" in prompt and "Instructed regulation task" in prompt
    assert '"decrease"' in prompt and "111_analysis_1" in prompt
    saved = json.loads((tmp_path / "run" / "outputs" / "annotation_results.json").read_text())
    assert sorted((d["analysis_id"], d["annotation_name"], d["include"]) for d in saved) == [
        ("111_analysis_0", "decrease", True), ("111_analysis_0", "look", False),
        ("111_analysis_1", "decrease", False), ("111_analysis_1", "look", True),
    ]


def test_the_annotation_stage_reads_the_combined_decisions_without_a_call(tmp_path):
    config = _config(tmp_path)
    studies = _studies(config)
    _screen(config, studies, FakeCompletions())
    included = [s for s in studies if s.pmid == "111"]
    included[0].status = StudyStatus.INCLUDED_FULLTEXT

    processor = AnnotationProcessor(
        config.annotation, combined_with=combined_link(config.annotation, config.screening.fulltext)
    )
    _wire(processor.client, FakeCompletions(answer=lambda prompt: pytest.fail("annotation called the model")))
    decisions = processor.process_studies(included_studies=included, output_dir=config.output.directory)

    assert len(decisions) == 4 and processor.cache_stats["reused"] == 4
    assert {(d.analysis_id, d.annotation_name): d.include for d in decisions}[("111_analysis_1", "look")] is True


def test_a_separate_annotation_stage_does_not_reuse_combined_decisions(tmp_path):
    config = _config(tmp_path)
    studies = _studies(config)
    _screen(config, studies, FakeCompletions())
    processor = AnnotationProcessor(config.annotation)  # no combined link: another way of deciding
    studies[0].status = StudyStatus.INCLUDED_FULLTEXT
    existing = processor._load_cached_results(config.output.directory)
    complete = processor._get_annotations_with_complete_results_for_study(
        existing, "111", config.annotation.annotations, studies[0]
    )
    assert complete == set()


def test_a_rerun_reuses_both_halves_and_a_lost_half_is_decided_again(tmp_path):
    config = _config(tmp_path)
    _screen(config, _studies(config), FakeCompletions())

    again = FakeCompletions()
    _screen(config, _studies(config), again)
    assert again.calls == []

    (tmp_path / "run" / "outputs" / "annotation_results.json").unlink()
    after_loss = FakeCompletions()
    results = _screen(config, _studies(config), after_loss)
    # Only the included study had analysis decisions to lose; the exclusion stands.
    assert len(after_loss.calls) == 1 and '"study_id": "111"' in after_loss.calls[0]["messages"][-1]["content"]
    assert results["111"].decision == StudyStatus.INCLUDED_FULLTEXT


def test_in_combined_mode_a_gap_is_a_failure_not_a_separate_call(tmp_path):
    config = _config(tmp_path)
    study = _studies(config)[0]
    study.status = StudyStatus.INCLUDED_FULLTEXT
    processor = AnnotationProcessor(
        config.annotation, combined_with=combined_link(config.annotation, config.screening.fulltext)
    )
    _wire(processor.client, FakeCompletions(answer=lambda prompt: pytest.fail("annotation called the model")))

    decisions = processor.process_studies(included_studies=[study], output_dir=config.output.directory)

    assert len(decisions) == 4 and all(d.include is None and d.failed for d in decisions)


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------

def _group():
    return StudyAnalysisGroup(study_id="111", analyses=[
        AnalysisMetadata(analysis_id=f"111_analysis_{i}", study_id="111", table_id="t", analysis_name=n)
        for i, n in enumerate(("reappraise > look", "look > reappraise"))
    ])


CRITERIA = [AnnotationCriteriaConfig(name="decrease"), AnnotationCriteriaConfig(name="look")]


def _client(answer):
    client = AnnotationClient(max_retries=1)
    _wire(client, FakeCompletions(answer=lambda prompt: answer))
    return client


def test_an_included_study_must_decide_every_analysis():
    answer = _answer('"study_id": "111"')
    answer["decisions"] = answer["decisions"][:1]
    with pytest.raises(ValueError, match="one decision set per analysis"):
        _client(answer).make_combined_decision(_group(), CRITERIA, ["analysis_name"], "block", model="m")


def test_an_excluded_study_has_no_analysis_decisions():
    answer = _answer('"study_id": "111"')
    answer["study_screening"]["decision"] = "EXCLUDED"
    screening, decisions = _client(answer).make_combined_decision(
        _group(), CRITERIA, ["analysis_name"], "block", model="m"
    )
    assert screening["decision"] == "EXCLUDED" and decisions == []


def test_unknown_full_text_criteria_ids_are_refused():
    answer = _answer('"study_id": "111"')
    answer["study_screening"]["inclusion_criteria_applied"] = ["I9"]
    with pytest.raises(ValueError, match="unknown full-text inclusion IDs"):
        _client(answer).make_combined_decision(
            _group(), CRITERIA, ["analysis_name"], "block", model="m", inclusion_ids=["I2"]
        )


def test_a_hallucinated_analysis_is_refused():
    answer = _answer('"study_id": "111"')
    answer["decisions"].append({"analysis_id": "111_analysis_7", "annotations": answer["decisions"][0]["annotations"]})
    with pytest.raises(ValueError, match="hallucinated"):
        _client(answer).make_combined_decision(_group(), CRITERIA, ["analysis_name"], "block", model="m")


# ---------------------------------------------------------------------------
# configuration and caching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("changes, message", [
    ({"retrieval": None}, "needs records that carry their analyses"),
    ({"annotation.prompt_type": "single_analysis"}, "prompt_type must be 'multi_analysis'"),
    ({"annotation.annotations": []}, "define at least one annotation"),
    ({"screening.fulltext.mode": "both"}, "must be 'separate' or 'combined'"),
])
def test_combined_mode_is_checked(tmp_path, changes, message):
    with pytest.raises(ConfigurationError, match=message):
        _config(tmp_path, **changes)


def test_text_records_cannot_be_combined(tmp_path):
    with pytest.raises(ConfigurationError, match="needs records that carry their analyses"):
        _config(tmp_path, **{"retrieval.records.kind": "text"})


def test_in_combined_mode_each_stage_depends_on_the_other(tmp_path):
    base = stage_hashes(_config(tmp_path))
    other_targets = stage_hashes(_config(tmp_path, **{
        "annotation.annotations": [{"name": "decrease", "inclusion_criteria": ["Regulate > look"]}]}))
    other_criteria = stage_hashes(_config(tmp_path, **{"screening.fulltext.inclusion_criteria": ["Other"]}))
    assert base["fulltext"] != other_targets["fulltext"]
    assert base["annotation"] != other_criteria["annotation"]

    separate = stage_hashes(_config(tmp_path, **{"screening.fulltext.mode": None}))
    separate_targets = stage_hashes(_config(tmp_path, **{
        "screening.fulltext.mode": None,
        "annotation.annotations": [{"name": "decrease", "inclusion_criteria": ["Regulate > look"]}]}))
    assert separate["fulltext"] == separate_targets["fulltext"]
    assert separate["fulltext"] != base["fulltext"]


# ---------------------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------------------

def test_combined_mode_through_the_pipeline(tmp_path):
    from unittest.mock import patch

    from autonima.config import get_sample_config_text
    from autonima.pipeline import AutonimaPipeline

    root = tmp_path / "records"
    for pmid in ("111", "222"):
        _write_record(root, pmid)
    raw = yaml.safe_load(get_sample_config_text())
    raw["search"]["pmids_list"] = ["111", "222"]
    raw["output"].update({"directory": str(tmp_path / "run"), "nimads": True})
    raw["parsing"]["parse_coordinates"] = False
    raw["retrieval"]["records"] = {"enabled": True, "kind": "analyses", "root": str(root)}
    raw["screening"]["fulltext"]["mode"] = "combined"
    raw["annotation"].update(deepcopy(CONFIG["annotation"]))
    pipeline = AutonimaPipeline(ConfigManager().load_from_dict(raw))
    pipeline.results.studies = [
        Study(pmid=p, title=f"Study {p}", abstract="An abstract.", authors=["A"], journal="J",
              publication_date="2020", status=StudyStatus.INCLUDED_ABSTRACT)
        for p in ("111", "222")
    ]
    asyncio.run(pipeline._execute_retrieval_phase())
    ids = json.loads((tmp_path / "run" / "outputs" / "criteria_mapping.json").read_text())
    fulltext = ids["screening"]["fulltext"]
    local = {name: next(iter(m["inclusion"])) for name, m in ids["annotation"]["annotations"].items()}

    def answer(prompt):
        def decide(name, include):
            return {"annotation_name": name, "include": include,
                    "reasoning": f"{'Meets' if include else 'Missing'} {local[name]}",
                    "inclusion_criteria_applied": [local[name]] if include else [],
                    "exclusion_criteria_applied": []}
        if '"study_id": "111"' in prompt:
            return {"study_id": "111",
                    "study_screening": {"decision": "INCLUDED", "confidence": 0.9, "reason": "all met",
                                        "inclusion_criteria_applied": list(fulltext["inclusion"]),
                                        "exclusion_criteria_applied": []},
                    "decisions": [
                        {"analysis_id": "111_analysis_0", "annotations": [decide("decrease", True), decide("look", False)]},
                        {"analysis_id": "111_analysis_1", "annotations": [decide("decrease", False), decide("look", True)]},
                    ]}
        return {"study_id": "222",
                "study_screening": {"decision": "EXCLUDED", "confidence": 0.8, "reason": "review",
                                    "inclusion_criteria_applied": [],
                                    "exclusion_criteria_applied": [next(iter(fulltext["exclusion"]))]},
                "decisions": []}

    completions = FakeCompletions(answer)
    model = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    with patch("autonima.annotation.client.GenericLLMClient", return_value=model):
        asyncio.run(pipeline._execute_fulltext_screening())
    assert len(completions.calls) == 2
    assert [s.status for s in pipeline.results.studies] == [
        StudyStatus.INCLUDED_FULLTEXT, StudyStatus.EXCLUDED_FULLTEXT,
    ]

    refuse = FakeCompletions(lambda prompt: pytest.fail("the annotation stage called the model"))
    refusing = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=refuse)))
    with patch("autonima.annotation.client.GenericLLMClient", return_value=refusing):
        asyncio.run(pipeline._execute_annotation_phase())
    asyncio.run(pipeline._generate_nimads_output())

    notes = json.loads((tmp_path / "run" / "outputs" / "nimads_annotation.json").read_text())["notes"]
    by_analysis = {n["analysis"]: n["note"] for n in notes}
    assert by_analysis["111_analysis_0"]["decrease"] is True and by_analysis["111_analysis_0"]["look"] is False
    assert by_analysis["111_analysis_1"]["look"] is True
