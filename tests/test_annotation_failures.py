"""Failed annotation calls must be recorded, retried and exported as unknown -- never as False."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autonima.annotation.client import AnnotationClient
from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.schema import (
    AnnotationConfig,
    AnnotationCriteriaConfig,
    AnnotationDecision,
)
from autonima.coordinates.nimads_models import (
    convert_to_nimads_studyset,
    create_annotations_from_results,
)
from autonima.coordinates.schema import Analysis
from autonima.meta import _unknown_analysis_count
from autonima.models.types import Study, StudyStatus


def _study(pmid, n_analyses=2, status=StudyStatus.INCLUDED_FULLTEXT):
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        abstract="Abstract",
        authors=["Doe"],
        journal="Journal",
        publication_date="2020",
        status=status,
        analyses=[
            Analysis(name=f"contrast {i}", description="desc", points=[])
            for i in range(n_analyses)
        ],
    )


def _config(names=("A", "B")):
    return AnnotationConfig(
        create_all_included_annotations=False,
        metadata_fields=["analysis_name"],
        annotations=[
            AnnotationCriteriaConfig(name=name, inclusion_criteria=[f"{name} criterion"])
            for name in names
        ],
    )


class FakeClient:
    """Stands in for AnnotationClient: answers include=True, or fails as told."""

    def __init__(self, fail_studies=(), omit=()):
        self.fail_studies = set(fail_studies)
        self.omit = set(omit)  # (analysis_id, annotation_name) pairs to leave out
        self.calls = []

    def make_decision(self, group, criteria, metadata_fields, model=None,
                      model_params=None, prompt_type=None):
        self.calls.append((group.study_id, [c.name for c in criteria]))
        if group.study_id in self.fail_studies:
            raise RuntimeError("Error code: 400 - context_length_exceeded")
        return [
            AnnotationDecision(
                annotation_name=c.name,
                analysis_id=a.analysis_id,
                study_id=group.study_id,
                include=True,
                reasoning="meets criteria",
                model_used=model or "m",
            )
            for a in group.analyses
            for c in criteria
            if (a.analysis_id, c.name) not in self.omit
        ]


def _processor(client, config=None, num_workers=1):
    processor = AnnotationProcessor(config or _config(), num_workers=num_workers)
    processor.client = client
    return processor


def _on_disk(output_dir):
    path = Path(output_dir) / "outputs" / "annotation_results.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("num_workers", [1, 2])
def test_a_failed_study_is_recorded_not_dropped(tmp_path, num_workers):
    client = FakeClient(fail_studies={"2"})
    processor = _processor(client, num_workers=num_workers)

    results = processor.process_studies([_study("1"), _study("2")], output_dir=str(tmp_path))

    failed = [r for r in results if r.study_id == "2"]
    assert len(failed) == 4  # 2 analyses x 2 annotations, none silently missing
    assert all(r.include is None and r.failed for r in failed)
    assert all("context_length_exceeded" in r.error for r in failed)
    assert all(r.cache_signature for r in failed)
    assert all(r.include is True for r in results if r.study_id == "1")
    assert processor.cache_stats["failed"] == 4
    assert processor.cache_stats["failed_studies"] == 1

    rows = [row for row in _on_disk(tmp_path) if row["study_id"] == "2"]
    assert len(rows) == 4 and all(row["include"] is None for row in rows)


def test_a_failed_study_is_retried_on_the_next_run(tmp_path):
    studies = [_study("1"), _study("2")]
    first = FakeClient(fail_studies={"2"})
    _processor(first).process_studies(studies, output_dir=str(tmp_path))

    second = FakeClient()
    processor = _processor(second)
    results = processor.process_studies(studies, output_dir=str(tmp_path))

    # Only the failed study is re-billed; the good one is reused from cache.
    assert [study_id for study_id, _ in second.calls] == ["2"]
    assert all(r.include is True for r in results)
    assert processor.cache_stats["failed"] == 0
    assert not any(row["include"] is None for row in _on_disk(tmp_path))


def test_pairs_left_out_of_a_response_become_failures(tmp_path):
    client = FakeClient(omit={("1_analysis_1", "B")})
    processor = _processor(client)

    results = processor.process_studies([_study("1")], output_dir=str(tmp_path))

    by_pair = {(r.analysis_id, r.annotation_name): r for r in results}
    assert len(by_pair) == 4
    assert by_pair[("1_analysis_1", "B")].failed
    assert "no decision" in by_pair[("1_analysis_1", "B")].error
    assert all(
        not r.failed for key, r in by_pair.items() if key != ("1_analysis_1", "B")
    )


def test_a_partial_gap_reprocesses_every_annotation_and_keeps_them_all(tmp_path):
    """Regression: re-annotating only B used to delete A, and the study flip-flopped."""
    study = [_study("1")]
    _processor(FakeClient(omit={("1_analysis_1", "B")})).process_studies(
        study, output_dir=str(tmp_path))

    for _ in range(3):
        client = FakeClient()
        results = _processor(client).process_studies(study, output_dir=str(tmp_path))
        names = {r.annotation_name for r in results if not r.failed}
        assert names == {"A", "B"}
        assert len([r for r in results if not r.failed]) == 4

    # Once complete, later runs make no calls at all.
    assert client.calls == []


def test_export_distinguishes_unknown_from_excluded():
    included = _study("100", n_analyses=2)
    excluded = _study("200", n_analyses=1, status=StudyStatus.EXCLUDED_FULLTEXT)
    studyset = convert_to_nimads_studyset("ss", [included, excluded])
    results = [
        AnnotationDecision(annotation_name="A", analysis_id="100_analysis_0",
                           study_id="100", include=False, reasoning="r", model_used="m"),
        AnnotationDecision(annotation_name="A", analysis_id="100_analysis_1",
                           study_id="100", include=None, reasoning="failed",
                           model_used="m", error="boom"),
        AnnotationDecision(annotation_name="B", analysis_id="100_analysis_0",
                           study_id="100", include=True, reasoning="r", model_used="m"),
        # B has no row at all for 100_analysis_1 -- the pre-fix shape of a failed call.
    ]

    annotation = create_annotations_from_results(
        "ss", studyset, results, unknown_when_missing={"A": {"100"}, "B": {"100"}})
    notes = {note.analysis_id: note.note for note in annotation.notes}

    assert notes["100_analysis_0"] == {"A": False, "B": True}  # a real exclusion stays False
    assert notes["100_analysis_1"] == {"A": None, "B": None}   # failed / never decided
    assert notes["200_analysis_0"] == {"A": False, "B": False}  # out of scope: not applicable

    exported = annotation.to_dict()
    assert _unknown_analysis_count(exported, "A") == 1
    assert _unknown_analysis_count(exported, "B") == 1


def test_export_without_scope_keeps_the_old_default():
    studyset = convert_to_nimads_studyset("ss", [_study("100", n_analyses=2)])
    results = [
        AnnotationDecision(annotation_name="A", analysis_id="100_analysis_0",
                           study_id="100", include=True, reasoning="r", model_used="m"),
    ]
    annotation = create_annotations_from_results("ss", studyset, results)
    notes = {note.analysis_id: note.note for note in annotation.notes}
    assert notes["100_analysis_1"] == {"A": False}


def test_make_decision_forwards_model_params():
    client = AnnotationClient.__new__(AnnotationClient)
    client.max_retries = 1
    with patch.object(AnnotationClient, "_make_decision_attempt", return_value=[]) as attempt:
        client.make_decision(
            object(), [AnnotationCriteriaConfig(name="A")], ["analysis_name"],
            model="m", model_params={"reasoning_effort": "low"},
        )
    assert attempt.call_args.kwargs["model_params"] == {"reasoning_effort": "low"}
