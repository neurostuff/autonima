"""`autonima.meta` runs a meta-analysis per annotation column on the pipeline's NiMADS outputs."""

import json

import pytest

pytest.importorskip("nimare")

from autonima import meta
from autonima.annotation.schema import AnnotationDecision
from autonima.coordinates.nimads_models import (
    convert_to_nimads_studyset,
    create_annotations_from_results,
    sanitize_annotation_dict,
    sanitize_studyset_dict,
)
from autonima.coordinates.schema import Analysis, CoordinatePoint
from autonima.models.types import Study, StudyStatus

PMIDS = ["101", "102", "103", "104", "105", "106"]

# Decisions on column "target" for <pmid>_analysis_1. Every <pmid>_analysis_0 is included.
# 105's is left undecided (annotation failed), and 106's analysis has no coordinates.
ANALYSIS_1_DECISIONS = {
    "101": True,
    "102": True,
    "103": False,
    "104": False,
    "106": True,
}


def _points(seed):
    return [
        CoordinatePoint(
            coordinates=[
                float((seed * 7 + i * 11) % 50 - 25),
                float(i * 9 - 20),
                float(seed - 10),
            ],
            space="MNI",
            values=[],
        )
        for i in range(5)
    ]


def _write_outputs(folder):
    """Write nimads_studyset.json and nimads_annotation.json the way the pipeline does."""
    studies = [
        Study(
            pmid=pmid,
            title=f"Study {pmid}",
            abstract="Abstract",
            authors=["Doe"],
            journal="Journal",
            publication_date="2020",
            status=StudyStatus.INCLUDED_FULLTEXT,
            analyses=[
                Analysis(name="contrast 0", description="d", points=_points(n)),
                Analysis(
                    name="contrast 1",
                    description="d",
                    points=[] if pmid == "106" else _points(n + 20),
                ),
            ],
        )
        for n, pmid in enumerate(PMIDS)
    ]
    decisions = [
        AnnotationDecision(
            annotation_name="target",
            analysis_id=analysis_id,
            study_id=pmid,
            include=include,
            reasoning="r",
            model_used="m",
        )
        for pmid in PMIDS
        for analysis_id, include in [
            (f"{pmid}_analysis_0", True),
            (f"{pmid}_analysis_1", ANALYSIS_1_DECISIONS.get(pmid)),
        ]
        if include is not None
    ]
    studyset = convert_to_nimads_studyset(
        "ss", studies, name="Autonima Generated Studyset"
    )
    annotation = create_annotations_from_results(
        "ss", studyset, decisions, unknown_when_missing={"target": set(PMIDS)}
    )
    folder.mkdir(parents=True)
    (folder / "nimads_studyset.json").write_text(
        json.dumps(sanitize_studyset_dict(studyset.to_dict()))
    )
    (folder / "nimads_annotation.json").write_text(
        json.dumps(sanitize_annotation_dict(annotation.to_dict()))
    )
    return folder


def _fitted_ids(result):
    return sorted(str(i) for i in result.estimator.dataset.ids)


def test_meta_analysis_fits_included_analyses_that_have_coordinates(tmp_path):
    outputs = _write_outputs(tmp_path / "outputs")

    results = meta.run_meta_analyses(outputs, fail_fast=True)

    assert list(results) == ["target"]
    assert _fitted_ids(results["target"]) == [
        "101-101_analysis_0",
        "101-101_analysis_1",
        "102-102_analysis_0",
        "102-102_analysis_1",
        "103-103_analysis_0",
        "104-104_analysis_0",
        "105-105_analysis_0",
        "106-106_analysis_0",
    ]
    column_dir = outputs / "meta_analysis_results" / "target"
    assert (column_dir / "z_corr-FDR_method-indep.nii.gz").exists()


def test_meta_analysis_keeps_only_include_ids_studies(tmp_path):
    outputs = _write_outputs(tmp_path / "outputs")

    results = meta.run_meta_analyses(
        outputs, include_ids={"101", "103"}, fail_fast=True
    )

    assert _fitted_ids(results["target"]) == [
        "101-101_analysis_0",
        "101-101_analysis_1",
        "103-103_analysis_0",
    ]
