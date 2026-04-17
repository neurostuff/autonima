from autonima.annotation.schema import AnnotationDecision
from autonima.coordinates.nimads_models import (
    convert_to_nimads_studyset,
    create_annotations_from_results,
)
from autonima.coordinates.schema import Analysis, CoordinatePoint
from autonima.models.types import Study, StudyStatus


def _make_study(pmid: str, status: StudyStatus) -> Study:
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        abstract="Abstract",
        authors=["Doe"],
        journal="Journal",
        publication_date="2020",
        status=status,
        analyses=[
            Analysis(
                name="contrast",
                description="desc",
                table_id="table_1",
                points=[
                    CoordinatePoint(
                        coordinates=[1.0, 2.0, 3.0],
                        space="MNI",
                        values=[],
                    )
                ],
            )
        ],
    )


def test_create_annotations_from_results_filters_out_non_studyset_analyses():
    included = _make_study("100", StudyStatus.INCLUDED_FULLTEXT)
    excluded = _make_study("200", StudyStatus.EXCLUDED_FULLTEXT)

    # Simulate export_excluded_studies: false by building the studyset from
    # only included studies.
    studyset = convert_to_nimads_studyset("studyset_1", [included])

    annotation_results = [
        AnnotationDecision(
            annotation_name="risky_dm",
            analysis_id="100_analysis_0",
            study_id="100",
            include=True,
            reasoning="included study analysis",
            model_used="test-model",
        ),
        AnnotationDecision(
            annotation_name="risky_dm",
            analysis_id="200_analysis_0",
            study_id="200",
            include=True,
            reasoning="excluded study analysis",
            model_used="test-model",
        ),
    ]

    annotation = create_annotations_from_results(
        "studyset_1", studyset, annotation_results
    )

    exported_analysis_ids = {note.analysis_id for note in annotation.notes}
    assert exported_analysis_ids == {"100_analysis_0"}
    assert annotation.notes[0].note["risky_dm"] is True
