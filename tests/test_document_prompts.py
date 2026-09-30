"""Prompts and cache keys for studies read from a document instead of the article.

The article side is pinned by digests recorded before documents existed: a study without a
document must get byte-identical prompts and identical cache signatures, or every article
run would re-screen and re-annotate on upgrade.
"""

import hashlib
import json

import pytest

from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.prompts import (
    create_single_study_annotation_prompt,
    create_study_multi_annotation_prompt,
)
from autonima.annotation.schema import AnnotationConfig, AnnotationCriteriaConfig
from autonima.coordinates.schema import Analysis, CoordinatePoint
from autonima.documents import (
    FolderDocumentSource,
    FolderRecordSource,
    IdentityError,
    attach_documents,
)
from autonima.execution import coordinate_study_input_hash
from autonima.models.types import ActivationTable, ScreeningConfig, Study, StudyStatus
from autonima.screening.prompts import PromptLibrary
from autonima.screening.screener import LLMScreener

ARTICLE_DIGESTS = {
    "fulltext_prompt_conf_False": "01198febf5b5155c7e66a0ca68a48eb8b7cb6a137151253f20c8c31c8f1b832b",
    "fulltext_prompt_conf_True": "e545fcc18d0d0f9b5b3b69b777cde096e9a604ddea92798ba706c02c5dd30bcc",
    "annotation_multi_prompt": "2077fcc5b7971fcb81cdb007e3485434c927ceb9fcfac017764a4c413e83081a",
    "annotation_single_prompt": "805f8b827ea192b98f5f733f662829dc446137857cd2cc1709f27ed64763c1d3",
    "annotation_study_input_hash": "a84a68c0ba090c9bae536899dceaf5cfab17b85ce55a7f751be7932e818899b0",
    "coordinate_study_input_hash": "648483a299a6f32d369b16bb0e3562906cfe1acb1715025deadae3d1221542a7",
    "screening_study_input_hash": "398ab081bc158611af85818b7c347562b30fb1f0f2effeceb63ab9bf2714d57a",
}
FIELDS = ["analysis_name", "analysis_description", "table_caption", "study_title", "study_fulltext"]
CRITERIA = [
    AnnotationCriteriaConfig(
        name="wm",
        description="Working memory",
        inclusion_criteria=["WM task"],
        exclusion_criteria=["Rest"],
    )
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _article_study(tmp_path) -> Study:
    text_file = tmp_path / "12345.txt"
    text_file.write_text("Introduction. Methods. Results. Discussion.", encoding="utf-8")
    return Study(
        pmid="12345",
        title="A study",
        abstract="An abstract.",
        authors=["A. Author", "B. Author"],
        journal="J Neuro",
        publication_date="2020",
        doi="10.1/x",
        status=StudyStatus.INCLUDED_ABSTRACT,
        full_text_path=str(text_file),
        fulltext_available=True,
        full_text_output_dir=str(tmp_path),
        activation_tables=[
            ActivationTable(table_id="t1", table_label="Table 1", table_caption="Cap", table_foot="Foot")
        ],
        analyses=[
            Analysis(
                name="A > B",
                description="desc",
                points=[CoordinatePoint(coordinates=[1, 2, 3], space="MNI")],
                parsed=False,
                table_id="t1",
            ),
            Analysis(name="B > A", description=None, points=[], parsed=True, table_id="t1"),
        ],
    )


def _fulltext_prompt(study: Study, tmp_path, confidence_reporting: bool = False) -> str:
    return PromptLibrary.get_fulltext_screening_prompt(
        study=study,
        inclusion_criteria=["Human participants", "fMRI"],
        exclusion_criteria=["Animal studies"],
        output_dir=str(tmp_path),
        objective="Find studies",
        confidence_reporting=confidence_reporting,
        additional_instructions="Be careful.",
    )


def _text_document_study(tmp_path, description="a structured extraction record") -> Study:
    root = tmp_path / "documents_source"
    root.mkdir(exist_ok=True)
    (root / "12345.md").write_text("## Design\nCase-control fMRI.", encoding="utf-8")
    study = _article_study(tmp_path)
    attach_documents(
        [study], FolderDocumentSource(root, description=description), output_dir=tmp_path / "run"
    )
    return study


def _record_study(tmp_path) -> Study:
    root = tmp_path / "records"
    root.mkdir()
    text = "Analyses: {{analysis:k1}}, {{analysis:k2}}"
    (root / "12345.md").write_text(text, encoding="utf-8")
    (root / "12345.analyses.json").write_text(
        json.dumps(
            {
                "document_sha256": _sha(text),
                "analyses": [
                    {"key": "k1", "name": "A > B", "table_id": "t1", "document": "effect: contrast"},
                    {"key": "k2", "name": "B > A", "table_id": "t1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    study = _article_study(tmp_path)
    attach_documents(
        [study], FolderRecordSource(root, description="an extraction record"),
        output_dir=tmp_path / "run",
    )
    return study


# -- article prompts and hashes are unchanged -----------------------------------------------


@pytest.mark.parametrize("confidence_reporting", [False, True])
def test_article_fulltext_prompt_is_byte_identical(tmp_path, confidence_reporting):
    prompt = _fulltext_prompt(_article_study(tmp_path), tmp_path, confidence_reporting)
    assert _sha(prompt) == ARTICLE_DIGESTS[f"fulltext_prompt_conf_{confidence_reporting}"]


def test_article_annotation_prompts_and_hashes_are_unchanged(tmp_path):
    processor = AnnotationProcessor(AnnotationConfig(annotations=CRITERIA, metadata_fields=FIELDS))
    study = _article_study(tmp_path)
    group = processor._build_study_analysis_group(study, FIELDS)
    metadata = processor._extract_analysis_metadata(
        study, study.analyses[0], "12345_analysis_0", FIELDS, analysis_index=0
    )

    assert _sha(create_study_multi_annotation_prompt(group, CRITERIA, FIELDS)) == (
        ARTICLE_DIGESTS["annotation_multi_prompt"]
    )
    assert _sha(create_single_study_annotation_prompt(metadata, CRITERIA, FIELDS)) == (
        ARTICLE_DIGESTS["annotation_single_prompt"]
    )
    assert processor._study_input_hash(study) == ARTICLE_DIGESTS["annotation_study_input_hash"]
    assert coordinate_study_input_hash(study) == ARTICLE_DIGESTS["coordinate_study_input_hash"]


def test_article_screening_signature_is_unchanged(tmp_path):
    screener = LLMScreener(ScreeningConfig(), output_dir=str(tmp_path / "out"))
    assert screener._study_input_hash(_article_study(tmp_path), "fulltext") == (
        ARTICLE_DIGESTS["screening_study_input_hash"]
    )


# -- document prompts -----------------------------------------------------------------------


def test_document_fulltext_prompt_names_the_document_not_the_article(tmp_path):
    prompt = _fulltext_prompt(_text_document_study(tmp_path), tmp_path)

    assert "Study Document: ## Design\nCase-control fMRI." in prompt
    assert "Document: a structured extraction record" in prompt
    assert "is not the article's full text" in prompt
    assert "Full Text Content" not in prompt
    # Instruction 8 no longer asks for article sections, which a document never has.
    assert "introduction/background, methods, results, and discussion" not in prompt
    assert 'provided "full text"' not in prompt
    assert "Set fulltext_incomplete=true ONLY when the study document itself" in prompt
    # Everything the article prompt asks for in the response is still asked for.
    assert "Be careful." in prompt
    assert "inclusion_criteria_applied" in prompt and "fulltext_incomplete" in prompt


def test_document_confidence_numbering_follows_instruction_8(tmp_path):
    prompt = _fulltext_prompt(_text_document_study(tmp_path), tmp_path, confidence_reporting=True)
    assert "9. Provide a confidence score" in prompt
    assert "10. Give a detailed reason" in prompt


def test_document_screening_signature_tracks_the_description(tmp_path):
    screener = LLMScreener(ScreeningConfig(), output_dir=str(tmp_path / "out"))
    article = screener._study_input_hash(_article_study(tmp_path), "fulltext")
    summary = screener._study_input_hash(_text_document_study(tmp_path, "a summary"), "fulltext")
    record = screener._study_input_hash(_text_document_study(tmp_path, "a record"), "fulltext")
    assert len({article, summary, record}) == 3


def test_document_annotation_prompts(tmp_path):
    fields = FIELDS + ["analysis_document"]
    processor = AnnotationProcessor(AnnotationConfig(annotations=CRITERIA, metadata_fields=fields))
    study = _record_study(tmp_path)

    multi = create_study_multi_annotation_prompt(
        processor._build_study_analysis_group(study, fields), CRITERIA, fields
    )
    assert (
        "Study Document (an extraction record; in place of the full text): "
        "Analyses: 12345_analysis_0, 12345_analysis_1"
    ) in multi
    assert "Study Full Text" not in multi
    assert "Analysis ID: 12345_analysis_0\n- Name: A > B\n- Analysis Document: effect: contrast" in multi
    assert "Analysis ID: 12345_analysis_1\n- Name: B > A\n" in multi  # no record, no line

    single = create_single_study_annotation_prompt(
        processor._extract_analysis_metadata(
            study, study.analyses[0], "12345_analysis_0", fields, analysis_index=0
        ),
        CRITERIA,
        fields,
    )
    assert "- Study Document (an extraction record; in place of the full text):" in single
    assert "- Analysis Document: effect: contrast" in single


def test_analysis_documents_need_their_metadata_field(tmp_path):
    processor = AnnotationProcessor(AnnotationConfig(annotations=CRITERIA, metadata_fields=FIELDS))
    study = _record_study(tmp_path)
    group = processor._build_study_analysis_group(study, FIELDS)
    assert "Analysis Document" not in create_study_multi_annotation_prompt(group, CRITERIA, FIELDS)


def test_annotation_hash_changes_with_the_analysis_documents(tmp_path):
    fields = FIELDS + ["analysis_document"]
    processor = AnnotationProcessor(AnnotationConfig(annotations=CRITERIA, metadata_fields=fields))
    study = _record_study(tmp_path)
    before = processor._study_input_hash(study)
    study.document.analysis_documents[0] = "effect: something else"
    assert processor._study_input_hash(study) != before


def test_reparsed_analyses_are_detected_before_annotation(tmp_path):
    fields = FIELDS + ["analysis_document"]
    processor = AnnotationProcessor(AnnotationConfig(annotations=CRITERIA, metadata_fields=fields))
    study = _record_study(tmp_path)
    study.analyses.append(Analysis(name="extra", points=[], table_id="t1"))
    with pytest.raises(IdentityError, match="changed after the documents were attached"):
        processor._build_study_analysis_group(study, fields)
