"""Document sources: the contract, the two folder kinds, and attaching them to studies."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from autonima.documents import (
    DocumentFormatError,
    DocumentSource,
    FolderDocumentSource,
    FolderRecordSource,
    IdentityError,
    PairingError,
    attach_documents,
)
from autonima.documents.source import substitute_analysis_ids
from autonima.models.types import ActivationTable, Study, StudyStatus
from autonima.coordinates.schema import Analysis, CoordinatePoint


RECORD_TEXT = """# Record for 12345

## Analyses
- {{analysis:t2#1}}: PTSD > controls, whole brain
- {{analysis:t2#2}}: controls > PTSD, the withheld half of {{analysis:t2#1}}
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_record(root: Path, pmid: str = "12345", text: str = RECORD_TEXT, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{pmid}.md").write_text(text, encoding="utf-8")
    payload = {
        "document_sha256": _sha256(text),
        "space": "MNI",
        "analyses": [
            {
                "key": "t2#1",
                "name": "PTSD > controls",
                "description": "Whole-brain group contrast",
                "table_id": "t2",
                "points": [
                    {"coordinates": [1, 2, 3], "values": [{"value": 3.1, "kind": "t-statistic"}]}
                ],
                "document": {"effect": {"kind": "contrast"}, "spatial_scope": "whole_brain"},
            },
            {
                "key": "t2#2",
                "name": "controls > PTSD",
                "table_id": "t2",
                "points": [{"coordinates": [4, 5, 6], "space": "TAL"}],
                "document": "Sign split of {{analysis:t2#1}}.",
            },
        ],
        "tables": [{"table_id": "t2", "table_label": "Table 2", "table_caption": "Regions"}],
    }
    payload.update(overrides)
    (root / f"{pmid}.analyses.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _study(pmid: str = "12345", **kwargs) -> Study:
    return Study(
        pmid=pmid,
        title="A study",
        abstract="An abstract.",
        authors=["A. Author"],
        journal="J",
        publication_date="2020",
        status=StudyStatus.INCLUDED_ABSTRACT,
        **kwargs,
    )


def _article_study(pmid: str, tmp_path: Path) -> Study:
    """A study as article retrieval leaves it: a text file, a table, ingested coordinates."""
    article = tmp_path / "articles" / f"{pmid}.txt"
    article.parent.mkdir(parents=True, exist_ok=True)
    article.write_text("The article.", encoding="utf-8")
    return _study(
        pmid,
        full_text_path=str(article),
        fulltext_available=True,
        activation_tables=[ActivationTable(table_id="1", table_label="Table 1")],
        analyses=[
            Analysis(name="1", points=[CoordinatePoint(coordinates=[0, 0, 0])], table_id="1")
        ],
    )


def _unavailable_rows(output_dir: Path):
    with open(output_dir / "outputs" / "documents_unavailable.csv", newline="") as stream:
        return list(csv.DictReader(stream))


# -- the contract ---------------------------------------------------------------------------


def test_capability_cannot_be_declared_without_its_method():
    with pytest.raises(TypeError):
        class ClaimsAnalyses(DocumentSource):  # noqa: F841
            kind = "bad"
            provides_analyses = True

    with pytest.raises(TypeError):
        class HidesAnalyses(DocumentSource):  # noqa: F841
            kind = "bad"
            provides_analyses = False

            def open_analyses(self, ref):
                return None


def test_substitution_rejects_references_the_source_does_not_supply():
    assert substitute_analysis_ids(
        "see {{analysis:a}} and {{analysis: b }}", {"a": "1_analysis_0", "b": "1_analysis_1"},
        where="doc",
    ) == "see 1_analysis_0 and 1_analysis_1"
    with pytest.raises(PairingError, match="missing"):
        substitute_analysis_ids("{{analysis:missing}}", {"a": "x"}, where="doc")


def test_host_id_map_is_positional_and_detects_a_length_change(tmp_path):
    _write_record(tmp_path)
    source = FolderRecordSource(tmp_path)
    source.prepare({12345})
    bundle = source.open_analyses(source.index()[12345])

    assert bundle.host_id_map(["h0", "h1"]) == {"t2#1": "h0", "t2#2": "h1"}
    with pytest.raises(IdentityError):
        bundle.host_id_map(["h0", "h1", "h2"])


# -- the text kind --------------------------------------------------------------------------


def test_text_source_indexes_documents_by_pmid(tmp_path):
    (tmp_path / "111.md").write_text("summary one", encoding="utf-8")
    (tmp_path / "222.json").write_text('{"summary": "two"}', encoding="utf-8")
    (tmp_path / "222.analyses.json").write_text("{}", encoding="utf-8")  # not a document
    (tmp_path / "notes.md").write_text("not a study", encoding="utf-8")

    source = FolderDocumentSource(tmp_path)
    report = source.prepare({111, 222, 333})

    assert sorted(source.index()) == [111, 222]
    assert (report.requested, report.ready) == (3, 2)
    assert [(u.pmid, u.reason) for u in report.unavailable] == [(333, "no_document")]
    document = source.open(source.index()[222])
    assert document.text == '{"summary": "two"}'
    assert document.content_hash == _sha256('{"summary": "two"}')
    assert "extraction record" in document.description  # the default


def test_text_source_refuses_two_documents_for_one_pmid(tmp_path):
    (tmp_path / "111.md").write_text("a", encoding="utf-8")
    (tmp_path / "111.txt").write_text("b", encoding="utf-8")
    with pytest.raises(IdentityError, match="111.md and 111.txt"):
        FolderDocumentSource(tmp_path).prepare({111})


def test_empty_document_is_rejected(tmp_path):
    (tmp_path / "111.md").write_text("  \n", encoding="utf-8")
    source = FolderDocumentSource(tmp_path)
    source.prepare({111})
    with pytest.raises(DocumentFormatError, match="empty"):
        source.open(source.index()[111])


def test_attach_text_documents(tmp_path):
    root = tmp_path / "summaries"
    root.mkdir()
    (root / "111.md").write_text("# Summary\nAn fMRI study.", encoding="utf-8")
    output_dir = tmp_path / "run"
    study = _article_study("111", tmp_path)
    tables, analyses = list(study.activation_tables), list(study.analyses)

    report = attach_documents(
        [study], FolderDocumentSource(root, description="a summary"), output_dir=output_dir
    )

    assert report.attached == 1 and report.unavailable == []
    assert study.full_text_path == str(output_dir / "documents" / "111.md")
    assert study.full_text_source == "document:text"
    assert study.full_text == "# Summary\nAn fMRI study."
    # A text source says nothing about analyses, so the article's stay for parsing.
    assert study.activation_tables == tables and study.analyses == analyses
    serialized = study.to_dict()["document"]
    assert serialized["kind"] == "text"
    assert serialized["description"] == "a summary"
    assert serialized["content_hash"] == _sha256("# Summary\nAn fMRI study.")
    assert "An fMRI study" not in json.dumps(serialized)  # the ref, never the body
    assert _unavailable_rows(output_dir) == []


def test_text_document_with_analysis_references_is_refused(tmp_path):
    root = tmp_path / "summaries"
    root.mkdir()
    (root / "111.md").write_text("see {{analysis:t1#1}}", encoding="utf-8")
    study = _article_study("111", tmp_path)

    report = attach_documents([study], FolderDocumentSource(root), output_dir=tmp_path / "run")

    assert report.attached == 0
    assert [u.reason for u in report.unavailable] == ["pairing_mismatch"]
    assert study.fulltext_available is False


def test_missing_document_leaves_the_study_unretrieved(tmp_path):
    root = tmp_path / "summaries"
    root.mkdir()
    output_dir = tmp_path / "run"
    study = _article_study("111", tmp_path)

    report = attach_documents([study], FolderDocumentSource(root), output_dir=output_dir)

    assert [u.reason for u in report.unavailable] == ["no_document"]
    # Never falls back to the article: it is not screened, and nothing from it survives.
    assert study.fulltext_available is False
    assert study.full_text_path is None
    assert study.full_text_source == "document:text"
    assert study.document is None
    assert study.analyses == [] and study.activation_tables == []
    rows = _unavailable_rows(output_dir)
    assert [(row["pmid"], row["kind"], row["reason"]) for row in rows] == [
        ("111", "text", "no_document")
    ]


def test_unchanged_document_is_not_rewritten(tmp_path):
    root = tmp_path / "summaries"
    root.mkdir()
    (root / "111.md").write_text("summary", encoding="utf-8")
    source = FolderDocumentSource(root)
    first = _study("111")
    attach_documents([first], source, output_dir=tmp_path / "run")
    mtime = Path(first.full_text_path).stat().st_mtime_ns

    second = _study("111")
    attach_documents([second], source, output_dir=tmp_path / "run")

    assert Path(second.full_text_path).stat().st_mtime_ns == mtime


# -- the records kind -----------------------------------------------------------------------


def test_attach_records_transports_analyses_under_host_ids(tmp_path):
    root = tmp_path / "records"
    _write_record(root)
    study = _study("12345")

    report = attach_documents([study], FolderRecordSource(root), output_dir=tmp_path / "run")

    assert report.attached == 1
    assert [a.name for a in study.analyses] == ["PTSD > controls", "controls > PTSD"]
    assert all(a.parsed is False for a in study.analyses)
    assert [p.space for a in study.analyses for p in a.points] == ["MNI", "TAL"]
    assert study.analyses[0].points[0].values[0].kind == "t-statistic"
    assert study.coordinate_space == "MNI"
    assert [(t.table_id, t.table_caption) for t in study.activation_tables] == [("t2", "Regions")]

    # The model reads one id space: autonima's, never the source's keys.
    text = study.full_text
    assert "12345_analysis_0: PTSD > controls" in text
    assert "the withheld half of 12345_analysis_0" in text
    assert "{{analysis:" not in text and "t2#" not in text

    document = study.document
    assert document.provides_analyses is True
    assert document.analysis_keys == ["t2#1", "t2#2"]
    assert document.analysis_ids == ["12345_analysis_0", "12345_analysis_1"]
    assert json.loads(document.analysis_documents[0])["spatial_scope"] == "whole_brain"
    assert document.analysis_documents[1] == "Sign split of 12345_analysis_0."
    assert study.to_dict()["document"]["analyses"] == [
        {"key": "t2#1", "analysis_id": "12345_analysis_0"},
        {"key": "t2#2", "analysis_id": "12345_analysis_1"},
    ]


def test_records_replace_whatever_the_article_left(tmp_path):
    root = tmp_path / "records"
    _write_record(root)
    study = _article_study("12345", tmp_path)

    attach_documents([study], FolderRecordSource(root), output_dir=tmp_path / "run")

    assert [a.table_id for a in study.analyses] == ["t2", "t2"]
    assert [t.table_id for t in study.activation_tables] == ["t2"]


def test_undeclared_tables_get_a_bare_entry(tmp_path):
    root = tmp_path / "records"
    payload = _write_record(root, tables=[])
    assert payload["tables"] == []
    study = _study("12345")

    attach_documents([study], FolderRecordSource(root), output_dir=tmp_path / "run")

    assert [(t.table_id, t.table_label) for t in study.activation_tables] == [("t2", "t2")]


@pytest.mark.parametrize(
    "text, overrides, reason, message",
    [
        (RECORD_TEXT, {"document_sha256": "0" * 64}, "pairing_mismatch", "hashes to"),
        (RECORD_TEXT, {"document_sha256": None}, "pairing_mismatch", "document_sha256"),
        (
            RECORD_TEXT,
            {"analyses": [{"key": "t2#1", "points": []}]},
            "pairing_mismatch",
            "t2#2",  # referenced by the document, not supplied
        ),
        (
            RECORD_TEXT,
            {"analyses": [{"key": "a", "points": []}, {"key": "a", "points": []}]},
            "identity_conflict",
            "key 'a'",
        ),
        (
            "A record that references no analyses.",
            {"analyses": [{"key": "t2#1", "points": [{"coordinates": [1, 2]}]}]},
            "invalid_document",
            "invalid point",
        ),
    ],
)
def test_records_that_do_not_pair_are_unavailable(tmp_path, text, overrides, reason, message):
    root = tmp_path / "records"
    _write_record(root, text=text, **overrides)
    study = _study("12345")

    report = attach_documents([study], FolderRecordSource(root), output_dir=tmp_path / "run")

    assert report.attached == 0
    assert [u.reason for u in report.unavailable] == [reason]
    assert message in report.unavailable[0].detail
    assert study.fulltext_available is False and study.analyses == []


def test_record_without_analyses_file_is_unavailable(tmp_path):
    root = tmp_path / "records"
    root.mkdir()
    (root / "12345.md").write_text("a record", encoding="utf-8")
    study = _study("12345")

    report = attach_documents([study], FolderRecordSource(root), output_dir=tmp_path / "run")

    assert [(u.reason, u.study_id) for u in report.unavailable] == [("pairing_mismatch", "12345")]
    assert "12345.analyses.json" in report.unavailable[0].detail
