"""Documents through the pipeline: retrieval, screening, parsing, and the run's outputs."""

import asyncio
import csv
import hashlib
import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import yaml

from autonima.config import ConfigManager, get_sample_config_text
from autonima.models.types import Study, StudyStatus
from autonima.pipeline import AutonimaPipeline

RECORD_TEXT = "Record.\n- {{analysis:t1#1}}: faces > shapes\n- {{analysis:t1#2}}: shapes > faces\n"


def _write_record(root, pmid="111"):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{pmid}.md").write_text(RECORD_TEXT, encoding="utf-8")
    (root / f"{pmid}.analyses.json").write_text(
        json.dumps(
            {
                "document_sha256": hashlib.sha256(RECORD_TEXT.encode("utf-8")).hexdigest(),
                "space": "MNI",
                "analyses": [
                    {"key": "t1#1", "name": "faces > shapes", "table_id": "t1",
                     "points": [{"coordinates": [10, 20, 30]}]},
                    {"key": "t1#2", "name": "shapes > faces", "table_id": "t1",
                     "points": [{"coordinates": [-10, -20, -30]}]},
                ],
            }
        ),
        encoding="utf-8",
    )


def _pipeline(tmp_path, kind, root, **overrides):
    config = yaml.safe_load(get_sample_config_text())
    config["search"]["pmids_list"] = ["111", "222"]
    config["output"]["directory"] = str(tmp_path / "run")
    config["parsing"]["parse_coordinates"] = kind != "records"
    config["documents"] = {"enabled": True, "kind": kind, "root": str(root)}
    for section, values in overrides.items():
        config[section].update(values)
    pipeline = AutonimaPipeline(ConfigManager().load_from_dict(deepcopy(config)))
    pipeline.results.studies = [
        Study(
            pmid=pmid,
            title=f"Study {pmid}",
            abstract="An abstract.",
            authors=["A. Author"],
            journal="J",
            publication_date="2020",
            status=StudyStatus.INCLUDED_ABSTRACT,
        )
        for pmid in ("111", "222")
    ]
    return pipeline


def _csv_rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def test_records_source_replaces_article_retrieval(tmp_path):
    root = tmp_path / "records"
    _write_record(root)
    pipeline = _pipeline(tmp_path, "records", root)

    async def fail(*args, **kwargs):
        raise AssertionError("a records source must not retrieve articles")

    pipeline._retrieve_articles = fail
    asyncio.run(pipeline._execute_retrieval_phase())

    with_document, without = pipeline.results.studies
    assert with_document.fulltext_available is True
    assert with_document.full_text_source == "document:records"
    assert [a.name for a in with_document.analyses] == ["faces > shapes", "shapes > faces"]
    assert without.fulltext_available is False and without.analyses == []

    stats = pipeline.results.execution_stats["retrieval"]
    assert stats["documents_attached"] == 1 and stats["documents_unavailable"] == 1
    assert pipeline._stage_counters("retrieval")["documents_unavailable"] == 1
    rows = _csv_rows(tmp_path / "run" / "outputs" / "documents_unavailable.csv")
    assert [(r["pmid"], r["reason"]) for r in rows] == [("222", "no_document")]


def test_text_source_still_retrieves_articles(tmp_path):
    root = tmp_path / "summaries"
    root.mkdir()
    (root / "111.md").write_text("A summary.", encoding="utf-8")
    pipeline = _pipeline(tmp_path, "text", root)
    calls = []

    async def retrieve(studies, pmids, load_excluded):
        calls.append(sorted(pmids))
        return []

    pipeline._retrieve_articles = retrieve
    asyncio.run(pipeline._execute_retrieval_phase())

    assert calls == [[111, 222]]  # coordinates still come from the article
    assert pipeline.results.studies[0].full_text == "A summary."


def test_fulltext_screening_reads_the_document(tmp_path):
    root = tmp_path / "records"
    _write_record(root)
    pipeline = _pipeline(tmp_path, "records", root)
    asyncio.run(pipeline._execute_retrieval_phase())

    response = MagicMock(
        decision="INCLUDED",
        confidence=0.9,
        reason="Meets criteria",
        fulltext_incomplete=False,
        inclusion_criteria_applied=[],
        exclusion_criteria_applied=[],
    )
    with patch("autonima.screening.screener.GenericLLMClient") as client_class:
        client_class.return_value.screen_fulltext.return_value = response
        asyncio.run(pipeline._execute_fulltext_screening())
        prompts = [call.args[0] for call in client_class.return_value.screen_fulltext.call_args_list]

    assert len(prompts) == 1  # the study without a document is not screened
    assert "Study Document: Record.\n- 111_analysis_0: faces > shapes" in prompts[0]
    assert "Full Text Content" not in prompts[0]
    assert [s.status for s in pipeline.results.studies] == [
        StudyStatus.INCLUDED_FULLTEXT,
        StudyStatus.INCLUDED_ABSTRACT,
    ]


def test_transported_analyses_are_never_reparsed(tmp_path):
    root = tmp_path / "records"
    _write_record(root)
    pipeline = _pipeline(tmp_path, "records", root)
    asyncio.run(pipeline._execute_retrieval_phase())
    study = pipeline.results.studies[0]
    study.status = StudyStatus.INCLUDED_FULLTEXT
    before = [a.model_dump() for a in study.analyses]
    # Config validation refuses this combination; force it to prove the stage guards too.
    pipeline.config.parsing.parse_coordinates = True

    with patch("autonima.coordinates.CoordinateProcessor") as processor:
        processor.side_effect = AssertionError("transported analyses were sent for parsing")
        asyncio.run(pipeline._execute_coordinate_parsing())

    assert [a.model_dump() for a in study.analyses] == before


def test_outputs_account_for_documents(tmp_path):
    root = tmp_path / "records"
    _write_record(root)
    pipeline = _pipeline(tmp_path, "records", root, output={"nimads": True})
    asyncio.run(pipeline._execute_retrieval_phase())
    pipeline.results.studies[0].status = StudyStatus.INCLUDED_FULLTEXT

    asyncio.run(pipeline._generate_basic_outputs())
    asyncio.run(pipeline._generate_nimads_output())

    outputs = tmp_path / "run" / "outputs"
    missing = _csv_rows(outputs / "missing_fulltexts.csv")
    assert [(r["pmid"], r["type"], r["source"]) for r in missing] == [
        ("222", "unavailable", "document:records")
    ]
    final = json.loads((outputs / "final_results.json").read_text())
    document = final["studies"][0]["document"]
    assert document["analyses"][1] == {"key": "t1#2", "analysis_id": "111_analysis_1"}
    assert "Record." not in json.dumps(document)

    studyset = json.loads((outputs / "nimads_studyset.json").read_text())
    analyses = studyset["studies"][0]["analyses"]
    assert [a["id"] for a in analyses] == ["111_analysis_0", "111_analysis_1"]
    assert {p["space"] for a in analyses for p in a["points"]} == {"MNI"}


@pytest.mark.parametrize("kind", ["text", "records"])
def test_skipped_abstract_screening_still_counts_missing_documents(tmp_path, kind):
    root = tmp_path / "source"
    if kind == "records":
        _write_record(root)
    else:
        root.mkdir()
        (root / "111.md").write_text("A summary.", encoding="utf-8")
    pipeline = _pipeline(tmp_path, kind, root)
    pipeline.config.screening.abstract["skip_stage"] = True
    for study in pipeline.results.studies:
        study.status = StudyStatus.PENDING

    async def retrieve(*args):
        return []

    pipeline._retrieve_articles = retrieve

    asyncio.run(pipeline._execute_abstract_screening())
    asyncio.run(pipeline._execute_retrieval_phase())
    asyncio.run(pipeline._generate_basic_outputs())

    # With no abstract decision on file, the missing document is the only record of why 222
    # never reached screening -- so it must be counted, not dropped.
    outputs = tmp_path / "run" / "outputs"
    missing = _csv_rows(outputs / "missing_fulltexts.csv")
    assert [(r["pmid"], r["type"]) for r in missing] == [("222", "unavailable")]
    reasons = _csv_rows(outputs / "documents_unavailable.csv")
    assert [(r["pmid"], r["reason"]) for r in reasons] == [("222", "no_document")]
