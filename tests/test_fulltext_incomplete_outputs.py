"""Tests for fulltext_incomplete output propagation."""

import asyncio
import csv
import json

from autonima.config import ConfigManager
from autonima.models.types import Study, StudyStatus
from autonima.pipeline import AutonimaPipeline


def _make_study(
    pmid: str,
    status: StudyStatus,
    full_text_path: str | None = None,
    full_text_source: str | None = None,
    retrieval_in_scope: bool = False,
    in_included_set: bool | None = None,
    fulltext_available: bool = True,
) -> Study:
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        abstract="Abstract text",
        authors=["Author A"],
        journal="Journal",
        publication_date="2023",
        status=status,
        full_text_path=full_text_path,
        full_text_source=full_text_source,
        retrieval_in_scope=retrieval_in_scope,
        in_included_set=in_included_set,
        fulltext_available=fulltext_available,
    )


def test_generate_basic_outputs_tracks_fulltext_incomplete(tmp_path):
    """Test PRISMA stats and consolidated missing fulltext report."""
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    pipeline = AutonimaPipeline(config)

    pipeline.results.studies = [
        _make_study("PMID_INCLUDED", StudyStatus.INCLUDED_FULLTEXT),
        _make_study(
            "PMID_UNAVAILABLE_INCLUDED",
            StudyStatus.INCLUDED_ABSTRACT,
            full_text_source="pubget",
            retrieval_in_scope=True,
            in_included_set=True,
            fulltext_available=False,
        ),
        _make_study(
            "PMID_UNAVAILABLE_EXCLUDED",
            StudyStatus.EXCLUDED_ABSTRACT,
            full_text_source="pubget",
            retrieval_in_scope=True,
            in_included_set=False,
            fulltext_available=False,
        ),
        _make_study(
            "PMID_INCOMPLETE",
            StudyStatus.FULLTEXT_INCOMPLETE,
            full_text_path="/tmp/incomplete_fulltext.txt",
            full_text_source="/tmp/local_corpus",
            retrieval_in_scope=True,
            in_included_set=True,
        ),
    ]

    asyncio.run(pipeline._generate_basic_outputs())

    prisma_stats = pipeline.results.execution_stats["prisma_stats"]
    assert prisma_stats["final_included"] == 1
    assert prisma_stats["fulltext_excluded"] == 0
    assert prisma_stats["fulltext_incomplete"] == 1
    assert prisma_stats["fulltext_assessed"] == 2

    missing_fulltexts_file = tmp_path / "outputs" / "missing_fulltexts.csv"
    assert missing_fulltexts_file.exists()
    with open(missing_fulltexts_file, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "pmid": "PMID_INCOMPLETE",
            "type": "incomplete",
            "source": "/tmp/local_corpus",
            "full_text_path": "/tmp/incomplete_fulltext.txt",
            "in_included_set": "True",
        },
        {
            "pmid": "PMID_UNAVAILABLE_EXCLUDED",
            "type": "unavailable",
            "source": "pubget",
            "full_text_path": "",
            "in_included_set": "False",
        },
        {
            "pmid": "PMID_UNAVAILABLE_INCLUDED",
            "type": "unavailable",
            "source": "pubget",
            "full_text_path": "",
            "in_included_set": "True",
        },
    ]

    missing_fulltexts_txt_file = tmp_path / "outputs" / "missing_fulltexts.txt"
    assert missing_fulltexts_txt_file.exists()
    assert missing_fulltexts_txt_file.read_text() == (
        "PMID_INCOMPLETE\n"
        "PMID_UNAVAILABLE_EXCLUDED\n"
        "PMID_UNAVAILABLE_INCLUDED\n"
    )

    assert not (tmp_path / "outputs" / "unavailable_fulltexts.txt").exists()
    assert not (tmp_path / "outputs" / "incomplete_fulltext.txt").exists()
    assert not (tmp_path / "outputs" / "incomplete_fulltext.csv").exists()

    final_results = json.loads(
        (tmp_path / "outputs" / "final_results.json").read_text()
    )
    assert len(final_results["studies"]) == 1
    assert final_results["studies"][0]["pmid"] == "PMID_INCLUDED"


def test_generate_basic_outputs_incomplete_pmcid_path_fallback(tmp_path):
    """Test incomplete rows include pubget text.csv reference for PMCID."""
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    pipeline = AutonimaPipeline(config)

    pipeline.results.studies = [
        _make_study(
            "PMID_INCOMPLETE_PMCID",
            StudyStatus.FULLTEXT_INCOMPLETE,
            full_text_path=None,
            full_text_source="pubget",
            retrieval_in_scope=True,
            in_included_set=True,
            fulltext_available=True,
        )
    ]
    pipeline.results.studies[0].pmcid = "12345"

    asyncio.run(pipeline._generate_basic_outputs())

    with open(tmp_path / "outputs" / "missing_fulltexts.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "pmid": "PMID_INCOMPLETE_PMCID",
            "type": "incomplete",
            "source": "pubget",
            "full_text_path": (
                f"{tmp_path / 'retrieval' / 'pubget_data' / 'text.csv'} "
                "[pmcid=12345]"
            ),
            "in_included_set": "True",
        }
    ]

    assert (
        (tmp_path / "outputs" / "missing_fulltexts.txt").read_text()
        == "PMID_INCOMPLETE_PMCID\n"
    )
