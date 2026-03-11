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
    )


def test_generate_basic_outputs_tracks_fulltext_incomplete(tmp_path):
    """Test PRISMA stats and files include fulltext_incomplete."""
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    pipeline = AutonimaPipeline(config)

    pipeline.results.studies = [
        _make_study("PMID_INCLUDED", StudyStatus.INCLUDED_FULLTEXT),
        _make_study("PMID_EXCLUDED", StudyStatus.EXCLUDED_FULLTEXT),
        _make_study(
            "PMID_INCOMPLETE",
            StudyStatus.FULLTEXT_INCOMPLETE,
            full_text_path="/tmp/incomplete_fulltext.txt",
        ),
    ]

    asyncio.run(pipeline._generate_basic_outputs())

    prisma_stats = pipeline.results.execution_stats["prisma_stats"]
    assert prisma_stats["final_included"] == 1
    assert prisma_stats["fulltext_excluded"] == 1
    assert prisma_stats["fulltext_incomplete"] == 1
    assert prisma_stats["fulltext_assessed"] == 3

    incomplete_file = tmp_path / "outputs" / "incomplete_fulltext.txt"
    assert incomplete_file.exists()
    assert incomplete_file.read_text().strip() == "PMID_INCOMPLETE"

    incomplete_csv_file = tmp_path / "outputs" / "incomplete_fulltext.csv"
    assert incomplete_csv_file.exists()
    with open(incomplete_csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {
            "pmid": "PMID_INCOMPLETE",
            "full_text_path": "/tmp/incomplete_fulltext.txt",
        }
    ]

    final_results = json.loads(
        (tmp_path / "outputs" / "final_results.json").read_text()
    )
    assert len(final_results["studies"]) == 1
    assert final_results["studies"][0]["pmid"] == "PMID_INCLUDED"
