"""Tests for fulltext_incomplete output propagation."""

import asyncio
import json

from autonima.config import ConfigManager
from autonima.models.types import Study, StudyStatus
from autonima.pipeline import AutonimaPipeline


def _make_study(pmid: str, status: StudyStatus) -> Study:
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        abstract="Abstract text",
        authors=["Author A"],
        journal="Journal",
        publication_date="2023",
        status=status,
    )


def test_generate_basic_outputs_tracks_fulltext_incomplete(tmp_path):
    """Test PRISMA stats and files include fulltext_incomplete."""
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    pipeline = AutonimaPipeline(config)

    pipeline.results.studies = [
        _make_study("PMID_INCLUDED", StudyStatus.INCLUDED_FULLTEXT),
        _make_study("PMID_EXCLUDED", StudyStatus.EXCLUDED_FULLTEXT),
        _make_study("PMID_INCOMPLETE", StudyStatus.FULLTEXT_INCOMPLETE),
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

    final_results = json.loads(
        (tmp_path / "outputs" / "final_results.json").read_text()
    )
    assert len(final_results["studies"]) == 1
    assert final_results["studies"][0]["pmid"] == "PMID_INCLUDED"
