"""A failed LLM call must not be cached as if it were a result.

Both stages below used to turn an exception into something the cache then reused: a
screening failure with a valid signature, and a table parse that returned ``[]``.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autonima.config import ConfigManager
from autonima.coordinates.processor import CoordinateProcessor
from autonima.coordinates.schema import Analysis
from autonima.models.types import ActivationTable, ScreeningConfig, Study, StudyStatus
from autonima.pipeline import AutonimaPipeline
from autonima.screening import LLMScreener


def _study(pmid, status=StudyStatus.PENDING, tables=None):
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        abstract=f"Abstract for {pmid}.",
        authors=["Author"],
        journal="Journal",
        publication_date="2024",
        status=status,
        activation_tables=tables or [],
    )


def _screening_config():
    config = ScreeningConfig()
    config.abstract.update({
        "objective": "Test objective",
        "inclusion_criteria": ["Included criterion"],
        "exclusion_criteria": ["Excluded criterion"],
    })
    return config


def _included_response():
    return SimpleNamespace(
        decision="INCLUDED",
        confidence=0.9,
        reason="Meets criteria",
        inclusion_criteria_applied=[],
        exclusion_criteria_applied=[],
    )


def test_cached_screening_failure_is_rescreened(tmp_path):
    studies = [_study("FAILS_FIRST"), _study("SUCCEEDS")]

    def first_run(prompt, model):
        # Routed on the abstract text, which the prompt is sure to contain.
        if "Abstract for FAILS_FIRST" in prompt:
            raise RuntimeError("429 rate limited")
        return _included_response()

    with patch("autonima.screening.screener.GenericLLMClient") as client_class:
        client = MagicMock()
        client.screen_abstract.side_effect = first_run
        client_class.return_value = client
        screener = LLMScreener(_screening_config(), output_dir=str(tmp_path))
        first = asyncio.run(screener.screen_abstracts(studies))

    by_id = {result.study_id: result for result in first}
    assert by_id["FAILS_FIRST"].decision == StudyStatus.SCREENING_FAILED
    assert by_id["SUCCEEDS"].decision == StudyStatus.INCLUDED_ABSTRACT

    # The failure lands beside the successes, where the next run looks, not in the run root.
    saved = json.loads((tmp_path / "outputs" / "abstract_screening_results.json").read_text())
    rows = saved["screening_results"] if isinstance(saved, dict) else saved
    assert {row["study_id"] for row in rows} == {"FAILS_FIRST", "SUCCEEDS"}
    assert not (tmp_path / "abstract_screening_results.json").exists()

    with patch("autonima.screening.screener.GenericLLMClient") as client_class:
        client = MagicMock()
        client.screen_abstract.return_value = _included_response()
        client_class.return_value = client
        screener = LLMScreener(_screening_config(), output_dir=str(tmp_path))
        second = asyncio.run(screener.screen_abstracts(studies))

    # Only the failure is sent again; the real decision is reused.
    assert client.screen_abstract.call_count == 1
    assert screener.cache_stats["abstract"] == {"eligible": 2, "reused": 1, "processed": 1}
    assert {r.study_id: r.decision for r in second} == {
        "FAILS_FIRST": StudyStatus.INCLUDED_ABSTRACT,
        "SUCCEEDS": StudyStatus.INCLUDED_ABSTRACT,
    }


@patch("autonima.coordinates.processor.CoordinateParsingClient")
def test_coordinate_processor_raises_instead_of_returning_empty(_client_class):
    processor = CoordinateProcessor()
    processor.client.parse_analyses.side_effect = RuntimeError("429 rate limited")
    table = ActivationTable(table_id="t1", table_label="Table 1", raw_table="<table/>")

    with pytest.raises(RuntimeError, match="429"):
        processor.process_single_table(table)


@patch("autonima.coordinates.processor.CoordinateParsingClient")
def test_coordinate_processor_still_returns_empty_for_unreadable_table(_client_class):
    processor = CoordinateProcessor()
    table = ActivationTable(table_id="t1", table_label="Table 1")

    assert processor.process_single_table(table) == []
    processor.client.parse_analyses.assert_not_called()


class _FlakyProcessor:
    """Parses every table except ``bad``, which raises as a rate-limited call would."""

    def __init__(self, *args, **kwargs):
        pass

    def process_single_table(self, table):
        if table.table_id == "bad":
            raise RuntimeError("429 rate limited")
        return [Analysis(name=f"from {table.table_id}", description="", points=[],
                         table_id=table.table_id)]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_study_with_a_failed_table_is_not_cached(tmp_path, monkeypatch, num_workers):
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    config.parsing.parse_coordinates = True
    pipeline = AutonimaPipeline(config)
    pipeline.num_workers = num_workers
    (tmp_path / "outputs").mkdir(exist_ok=True)

    partial = _study(
        "PARTIAL",
        status=StudyStatus.INCLUDED_FULLTEXT,
        tables=[
            ActivationTable(table_id="good", table_label="Table 1", raw_table="<table/>"),
            ActivationTable(table_id="bad", table_label="Table 2", raw_table="<table/>"),
        ],
    )
    complete = _study(
        "COMPLETE",
        status=StudyStatus.INCLUDED_FULLTEXT,
        tables=[ActivationTable(table_id="ok", table_label="Table 1", raw_table="<table/>")],
    )
    pipeline.results.studies = [partial, complete]
    monkeypatch.setattr("autonima.coordinates.CoordinateProcessor", _FlakyProcessor)

    asyncio.run(pipeline._execute_coordinate_parsing())

    # This run still uses what did parse...
    assert [a.table_id for a in partial.analyses] == ["good"]
    # ...but only the complete study is cached, so the partial one is retried next run.
    cached = json.loads((tmp_path / "outputs" / "coordinate_parsing_results.json").read_text())
    assert [s["pmid"] for s in cached["studies"]] == ["COMPLETE"]

    stats = pipeline.results.execution_stats["coordinate_parsing"]
    assert stats["tables_failed"] == 1
    assert stats["failed_tables"] == {"PARTIAL": ["bad"]}
    assert stats["tables_processed"] == 2
