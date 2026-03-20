"""Tests for pipeline stage cutoff behavior."""

import asyncio

import pytest

from autonima.config import ConfigManager
from autonima.pipeline import AutonimaPipeline


def _async_recorder(phase_name: str, calls: list[str]):
    async def _record() -> None:
        calls.append(phase_name)

    return _record


def _async_fail(phase_name: str):
    async def _fail() -> None:
        raise AssertionError(f"{phase_name} should not run for this cutoff")

    return _fail


def _build_pipeline(tmp_path) -> AutonimaPipeline:
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    return AutonimaPipeline(config)


def test_pipeline_stop_after_search_runs_only_search(tmp_path, monkeypatch):
    pipeline = _build_pipeline(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline, "_execute_search_phase", _async_recorder("search", calls)
    )
    monkeypatch.setattr(
        pipeline, "_execute_abstract_screening", _async_fail("abstract")
    )
    monkeypatch.setattr(
        pipeline, "_execute_retrieval_phase", _async_fail("retrieval")
    )
    monkeypatch.setattr(
        pipeline, "_execute_fulltext_screening", _async_fail("fulltext")
    )
    monkeypatch.setattr(
        pipeline, "_execute_coordinate_parsing", _async_fail("parsing")
    )
    monkeypatch.setattr(
        pipeline, "_execute_annotation_phase", _async_fail("annotation")
    )
    monkeypatch.setattr(
        pipeline, "_execute_output_phase", _async_fail("output")
    )

    results = asyncio.run(pipeline.run(stop_after_stage="search"))

    assert calls == ["search"]
    assert results.execution_stats["stop_after_stage"] == "search"
    assert results.execution_stats["completed_stage"] == "search"
    assert results.completed_at is not None


def test_pipeline_stop_after_abstract_runs_search_and_abstract(
    tmp_path, monkeypatch
):
    pipeline = _build_pipeline(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline, "_execute_search_phase", _async_recorder("search", calls)
    )
    monkeypatch.setattr(
        pipeline,
        "_execute_abstract_screening",
        _async_recorder("abstract", calls),
    )
    monkeypatch.setattr(
        pipeline, "_execute_retrieval_phase", _async_fail("retrieval")
    )
    monkeypatch.setattr(
        pipeline, "_execute_fulltext_screening", _async_fail("fulltext")
    )
    monkeypatch.setattr(
        pipeline, "_execute_coordinate_parsing", _async_fail("parsing")
    )
    monkeypatch.setattr(
        pipeline, "_execute_annotation_phase", _async_fail("annotation")
    )
    monkeypatch.setattr(
        pipeline, "_execute_output_phase", _async_fail("output")
    )

    results = asyncio.run(pipeline.run(stop_after_stage="abstract"))

    assert calls == ["search", "abstract"]
    assert results.execution_stats["stop_after_stage"] == "abstract"
    assert results.execution_stats["completed_stage"] == "abstract"
    assert results.completed_at is not None


def test_pipeline_full_run_still_executes_all_phases(tmp_path, monkeypatch):
    pipeline = _build_pipeline(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline, "_execute_search_phase", _async_recorder("search", calls)
    )
    monkeypatch.setattr(
        pipeline,
        "_execute_abstract_screening",
        _async_recorder("abstract", calls),
    )
    monkeypatch.setattr(
        pipeline, "_execute_retrieval_phase", _async_recorder("retrieval", calls)
    )
    monkeypatch.setattr(
        pipeline,
        "_execute_fulltext_screening",
        _async_recorder("fulltext", calls),
    )
    monkeypatch.setattr(
        pipeline,
        "_execute_coordinate_parsing",
        _async_recorder("parsing", calls),
    )
    monkeypatch.setattr(
        pipeline,
        "_execute_annotation_phase",
        _async_recorder("annotation", calls),
    )
    monkeypatch.setattr(
        pipeline, "_execute_output_phase", _async_recorder("output", calls)
    )

    results = asyncio.run(pipeline.run())

    assert calls == [
        "search",
        "abstract",
        "retrieval",
        "fulltext",
        "parsing",
        "annotation",
        "output",
    ]
    assert results.execution_stats["stop_after_stage"] == "full"
    assert results.execution_stats["completed_stage"] == "full"
    assert results.completed_at is not None


def test_pipeline_rejects_invalid_stop_after_stage(tmp_path):
    pipeline = _build_pipeline(tmp_path)

    with pytest.raises(ValueError, match="Invalid stop_after_stage"):
        asyncio.run(pipeline.run(stop_after_stage="retrieval"))  # type: ignore[arg-type]
