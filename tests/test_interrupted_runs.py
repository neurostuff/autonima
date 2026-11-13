"""A run stopped partway keeps what it finished, and the next run redoes only the rest.

Annotation and coordinate parsing used to write their results once, after every study was
done, so a run killed partway lost every decision it had paid for.
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.schema import (
    AnnotationConfig,
    AnnotationCriteriaConfig,
    AnnotationDecision,
)
from autonima.config import ConfigManager
from autonima.coordinates.schema import Analysis
from autonima.models.types import ActivationTable, Study, StudyStatus
from autonima.pipeline import AutonimaPipeline


def _study(pmid, n_analyses=2, n_tables=0):
    return Study(
        pmid=pmid,
        title=f"Study {pmid}",
        abstract="Abstract",
        authors=["Doe"],
        journal="Journal",
        publication_date="2020",
        status=StudyStatus.INCLUDED_FULLTEXT,
        analyses=[
            Analysis(name=f"contrast {i}", description="desc", points=[])
            for i in range(n_analyses)
        ],
        activation_tables=[
            ActivationTable(table_id=f"{pmid}-t{i}", table_label=f"Table {i}", raw_table="<table/>")
            for i in range(n_tables)
        ],
    )


def _annotation_rows(output_dir):
    path = Path(output_dir) / "outputs" / "annotation_results.json"
    return json.loads(path.read_text()) if path.exists() else []


def _annotated_studies(output_dir):
    """Studies with custom decisions saved. The all_analyses rows need no call, so every
    study has those from the start."""
    return {
        row["study_id"] for row in _annotation_rows(output_dir)
        if row["annotation_name"] != "all_analyses"
    }


class InterruptingClient:
    """Stands in for AnnotationClient: answers ``answer`` studies, then the next call raises
    KeyboardInterrupt as Ctrl-C would. Calls already under way at that point still answer,
    after ``linger`` seconds, as a real in-flight request does.

    ``on_disk`` is what annotation_results.json held at the interrupt: all a hard kill, which
    runs no cleanup at all, would have left behind.
    """

    def __init__(self, answer=None, output_dir=None, linger=0.3):
        self.answer = answer
        self.output_dir = output_dir
        self.linger = linger
        self.calls = []
        self.answered = []
        self.on_disk = None
        self._lock = threading.Lock()

    def make_decision(self, group, criteria, metadata_fields, model=None,
                      model_params=None, prompt_type=None):
        with self._lock:
            self.calls.append(group.study_id)
            call = len(self.calls)
            if self.answer is not None and call == self.answer + 1:
                if self.output_dir is not None:
                    self.on_disk = _annotated_studies(self.output_dir)
                raise KeyboardInterrupt
        if self.answer is not None and call > self.answer + 1:
            time.sleep(self.linger)
        with self._lock:
            self.answered.append(group.study_id)
        return [
            AnnotationDecision(
                annotation_name=c.name,
                analysis_id=a.analysis_id,
                study_id=group.study_id,
                include=True,
                reasoning="meets criteria",
                model_used=model or "m",
            )
            for a in group.analyses
            for c in criteria
        ]


def _processor(client, num_workers=1, **kwargs):
    config = AnnotationConfig(
        metadata_fields=["analysis_name"],
        annotations=[
            AnnotationCriteriaConfig(name=name, inclusion_criteria=[f"{name} criterion"])
            for name in ("A", "B")
        ],
    )
    processor = AnnotationProcessor(config, num_workers=num_workers, **kwargs)
    processor.client = client
    return processor


def test_each_finished_study_is_on_disk_before_the_run_is_killed(tmp_path):
    studies = [_study(str(i)) for i in range(1, 7)]
    client = InterruptingClient(answer=3, output_dir=tmp_path)

    with pytest.raises(KeyboardInterrupt):
        _processor(client, save_interval=0).process_studies(studies, output_dir=str(tmp_path))

    # Written as each study finished, not by cleanup on the way out.
    assert client.on_disk == {"1", "2", "3"}
    rows = _annotation_rows(tmp_path)
    for study_id in ("1", "2", "3"):
        study_rows = [row for row in rows if row["study_id"] == study_id]
        # 2 analyses x (all_analyses + A + B), every one signed for reuse
        assert len(study_rows) == 6
        assert all(row["cache_signature"] for row in study_rows)
    # The unfinished studies have their system annotations only, so they are retried.
    assert {row["annotation_name"] for row in rows if row["study_id"] == "5"} == {"all_analyses"}


@pytest.mark.parametrize("num_workers", [1, 3])
def test_a_rerun_reuses_what_an_interrupted_run_finished(tmp_path, num_workers):
    studies = [_study(str(i)) for i in range(1, 21)]
    first = InterruptingClient(answer=4)

    with pytest.raises(KeyboardInterrupt):
        _processor(first, num_workers=num_workers).process_studies(
            studies, output_dir=str(tmp_path))

    finished = set(first.answered)
    assert len(finished) >= 4
    # Every study that got its decisions kept them, the calls still in flight included...
    assert _annotated_studies(tmp_path) == finished
    # ...and the studies still queued were never started.
    assert len(first.calls) < len(studies)

    second = InterruptingClient()
    processor = _processor(second, num_workers=num_workers)
    results = processor.process_studies(studies, output_dir=str(tmp_path))

    # Only the unfinished studies are billed again.
    assert sorted(second.calls, key=int) == sorted(
        {study.pmid for study in studies} - finished, key=int)
    assert len(results) == len(studies) * 6
    assert all(r.include is True for r in results)
    # 2 analyses each: all_analyses for every study, A and B for the finished ones
    assert processor.cache_stats["reused"] == len(studies) * 2 + len(finished) * 4
    assert processor.cache_stats["processed"] == (len(studies) - len(finished)) * 4

    # And a third run has nothing left to do.
    third = InterruptingClient()
    _processor(third, num_workers=num_workers).process_studies(studies, output_dir=str(tmp_path))
    assert third.calls == []


def _parser(answer=None, output_dir=None):
    """A CoordinateProcessor stand-in that parses ``answer`` tables, then raises
    KeyboardInterrupt. ``state["on_disk"]`` is what the parsing cache held at that point."""
    state = {"calls": [], "answered": [], "on_disk": None}
    lock = threading.Lock()
    cache = Path(output_dir or ".") / "outputs" / "coordinate_parsing_results.json"

    class Parser:
        def __init__(self, *args, **kwargs):
            pass

        def process_single_table(self, table):
            with lock:
                state["calls"].append(table.table_id)
                if answer is not None and len(state["calls"]) > answer:
                    if state["on_disk"] is None and cache.exists():
                        state["on_disk"] = [
                            s["pmid"] for s in json.loads(cache.read_text())["studies"]
                        ]
                    raise KeyboardInterrupt
                state["answered"].append(table.table_id)
            return [Analysis(name=f"from {table.table_id}", description="", points=[],
                             table_id=table.table_id, parsed=True)]

    return Parser, state


def _parsing_pipeline(tmp_path, num_workers):
    config = ConfigManager().create_sample_config()
    config.output.directory = str(tmp_path)
    config.parsing.parse_coordinates = True
    pipeline = AutonimaPipeline(config)
    pipeline.num_workers = num_workers
    pipeline.coordinate_save_interval = 0
    (tmp_path / "outputs").mkdir(exist_ok=True)
    # Two tables each, so a study can be caught half parsed.
    pipeline.results.studies = [_study(f"S{i}", n_analyses=0, n_tables=2) for i in range(1, 7)]
    return pipeline


def test_parsing_saves_each_finished_study_before_the_run_is_killed(tmp_path, monkeypatch):
    pipeline = _parsing_pipeline(tmp_path, num_workers=1)
    parser, state = _parser(answer=5, output_dir=tmp_path)
    monkeypatch.setattr("autonima.coordinates.CoordinateProcessor", parser)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(pipeline._execute_coordinate_parsing())

    # S3 had one of its two tables parsed: it is neither saved nor reused as complete.
    assert state["on_disk"] == ["S1", "S2"]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_a_parsing_rerun_reuses_what_an_interrupted_run_finished(tmp_path, monkeypatch, num_workers):
    pipeline = _parsing_pipeline(tmp_path, num_workers)
    parser, first = _parser(answer=5)
    monkeypatch.setattr("autonima.coordinates.CoordinateProcessor", parser)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(pipeline._execute_coordinate_parsing())

    answered = set(first["answered"])
    finished = {
        study.pmid for study in pipeline.results.studies
        if {table.table_id for table in study.activation_tables} <= answered
    }
    assert finished
    cached = json.loads((tmp_path / "outputs" / "coordinate_parsing_results.json").read_text())
    assert {s["pmid"] for s in cached["studies"]} == finished

    rerun = _parsing_pipeline(tmp_path, num_workers)
    parser, second = _parser()
    monkeypatch.setattr("autonima.coordinates.CoordinateProcessor", parser)
    asyncio.run(rerun._execute_coordinate_parsing())

    # Only the unfinished studies' tables are parsed again -- both tables of each.
    assert sorted(second["calls"]) == sorted(
        table.table_id
        for study in rerun.results.studies if study.pmid not in finished
        for table in study.activation_tables
    )
    assert rerun.results.execution_stats["coordinate_parsing"]["studies_reused"] == len(finished)
    for study in rerun.results.studies:
        assert [a.table_id for a in study.analyses] == [f"{study.pmid}-t0", f"{study.pmid}-t1"]
