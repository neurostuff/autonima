"""Tests for per-stage token and cost accounting."""

import json

import pytest

from autonima.llm import usage as llm_usage


class _Details:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _Usage:
    """Shape returned by the OpenAI chat-completions API."""

    def __init__(self, prompt_tokens, completion_tokens, cached_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _Details(cached_tokens) if cached_tokens is not None else None


@pytest.fixture(autouse=True)
def _clean_accumulator():
    llm_usage.reset()
    yield
    llm_usage.reset()


def test_records_tokens_per_stage_and_model():
    llm_usage.record("annotation", "gpt-5-mini-2025-08-07", _Usage(1000, 200))
    llm_usage.record("annotation", "gpt-5-mini-2025-08-07", _Usage(500, 100))
    llm_usage.record("parsing", "gpt-5-mini-2025-08-07", _Usage(600, 900))

    ann = llm_usage.snapshot("annotation")
    assert ann["calls"] == 2
    assert ann["input_tokens"] == 1500
    assert ann["output_tokens"] == 300

    parsing = llm_usage.snapshot("parsing")
    assert parsing["calls"] == 1
    assert parsing["output_tokens"] == 900

    assert llm_usage.snapshot("fulltext") is None


def test_cached_input_is_billed_at_its_own_rate_not_double_counted():
    llm_usage.record("annotation", "gpt-5-mini-2025-08-07", _Usage(2000, 300, cached_tokens=1096))
    snap = llm_usage.snapshot("annotation")

    # input_tokens stays the full prompt size; the cached portion is broken out, not added on top.
    assert snap["input_tokens"] == 2000
    assert snap["cached_input_tokens"] == 1096
    assert snap["uncached_input_tokens"] == 904

    expected = 904 * 0.25 / 1e6 + 1096 * 0.03 / 1e6 + 300 * 2.00 / 1e6
    # Reported to microdollar precision; that is the resolution of the artifact.
    assert snap["cost_usd"] == pytest.approx(expected, abs=1e-6)


def test_dated_model_snapshot_resolves_to_its_price_prefix():
    llm_usage.record("parsing", "gpt-5-mini-2025-08-07", _Usage(1_000_000, 0))
    assert llm_usage.snapshot("parsing")["cost_usd"] == pytest.approx(0.25)


def test_unpriced_model_records_tokens_but_reports_no_cost():
    llm_usage.record("parsing", "some-unreleased-model", _Usage(1000, 500))
    snap = llm_usage.snapshot("parsing")
    assert snap["input_tokens"] == 1000
    assert snap["cost_usd"] is None


def test_one_unpriced_model_makes_the_aggregate_cost_unknown_rather_than_partial():
    llm_usage.record("annotation", "gpt-5-mini-2025-08-07", _Usage(1000, 100))
    llm_usage.record("annotation", "some-unreleased-model", _Usage(1000, 100))
    # A partial sum would read as a full one, so it is withheld entirely.
    assert llm_usage.snapshot("annotation")["cost_usd"] is None


def test_accepts_dict_usage_and_responses_api_field_names():
    llm_usage.record("abstract", "gpt-5-mini", {"prompt_tokens": 10, "completion_tokens": 2})
    llm_usage.record("abstract", "gpt-5-mini", {"input_tokens": 20, "output_tokens": 4})
    snap = llm_usage.snapshot("abstract")
    assert snap["calls"] == 2
    assert snap["input_tokens"] == 30
    assert snap["output_tokens"] == 6


@pytest.mark.parametrize("bad", [None, object(), {}, "nonsense"])
def test_never_raises_on_malformed_usage(bad):
    llm_usage.record("abstract", "gpt-5-mini", bad)  # must not raise


def test_prices_can_be_extended_by_environment(monkeypatch):
    monkeypatch.setenv(
        llm_usage.MODEL_PRICES_ENV,
        json.dumps({"house-model": {"input": 1.0, "cached_input": 0.1, "output": 4.0}}),
    )
    llm_usage.record("annotation", "house-model", _Usage(1_000_000, 1_000_000))
    assert llm_usage.snapshot("annotation")["cost_usd"] == pytest.approx(5.0)


def test_reset_clears_only_the_named_stage():
    llm_usage.record("annotation", "gpt-5-mini", _Usage(10, 1))
    llm_usage.record("parsing", "gpt-5-mini", _Usage(10, 1))
    llm_usage.reset("annotation")
    assert llm_usage.snapshot("annotation") is None
    assert llm_usage.snapshot("parsing") is not None


def test_totals_roll_up_across_stages():
    llm_usage.record("annotation", "gpt-5-mini", _Usage(100, 10))
    llm_usage.record("parsing", "gpt-5-mini", _Usage(200, 20))
    total = llm_usage.totals()
    assert set(total["stages"]) == {"annotation", "parsing"}
    assert total["total"]["calls"] == 2
    assert total["total"]["input_tokens"] == 300
    assert total["total"]["output_tokens"] == 30


def test_concurrent_records_are_not_lost():
    from concurrent.futures import ThreadPoolExecutor

    def one(_):
        llm_usage.record("annotation", "gpt-5-mini", _Usage(10, 1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(400)))

    snap = llm_usage.snapshot("annotation")
    assert snap["calls"] == 400
    assert snap["input_tokens"] == 4000


def test_snapshot_without_a_stage_does_not_deadlock():
    # snapshot() delegates to totals(); both take the same non-reentrant lock, so doing that
    # while already holding it would hang the run rather than fail it.
    llm_usage.record("annotation", "gpt-5-mini", _Usage(10, 1))
    result = llm_usage.snapshot()
    assert result["total"]["calls"] == 1


def test_usage_is_persisted_onto_the_stage_and_rolled_up(tmp_path):
    """The point of the accumulator: it has to reach execution_progress.json."""
    from autonima.execution import (
        complete_execution_progress,
        initialize_execution_progress,
        load_execution_progress,
        update_execution_progress_stage,
    )

    (tmp_path / "outputs").mkdir()
    initialize_execution_progress(tmp_path, {})

    llm_usage.record("parsing", "gpt-5-mini-2025-08-07", _Usage(586, 911))
    update_execution_progress_stage(tmp_path, "parsing", status="completed")
    llm_usage.record("annotation", "gpt-5-mini-2025-08-07", _Usage(2008, 296, cached_tokens=1096))
    update_execution_progress_stage(tmp_path, "annotation", status="completed")
    complete_execution_progress(tmp_path, status="completed")

    progress = load_execution_progress(tmp_path)
    by_stage = {s["stage"]: s for s in progress["stages"] if isinstance(s, dict)}

    assert by_stage["parsing"]["usage"]["calls"] == 1
    assert by_stage["annotation"]["usage"]["cached_input_tokens"] == 1096
    # A stage that made no calls carries no usage key at all, rather than a misleading zero.
    assert "usage" not in by_stage["abstract"]

    total = progress["usage_total"]
    assert total["calls"] == 2
    assert total["input_tokens"] == 586 + 2008
    assert total["cost_usd"] == pytest.approx(
        by_stage["parsing"]["usage"]["cost_usd"] + by_stage["annotation"]["usage"]["cost_usd"],
        abs=1e-6,
    )
