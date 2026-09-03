"""Per-stage token and cost accounting for LLM calls.

Nothing in a run used to record what it cost. `execution_manifest.json` and
`execution_progress.json` carried stage timings and cached-versus-computed counts, but no token
counts, so the only way to answer "what does a project cost" was to multiply item counts by a
token estimate after the fact -- which is both laborious and wrong whenever prompt sizes differ
from the median.

This module is a process-global, thread-safe accumulator. Call sites report the `usage` block
that the API already returns; `execution.py` snapshots the totals onto each stage as it completes.

Two design notes:

- **Reported cost is the cost of *this execution*, not of producing the artifact from scratch.**
  Stages are incremental, so a re-run that computes 3 of 500 items records only those 3 calls.
  That is the honest number for "what did this run cost", and the `counters` already on each stage
  say how many items were reused, so the distinction stays legible. A from-scratch figure needs
  `--cache ignore`.
- **Unknown models are counted but not priced.** Prices change and vary by region and tier, so
  guessing is worse than abstaining: tokens are always recorded, `cost_usd` is null unless the
  model is in the price table.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

__all__ = [
    "record",
    "snapshot",
    "reset",
    "totals",
    "price_for",
    "MODEL_PRICES_ENV",
]

MODEL_PRICES_ENV = "AUTONIMA_MODEL_PRICES"

# USD per 1M tokens. Deliberately sparse: only models whose pricing has been confirmed, since a
# stale guess produces a confident wrong number. Extend via AUTONIMA_MODEL_PRICES, e.g.
#   AUTONIMA_MODEL_PRICES='{"gpt-5-mini": {"input": 0.25, "cached_input": 0.03, "output": 2.00}}'
# Keys are matched against the bare model name, longest match first, so a dated snapshot like
# "gpt-5-mini-2025-08-07" resolves via the "gpt-5-mini" entry.
DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.03, "output": 2.00},
}

_LOCK = threading.Lock()
_USAGE: Dict[str, Dict[str, Dict[str, int]]] = {}


def _load_prices() -> Dict[str, Dict[str, float]]:
    prices = dict(DEFAULT_PRICES)
    raw = os.getenv(MODEL_PRICES_ENV, "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if isinstance(value, dict):
                    prices[str(key)] = {k: float(v) for k, v in value.items()}
    return prices


def price_for(model: Optional[str]) -> Optional[Dict[str, float]]:
    """Price table entry for a model, matching the longest configured prefix."""
    if not model:
        return None
    bare = str(model).split("/")[-1]
    prices = _load_prices()
    matches = [key for key in prices if key and key in bare]
    if not matches:
        return None
    return prices[max(matches, key=len)]


def record(stage: str, model: Optional[str], usage: Any) -> None:
    """Accumulate one call's usage against a stage.

    `usage` is the object the API returns. Accepts the OpenAI chat-completions shape
    (`prompt_tokens` / `completion_tokens`, with `prompt_tokens_details.cached_tokens`) and the
    Responses shape (`input_tokens` / `output_tokens`). Never raises: accounting must not be able
    to fail a run.
    """
    try:
        if usage is None:
            return

        def _get(obj: Any, *names: str) -> int:
            for name in names:
                value = getattr(obj, name, None)
                if value is None and isinstance(obj, dict):
                    value = obj.get(name)
                if isinstance(value, (int, float)):
                    return int(value)
            return 0

        prompt = _get(usage, "prompt_tokens", "input_tokens")
        completion = _get(usage, "completion_tokens", "output_tokens")
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None and isinstance(usage, dict):
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
        cached = _get(details, "cached_tokens") if details is not None else 0

        key = str(model or "unknown")
        with _LOCK:
            by_model = _USAGE.setdefault(str(stage), {})
            bucket = by_model.setdefault(
                key, {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += prompt
            bucket["cached_input_tokens"] += min(cached, prompt)
            bucket["output_tokens"] += completion
    except Exception:  # noqa: BLE001 - accounting is never worth failing a run for
        return


def _summarize(by_model: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    calls = fresh_in = cached_in = out = 0
    cost: Optional[float] = 0.0
    for model, bucket in sorted(by_model.items()):
        # Cached input is billed at its own rate, so it must be subtracted from the full input
        # count rather than added alongside it.
        uncached = max(bucket["input_tokens"] - bucket["cached_input_tokens"], 0)
        price = price_for(model)
        model_cost: Optional[float] = None
        if price:
            # Kept unrounded while it is summed; rounding here and again at the total would
            # compound the error across models.
            model_cost = (
                uncached * price.get("input", 0.0) / 1e6
                + bucket["cached_input_tokens"] * price.get("cached_input", price.get("input", 0.0)) / 1e6
                + bucket["output_tokens"] * price.get("output", 0.0) / 1e6
            )
        models[model] = {
            **bucket,
            "cost_usd": round(model_cost, 6) if model_cost is not None else None,
        }

        calls += bucket["calls"]
        fresh_in += uncached
        cached_in += bucket["cached_input_tokens"]
        out += bucket["output_tokens"]
        if model_cost is None:
            cost = None
        elif cost is not None:
            cost += model_cost

    return {
        "calls": calls,
        "input_tokens": fresh_in + cached_in,
        "uncached_input_tokens": fresh_in,
        "cached_input_tokens": cached_in,
        "output_tokens": out,
        "cost_usd": round(cost, 6) if cost is not None else None,
        "by_model": models,
    }


def snapshot(stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Totals for one stage, or None if that stage made no calls."""
    # Delegated before taking the lock: _LOCK is not reentrant, and totals() acquires it too.
    if stage is None:
        return totals()
    with _LOCK:
        by_model = dict(_USAGE.get(str(stage), {}))
    if not by_model:
        return None
    return _summarize(by_model)


def totals() -> Dict[str, Any]:
    """Per-stage totals plus a run-wide roll-up."""
    with _LOCK:
        stages = {name: dict(models) for name, models in _USAGE.items()}
    combined: Dict[str, Dict[str, int]] = {}
    for by_model in stages.values():
        for model, bucket in by_model.items():
            agg = combined.setdefault(
                model, {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
            )
            for field, value in bucket.items():
                agg[field] += value
    return {
        "stages": {name: _summarize(models) for name, models in sorted(stages.items()) if models},
        "total": _summarize(combined) if combined else None,
    }


def reset(stage: Optional[str] = None) -> None:
    """Clear accumulated usage for one stage, or all of it."""
    with _LOCK:
        if stage is None:
            _USAGE.clear()
        else:
            _USAGE.pop(str(stage), None)
