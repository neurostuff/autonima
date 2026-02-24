"""Label-quality heuristics for coordinate analyses."""

from __future__ import annotations

import re
from typing import Optional


_GENERIC_NAME_PATTERNS = (
    "analysis",
    "contrast",
    "table",
    "result",
    "results",
    "standard deviation",
    "sd",
)

_GENERIC_NAME_REGEXES = (
    re.compile(r"^(analysis|contrast|result|results|table)[ _-]?\d+$"),
    re.compile(r"^(analysis|contrast|result|results|table)\s+[ivxlcdm]+$"),
)

def normalize_text(value: Optional[str]) -> str:
    """Normalize text for deterministic heuristic matching."""
    if not value:
        return ""
    text = value.strip().lower()
    text = (
        text.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
        .replace("\u2009", " ")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def compute_name_quality_score(name: Optional[str]) -> float:
    """Score how discriminative an analysis name is (0-1)."""
    text = normalize_text(name)
    if not text:
        return 0.0

    score = 0.0
    if any(token in text for token in (">", "<", " vs ", " versus ")):
        score += 0.5
    if any(token in text for token in ("group", "run", "interaction", "main effect")):
        score += 0.25
    if any(
        token in text
        for token in (
            "positive correlation",
            "negative correlation",
            "increase",
            "decrease",
        )
    ):
        score += 0.25
    return min(score, 1.0)


def is_generic_analysis_name(name: Optional[str]) -> bool:
    """Return True for placeholder labels like analysis_0/contrast 1."""
    text = normalize_text(name)
    if not text:
        return True

    if text in _GENERIC_NAME_PATTERNS:
        return True

    return any(rx.match(text) for rx in _GENERIC_NAME_REGEXES)
