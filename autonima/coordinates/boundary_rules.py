"""Deterministic boundary and label-quality heuristics for coordinate tables."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_DIRECTION_KEYWORDS = (
    " vs ",
    " versus ",
    " > ",
    " < ",
    "positive correlation",
    "negative correlation",
    "main effect",
    "interaction",
)

_GENERIC_NAME_PATTERNS = (
    "analysis",
    "contrast",
    "table",
    "result",
    "results",
    "standard deviation",
    "sd",
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


def _extract_direction_signature(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    # Preserve explicit directional comparisons where possible.
    gt_match = re.search(r"([a-z0-9()/_\-\s]{2,80})\s>\s([a-z0-9()/_\-\s]{2,80})", text)
    if gt_match:
        left = re.sub(r"\s+", " ", gt_match.group(1)).strip()
        right = re.sub(r"\s+", " ", gt_match.group(2)).strip()
        return f"{left}>{right}"

    lt_match = re.search(r"([a-z0-9()/_\-\s]{2,80})\s<\s([a-z0-9()/_\-\s]{2,80})", text)
    if lt_match:
        left = re.sub(r"\s+", " ", lt_match.group(1)).strip()
        right = re.sub(r"\s+", " ", lt_match.group(2)).strip()
        return f"{left}<{right}"

    if "positive correlation" in text:
        return "positive_correlation"
    if "negative correlation" in text:
        return "negative_correlation"
    if "interaction" in text:
        return "interaction"
    if " versus " in text or " vs " in text:
        return "vs"
    if "main effect" in text:
        return "main_effect"

    return ""


def _has_directionality(value: str) -> bool:
    text = normalize_text(value)
    return any(token in text for token in _DIRECTION_KEYWORDS)


def compute_boundary_key(row: Dict[str, Any]) -> str:
    """Compute a deterministic boundary key for a row."""
    section_label = normalize_text(row.get("section_label"))
    primary_label = normalize_text(row.get("primary_label"))

    candidates = []
    if section_label:
        candidates.append(section_label)
    if primary_label and _has_directionality(primary_label):
        candidates.append(primary_label)

    if not candidates:
        return "default"

    joined = " | ".join(candidates)
    direction_sig = _extract_direction_signature(joined)
    if direction_sig:
        return direction_sig

    # Fall back to short normalized section text.
    compact = re.sub(r"[^a-z0-9]+", "_", candidates[0]).strip("_")
    return compact[:120] or "default"


def build_boundary_markers(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assign boundary keys and segment indices to coordinate-bearing rows."""
    segments: List[Dict[str, Any]] = []
    current_segment: Optional[Dict[str, Any]] = None
    current_key: Optional[str] = None

    for row in rows:
        has_coordinates = bool(row.get("has_coordinates"))
        if not has_coordinates:
            row["boundary_key"] = current_key
            row["segment_index"] = (
                current_segment["segment_index"] if current_segment else None
            )
            continue

        boundary_key = compute_boundary_key(row)
        row["boundary_key"] = boundary_key

        if current_key != boundary_key:
            current_key = boundary_key
            current_segment = {
                "segment_index": len(segments),
                "boundary_key": boundary_key,
                "row_indices": [],
                "row_start": row["row_index"],
                "row_end": row["row_index"],
            }
            segments.append(current_segment)

        row["segment_index"] = current_segment["segment_index"]
        current_segment["row_indices"].append(row["row_index"])
        current_segment["row_end"] = row["row_index"]

    return {
        "segment_count": len(segments),
        "segments": segments,
    }


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
    """Return True when the name lacks discriminative contrast semantics."""
    text = normalize_text(name)
    if not text:
        return True

    if text in _GENERIC_NAME_PATTERNS:
        return True

    if any(pattern == text for pattern in _GENERIC_NAME_PATTERNS):
        return True

    if len(text.split()) <= 2 and not _has_directionality(text):
        return True

    for pattern in _GENERIC_NAME_PATTERNS:
        if text == pattern or text.startswith(f"{pattern} "):
            return True

    # Require at least one discriminative cue.
    return compute_name_quality_score(text) < 0.5

