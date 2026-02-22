"""Tests for deterministic boundary and naming heuristics."""

from autonima.coordinates.boundary_rules import (
    build_boundary_markers,
    compute_name_quality_score,
    is_generic_analysis_name,
)


def test_name_quality_and_generic_detection():
    assert is_generic_analysis_name("analysis")
    assert is_generic_analysis_name("standard deviation")
    assert compute_name_quality_score("analysis") == 0.0

    assert not is_generic_analysis_name("emotional > neutral")
    assert compute_name_quality_score("emotional > neutral") >= 0.5


def test_boundary_markers_split_directional_blocks():
    rows = [
        {
            "row_index": 0,
            "has_coordinates": False,
            "section_label": "emotional > neutral",
            "primary_label": "emotional > neutral",
        },
        {
            "row_index": 1,
            "has_coordinates": True,
            "section_label": "emotional > neutral",
            "primary_label": "Occipital Pole",
        },
        {
            "row_index": 2,
            "has_coordinates": False,
            "section_label": "neutral > emotional",
            "primary_label": "neutral > emotional",
        },
        {
            "row_index": 3,
            "has_coordinates": True,
            "section_label": "neutral > emotional",
            "primary_label": "Precuneus",
        },
    ]

    markers = build_boundary_markers(rows)
    assert markers["segment_count"] == 2
    assert markers["segments"][0]["boundary_key"] != markers["segments"][1]["boundary_key"]

