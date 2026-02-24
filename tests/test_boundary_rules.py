"""Tests for naming heuristics."""

from autonima.coordinates.boundary_rules import (
    compute_name_quality_score,
    is_generic_analysis_name,
)


def test_name_quality_and_generic_detection():
    assert is_generic_analysis_name("analysis")
    assert is_generic_analysis_name("analysis_0")
    assert is_generic_analysis_name("contrast 2")
    assert is_generic_analysis_name("standard deviation")
    assert compute_name_quality_score("analysis") == 0.0

    assert not is_generic_analysis_name("emotional > neutral")
    assert not is_generic_analysis_name("Self-serving bias")
    assert not is_generic_analysis_name("Partnering.")
    assert compute_name_quality_score("emotional > neutral") >= 0.5
