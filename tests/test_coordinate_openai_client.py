"""Tests for coordinate parsing client kind normalization."""

from autonima.coordinates.openai_client import CoordinateParsingClient


def test_normalize_kind_handles_schema_mismatches():
    normalize = CoordinateParsingClient._normalize_kind

    assert normalize("t-statistic") == "t-statistic"
    assert normalize("stat1") == "t-statistic"
    assert normalize("stat2") == "other"
    assert normalize("cluster_size") is None
    assert normalize("BA") is None
    assert normalize("z") == "z-statistic"
    assert normalize("pvalue") == "p-value"

