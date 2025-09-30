"""Coordinate parsing submodule for Autonima."""

from .schema import CoordinatePoint, Analysis, ParseAnalysesOutput
from .parser import parse_tables
from .openai_client import CoordinateParsingClient
from .processor import CoordinateProcessor

__all__ = [
    "CoordinatePoint",
    "Analysis",
    "ParseAnalysesOutput",
    "parse_tables",
    "CoordinateParsingClient",
    "CoordinateProcessor"
]