"""Pydantic models for coordinate parsing output."""

from pydantic import BaseModel
from typing import List, Optional, Union


class CoordinatePoint(BaseModel):
    """Represents a single coordinate point with its space information."""
    coordinates: List[float]  # [x, y, z] values
    space: Optional[str] = None  # Template space (e.g., MNI or TAL)


class Analysis(BaseModel):
    """Represents a single analysis with its metadata and points."""
    name: Optional[str] = None  # Name of the contrast being performed
    description: Optional[str] = None  # Long form description of how the contrast was performed
    points: List[CoordinatePoint]  # Coordinates of significance associated with the contrast


class ParseAnalysesOutput(BaseModel):
    """Output schema for the parse_analyses function."""
    analyses: List[Analysis]  # List of distinct analyses parsed from the table