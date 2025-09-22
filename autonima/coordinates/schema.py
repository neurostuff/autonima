"""Pydantic models for coordinate parsing output."""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Union


class CoordinatePoint(BaseModel):
    """Represents a single coordinate point with its space information."""
    coordinates: List[float]  # [x, y, z] values
    space: Optional[str] = None  # Template space (e.g., MNI or TAL)
    
    @field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v):
        """Validate that coordinates is a list of exactly 3 floats."""
        if not isinstance(v, list):
            raise ValueError('Coordinates must be a list')
        if len(v) != 3:
            raise ValueError('Coordinates must contain exactly 3 values [x, y, z]')
        for i, coord in enumerate(v):
            if not isinstance(coord, (int, float)):
                raise ValueError(f'Coordinate at index {i} must be a number')
        return [float(coord) for coord in v]


class Analysis(BaseModel):
    """Represents a single analysis with its metadata and points."""
    name: Optional[str] = None  # Name of the contrast being performed
    description: Optional[str] = None  # Long form description of how the contrast was performed
    points: List[CoordinatePoint]  # Coordinates of significance associated with the contrast


class ParseAnalysesOutput(BaseModel):
    """Output schema for the parse_analyses function."""
    analyses: List[Analysis]  # List of distinct analyses parsed from the table