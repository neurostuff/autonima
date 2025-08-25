"""Pydantic models for screening output schema."""

from pydantic import BaseModel
from typing import Literal


class AbstractScreeningOutput(BaseModel):
    """Schema for abstract screening output."""
    decision: Literal["INCLUDED", "EXCLUDED"]
    confidence: float
    reason: str


class FullTextScreeningOutput(BaseModel):
    """Schema for full-text screening output."""
    decision: Literal["INCLUDED", "EXCLUDED"]
    confidence: float
    reason: str