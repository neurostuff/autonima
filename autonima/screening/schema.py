"""Pydantic models for screening output schema."""

from pydantic import BaseModel
from typing import Dict, Literal, List


class AbstractScreeningOutput(BaseModel):
    """Schema for abstract screening output."""
    decision: Literal["INCLUDED", "EXCLUDED"]
    confidence: float
    reason: str
    inclusion_criteria_applied: List[str] = []
    exclusion_criteria_applied: List[str] = []
    # Criterion ID -> probability of truth. Only calibrated backends fill this; chat models
    # leave it empty. See ScreeningResult.criterion_probabilities.
    criterion_probabilities: Dict[str, float] = {}


class FullTextScreeningOutput(BaseModel):
    """Schema for full-text screening output."""
    decision: Literal["INCLUDED", "EXCLUDED"]
    confidence: float
    reason: str
    fulltext_incomplete: bool = False
    inclusion_criteria_applied: List[str] = []
    exclusion_criteria_applied: List[str] = []
    criterion_probabilities: Dict[str, float] = {}
