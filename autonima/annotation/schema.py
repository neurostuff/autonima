"""Pydantic models for annotation configuration and results."""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class AnnotationCriteriaConfig(BaseModel):
    """Configuration for a single annotation criteria."""
    name: str
    description: Optional[str] = None
    inclusion_criteria: List[str] = []
    exclusion_criteria: List[str] = []
    metadata_fields: List[str] = []
    
    # NEW: Store criteria mappings
    criteria_mapping: Optional[Dict[str, Dict[str, str]]] = None


class AnnotationConfig(BaseModel):
    """Configuration for the annotation phase."""
    model: str = "gpt-4o-mini"
    # Create "all_analyses" annotation with all analyses from INCLUDED studies
    create_all_included_annotation: bool = True
    # Create "all_studies" annotation from INCLUDED and EXCLUDED studies
    create_all_from_search_annotation: bool = False
    annotations: List[AnnotationCriteriaConfig] = []
    enabled: bool = True
    metadata_fields: List[str] = [
        "analysis_name",
        "analysis_description",
        "table_caption",
        "study_title",
        "study_fulltext"
    ]
    inclusion_criteria: List[str] = []
    exclusion_criteria: List[str] = []


class AnnotationDecision(BaseModel):
    """Result of an annotation decision for a single analysis."""
    annotation_name: str
    analysis_id: str
    study_id: str
    include: bool
    reasoning: str
    confidence: Optional[float] = None
    model_used: str
    timestamp: datetime = datetime.now()
    
    # NEW: Track which criteria were applied
    inclusion_criteria_applied: List[str] = []
    exclusion_criteria_applied: List[str] = []


class AnalysisMetadata(BaseModel):
    """Metadata for an analysis used in annotation decisions."""
    analysis_id: str
    study_id: str
    analysis_name: Optional[str] = None
    analysis_description: Optional[str] = None
    table_caption: Optional[str] = None
    table_footer: Optional[str] = None
    study_title: Optional[str] = None
    study_abstract: Optional[str] = None
    study_authors: Optional[List[str]] = None
    study_journal: Optional[str] = None
    study_publication_date: Optional[str] = None
    study_fulltext: Optional[str] = None
    # Add any other fields as needed
    custom_fields: Dict[str, Any] = {}
    
    @field_validator('analysis_name', 'analysis_description', 'table_caption', 
                     'table_footer', 'study_title', 'study_abstract', 
                     'study_journal', 'study_publication_date', mode='before')
    @classmethod
    def validate_string_fields(cls, v):
        """
        Validate string fields and handle nan values.
        """
        if v is None:
            return None
        # Handle nan values (both float nan and string 'nan')
        if isinstance(v, float):
            import math
            if math.isnan(v):
                return None
        if isinstance(v, str) and v.lower() == 'nan':
            return None
        # Convert to string if needed
        return str(v) if v is not None else None