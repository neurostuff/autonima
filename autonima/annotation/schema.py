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
    # Options: "single_analysis" (per-analysis) or
    # "multi_analysis" (whole study)
    prompt_type: str = "single_analysis"
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


class TableMetadata(BaseModel):
    """Metadata for a table containing analyses."""
    table_id: str
    caption: Optional[str] = None
    footer: Optional[str] = None
    # Add any other table-level fields as needed

    @field_validator('caption', 'footer', mode='before')
    @classmethod
    def validate_string_fields(cls, v):
        """Handle nan values in string fields."""
        if v is None:
            return None
        if isinstance(v, float):
            import math
            if math.isnan(v):
                return None
        if isinstance(v, str) and v.lower() == 'nan':
            return None
        return str(v) if v is not None else None


class AnalysisMetadata(BaseModel):
    """Metadata for an analysis used in annotation decisions."""
    analysis_id: str
    study_id: str
    table_id: str  # Reference to TableMetadata
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
    custom_fields: Dict[str, Any] = {}


class StudyAnalysisGroup(BaseModel):
    """Container for all analyses and tables in a study"""
    study_id: str
    study_title: Optional[str] = None
    study_abstract: Optional[str] = None
    study_authors: Optional[List[str]] = None
    study_journal: Optional[str] = None
    study_publication_date: Optional[str] = None
    study_fulltext: Optional[str] = None
    tables: List[TableMetadata] = []
    analyses: List[AnalysisMetadata] = []