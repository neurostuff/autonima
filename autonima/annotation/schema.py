"""Pydantic models for annotation configuration and results."""
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict, create_model, field_validator, model_validator
from typing import Any, Dict, List, Optional, Tuple, Type, Literal
from datetime import datetime
import re



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


# ---------- helpers ----------

_CRIT_ID_RE = re.compile(r"^(?:[A-Z0-9_]+_)?[IE]\d+$")


def _build_allowed_criteria_ids(criteria_list: List["AnnotationCriteriaConfig"]) -> Dict[str, Dict[str, set[str]]]:
    """
    Returns:
      {
        "<annotation_name>": {
          "inclusion": {"I1","I2",...},
          "exclusion": {"E1","E2",...}
        },
        ...
      }
    """
    allowed: Dict[str, Dict[str, set[str]]] = {}
    for c in criteria_list:
        inc_ids: set[str] = set()
        exc_ids: set[str] = set()

        if c.criteria_mapping:
            inc_ids = set((c.criteria_mapping.get("inclusion") or {}).keys())
            exc_ids = set((c.criteria_mapping.get("exclusion") or {}).keys())

        allowed[c.name] = {"inclusion": inc_ids, "exclusion": exc_ids}
    return allowed


def build_dynamic_multi_annotation_models(
    criteria_list: List["AnnotationCriteriaConfig"],
    *,
    enforce_complete_coverage: bool = True,
    forbid_extra_fields: bool = True,
) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """
    Dynamically generates two Pydantic models:
      - DecisionModel: one decision (annotation_name constrained to criteria_list names)
      - OutputListModel: { decisions: List[DecisionModel] } with optional coverage checks

    Returns: (DecisionModel, OutputListModel)
    """
    if not criteria_list:
        raise ValueError("criteria_list is empty; cannot build dynamic models.")

    allowed_annotation_names = [c.name for c in criteria_list]
    allowed_by_annotation = _build_allowed_criteria_ids(criteria_list)

    # Build Literal[...] dynamically (works in runtime)
    AnnotationNameLiteral = Literal[tuple(allowed_annotation_names)]  # type: ignore

    # Create the config dict based on parameters
    config_dict = ConfigDict(extra="forbid") if forbid_extra_fields else ConfigDict()

    # ---- Decision model ----
    class _DecisionBase(BaseModel):
        # Set model_config at class definition using the ConfigDict
        model_config = config_dict

        annotation_name: AnnotationNameLiteral = Field(
            ...,
            description="Name of the annotation (must match one of the provided criteria names exactly).",
        )
        include: bool = Field(..., description="Boolean decision: true to include, false to exclude")
        reasoning: str = Field(..., description="Brief explanation for the decision")
        inclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of inclusion criterion IDs that apply")
        exclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of exclusion criterion IDs that apply")

        @field_validator("include", mode="before")
        @classmethod
        def validate_include(cls, v: Any) -> bool:
            if v is None:
                raise ValueError("'include' field cannot be None - must be true or false")
            if isinstance(v, str):
                v_lower = v.lower()
                if v_lower in ("true", "yes", "1"):
                    return True
                if v_lower in ("false", "no", "0"):
                    return False
                raise ValueError(f"Invalid string value for 'include': {v}")
            return bool(v)

        @model_validator(mode="after")
        def validate_criteria_ids(self):
            """
            Dynamic constraint for criteria IDs:
            - If criteria_mapping exists for this annotation, enforce membership in the mapping keys.
            - Else enforce regex pattern ^(?:[A-Z0-9_]+_)?[IE]\\d+$.
            """
            ann = self.annotation_name  # constrained already
            allowed_inc = allowed_by_annotation[ann]["inclusion"]
            allowed_exc = allowed_by_annotation[ann]["exclusion"]

            # If mappings exist (non-empty sets), enforce strict membership
            if allowed_inc:
                bad = [x for x in self.inclusion_criteria_applied if x not in allowed_inc]
                if bad:
                    raise ValueError(
                        f'Invalid inclusion_criteria_applied for annotation "{ann}": {bad}. '
                        f"Allowed: {sorted(allowed_inc)}"
                    )
            else:
                bad = [x for x in self.inclusion_criteria_applied if not _CRIT_ID_RE.match(x)]
                if bad:
                    raise ValueError(
                        "Invalid inclusion_criteria_applied IDs "
                        f'(must match ^(?:[A-Z0-9_]+_)?[IE]\\d+$) for annotation "{ann}": {bad}'
                    )

            if allowed_exc:
                bad = [x for x in self.exclusion_criteria_applied if x not in allowed_exc]
                if bad:
                    raise ValueError(
                        f'Invalid exclusion_criteria_applied for annotation "{ann}": {bad}. '
                        f"Allowed: {sorted(allowed_exc)}"
                    )
            else:
                bad = [x for x in self.exclusion_criteria_applied if not _CRIT_ID_RE.match(x)]
                if bad:
                    raise ValueError(
                        "Invalid exclusion_criteria_applied IDs "
                        f'(must match ^(?:[A-Z0-9_]+_)?[IE]\\d+$) for annotation "{ann}": {bad}'
                    )

            if self.include and allowed_inc and not self.inclusion_criteria_applied:
                raise ValueError(
                    f'annotation "{ann}" has include=true but inclusion_criteria_applied is empty'
                )

            if (
                not self.include
                and allowed_inc
                and not self.exclusion_criteria_applied
            ):
                reasoning_upper = (self.reasoning or "").upper()
                mentioned_missing_ids = [
                    crit_id
                    for crit_id in allowed_inc
                    if crit_id.upper() in reasoning_upper
                ]
                if not mentioned_missing_ids:
                    raise ValueError(
                        f'annotation "{ann}" has include=false and no '
                        "exclusion_criteria_applied; reasoning must cite "
                        "missing/not-met inclusion criteria IDs"
                    )

            return self

    DecisionModel = _DecisionBase

    # ---- Output list model ----
    class _DecisionListBase(BaseModel):
        model_config = config_dict

        decisions: List[DecisionModel] = Field(..., description="List of annotation decisions")

        @model_validator(mode="after")
        def validate_coverage_and_uniqueness(self):
            names = [d.annotation_name for d in self.decisions]

            # Uniqueness always helps consistency
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate annotation_name values found: {names}")

            if enforce_complete_coverage:
                missing = [n for n in allowed_annotation_names if n not in set(names)]
                extra = [n for n in set(names) if n not in set(allowed_annotation_names)]
                if missing:
                    raise ValueError(f"Missing decisions for annotation_name(s): {missing}")
                if extra:
                    raise ValueError(f"Unexpected annotation_name(s): {extra}")

            return self

    OutputListModel = _DecisionListBase

    return DecisionModel, OutputListModel
