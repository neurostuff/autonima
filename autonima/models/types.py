"""Type definitions and data models for Autonima."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from ..coordinates.schema import Analysis
from ..annotation.schema import AnnotationConfig


class StudyStatus(Enum):
    """Status of a study in the systematic review pipeline."""
    PENDING = "pending"
    INCLUDED_ABSTRACT = "included_abstract"
    EXCLUDED_ABSTRACT = "excluded_abstract"
    INCLUDED_FULLTEXT = "included_fulltext"
    EXCLUDED_FULLTEXT = "excluded_fulltext"
    RETRIEVAL_FAILED = "retrieval_failed"
    SCREENING_FAILED = "screening_failed"


@dataclass
class ActivationTable:
    """Represents a table containing activation coordinates from a study."""
    table_id: str  # New identifier for the table
    table_label: str  # Label or identifier for the table
    table_caption: Optional[str] = None  # Caption of the table
    table_foot: Optional[str] = None  # Footer of the table
    table_data_path: Optional[str] = None  # Path to processed table data file
    table_raw_path: Optional[str] = None  # Path to raw table data file
    raw_table: Optional[str] = None  # Raw table XML content
    
    def load_raw_table(self) -> None:
        """
        Load the raw table content from the preferred path.
        
        This method populates the raw_table attribute with the content
        of the raw table file. If the raw_table attribute is already
        populated, it doesn't need to do anything. It keeps the
        preference hierarchy for which path to use for populating it.
        """
        # If raw_table is already populated, no need to do anything
        if self.raw_table is not None:
            return
        
        # Define the path preference hierarchy
        path_preference = ['table_raw_path', 'table_data_path']
        
        # Try to load the raw table content from the preferred paths
        for path_attr in path_preference:
            table_path_value = getattr(self, path_attr, None)
            if table_path_value:
                try:
                    with open(table_path_value, 'r', encoding='utf-8') as f:
                        self.raw_table = f.read()
                    return
                except Exception as e:
                    logging.warning(
                        f"Failed to load raw table from "
                        f"{table_path_value}: {e}"
                    )
                    continue
        
        logging.warning(
            f"No valid table path found for table: {self.table_id}"
        )


@dataclass
class Study:
    """Represents a single study in the systematic review."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    status: StudyStatus = StudyStatus.PENDING
    abstract_screening_reason: Optional[str] = None
    fulltext_screening_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    abstract_screening_score: Optional[float] = None
    fulltext_screening_score: Optional[float] = None
    retrieved_at: Optional[datetime] = None
    screened_at: Optional[datetime] = None
    pmcid: Optional[str] = None
    full_text_path: Optional[str] = None
    fulltext_available: bool = False  # Whether full text is available
    coordinate_space: Optional[str] = None
    activation_tables: List[ActivationTable] = field(default_factory=list)
    analyses: List[Analysis] = field(default_factory=list)
    
    abstract_inclusion_criteria_applied: List[str] = field(
        default_factory=list
    )
    abstract_exclusion_criteria_applied: List[str] = field(
        default_factory=list
    )
    fulltext_inclusion_criteria_applied: List[str] = field(
        default_factory=list
    )
    fulltext_exclusion_criteria_applied: List[str] = field(
        default_factory=list
    )
    full_text_output_dir: Optional[str] = None  # Full text output dir
    _full_text: Optional[str] = None  # Cached full text content

    def to_dict(self) -> Dict[str, Any]:
        """Convert study to dictionary representation."""
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "doi": self.doi,
            "keywords": self.keywords,
            "status": self.status.value,
            "abstract_screening_reason": self.abstract_screening_reason,
            "fulltext_screening_reason": self.fulltext_screening_reason,
            "metadata": self.metadata,
            "abstract_screening_score": self.abstract_screening_score,
            "fulltext_screening_score": self.fulltext_screening_score,
            "retrieved_at": (
                self.retrieved_at.isoformat() if self.retrieved_at else None
            ),
            "screened_at": (
                self.screened_at.isoformat() if self.screened_at else None
            ),
            "full_text_path": self.full_text_path,
            "fulltext_available": self.fulltext_available,
            "coordinate_space": self.coordinate_space,
            "activation_tables": [
                {
                    "table_id": table.table_id,
                    "table_label": table.table_label,
                    "table_caption": table.table_caption,
                    "table_foot": table.table_foot,
                    "table_data_path": table.table_data_path,
                    "table_raw_path": table.table_raw_path
                } for table in self.activation_tables
            ],
            "analyses": [
                {
                    "name": analysis.name,
                    "description": analysis.description,
                    "points": [
                        {
                            "coordinates": point.coordinates,
                            "space": point.space,
                            "values": [
                                {
                                    "value": value.value,
                                    "kind": value.kind
                                }
                                for value in point.values or []
                            ]
                        }
                        for point in analysis.points
                    ]
                }
                for analysis in self.analyses
            ],
            # Criteria tracking fields
            "abstract_inclusion_criteria_applied":
                self.abstract_inclusion_criteria_applied,
            "abstract_exclusion_criteria_applied":
                self.abstract_exclusion_criteria_applied,
            "fulltext_inclusion_criteria_applied":
                self.fulltext_inclusion_criteria_applied,
            "fulltext_exclusion_criteria_applied":
                self.fulltext_exclusion_criteria_applied
        }
    
    @property
    def full_text(self) -> str:
        """Lazy-load and cache full text content.
        
        Returns:
            The full text content as a string, or None if not found
            
        Raises:
            ValueError: If full_text_output_dir is not set
            FileNotFoundError: If the text file doesn't exist at the
            expected location
        """
        if self._full_text is not None:
            return self._full_text
            
        if not self.full_text_output_dir:
            raise ValueError(
                "full_text_output_dir must be set before accessing full_text"
            )
        
        # Import here to avoid circular imports
        from ..retrieval.utils import _load_full_text
        
        # Load and cache the full text
        self._full_text = _load_full_text(
            self, output_dir=self.full_text_output_dir
        )
        return self._full_text


@dataclass
class SearchConfig:
    """Configuration for the search phase."""
    database: str = "pubmed"
    query: str = ""
    max_results: int = 1000
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    email: Optional[str] = None  # Required for NCBI API
    pmids_file: Optional[str] = None  # Path to file with PMIDs (one per line)
    pmids_list: Optional[List[str]] = None  # Direct list of PMIDs


@dataclass
class ScreeningConfig:
    """Configuration for the screening phase."""
    abstract: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4",
        "threshold": None,
        "confidence_reporting": False,
        "objective": None,
        "inclusion_criteria": None,
        "exclusion_criteria": None,
        "additional_instructions": None,
    })
    fulltext: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4",
        "threshold": None,
        "confidence_reporting": False,
        "objective": None,
        "inclusion_criteria": None,
        "exclusion_criteria": None,
        "additional_instructions": None,
    })


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval phase."""
    sources: List[str] = field(default_factory=lambda: ["pubget", "ace"])
    timeout: int = 30
    max_retries: int = 3
    download_directory: str = "downloads"
    n_jobs: int = 1
    load_excluded: bool = False  # Load full texts for excluded studies?
    # Optional full text source configurations
    full_text_sources: List[Dict[str, Any]] = field(default_factory=list)
    # Coordinate parsing configuration
    parse_coordinates: bool = False
    coordinate_model: str = "gpt-4o-mini"


@dataclass
class ParsingConfig:
    """Configuration for the parsing phase."""
    parse_coordinates: bool = False
    coordinate_model: str = "gpt-4o-mini"


@dataclass
class OutputConfig:
    """Configuration for the output phase."""
    directory: str = "results"
    prisma_diagram: bool = True
    formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    nimads: bool = False
    export_excluded_studies: bool = False


@dataclass
class PipelineConfig:
    """Main configuration for the Autonima pipeline."""
    search: SearchConfig
    screening: ScreeningConfig
    retrieval: RetrievalConfig
    output: OutputConfig
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        from ..utils.criteria import CriteriaMapping
        
        # Helper function to serialize screening config dicts
        def serialize_screening_dict(
            screening_dict: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Serialize a screening dict, handling CriteriaMapping objects."""
            result = {}
            for key, value in screening_dict.items():
                if isinstance(value, CriteriaMapping):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
            return result
        
        return {
            "search": {
                "database": self.search.database,
                "query": self.search.query,
                "max_results": self.search.max_results,
                "date_from": self.search.date_from,
                "date_to": self.search.date_to,
                "email": self.search.email,
                "pmids_file": self.search.pmids_file,
                "pmids_list": self.search.pmids_list,
            },
            "screening": {
                "abstract": serialize_screening_dict(self.screening.abstract),
                "fulltext": serialize_screening_dict(self.screening.fulltext),
            },
            "retrieval": {
                "sources": self.retrieval.sources,
                "timeout": self.retrieval.timeout,
                "max_retries": self.retrieval.max_retries,
                "download_directory": self.retrieval.download_directory,
                "n_jobs": self.retrieval.n_jobs,
                "load_excluded": self.retrieval.load_excluded,
                "full_text_sources": self.retrieval.full_text_sources,
                "parse_coordinates": self.retrieval.parse_coordinates,
                "coordinate_model": self.retrieval.coordinate_model,
            },
            "parsing": {
                "parse_coordinates": self.parsing.parse_coordinates,
                "coordinate_model": self.parsing.coordinate_model,
            },
            "annotation": {
                "model": self.annotation.model,
                "create_all_included_annotation": (
                    self.annotation.create_all_included_annotation
                ),
                "create_all_from_search_annotation": (
                    self.annotation.create_all_from_search_annotation
                ),
                "annotations": [
                    {
                        "name": criteria.name,
                        "description": criteria.description,
                        "inclusion_criteria": criteria.inclusion_criteria,
                        "exclusion_criteria": criteria.exclusion_criteria,
                        "metadata_fields": criteria.metadata_fields,
                    }
                    for criteria in self.annotation.annotations
                ],
                "enabled": self.annotation.enabled,
            },
            "output": {
                "directory": self.output.directory,
                "prisma_diagram": self.output.prisma_diagram,
                "formats": self.output.formats,
                "nimads": self.output.nimads,
                "export_excluded_studies": self.output.export_excluded_studies,
            },
        }


@dataclass
class ScreeningResult:
    """Result of screening a single study."""
    study_id: str
    decision: StudyStatus
    reason: str
    confidence: float
    model_used: str
    screening_type: str  # "abstract" or "fulltext"
    timestamp: datetime = field(default_factory=datetime.now)
    inclusion_criteria_applied: List[str] = field(default_factory=list)
    exclusion_criteria_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert screening result to dictionary."""
        return {
            "study_id": self.study_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "screening_type": self.screening_type,
            "timestamp": self.timestamp.isoformat(),
            "inclusion_criteria_applied": self.inclusion_criteria_applied,
            "exclusion_criteria_applied": self.exclusion_criteria_applied,
        }


@dataclass
class PipelineResult:
    """Results from running the complete pipeline."""
    config: PipelineConfig
    studies: List[Study] = field(default_factory=list)
    abstract_screening_results: List[ScreeningResult] = field(
        default_factory=list)
    fulltext_screening_results: List[ScreeningResult] = field(
        default_factory=list)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self, final_studies_only: bool = False) -> Dict[str, Any]:
        """Convert pipeline result to dictionary.
        
        Args:
            final_studies_only: If True, only include studies with status
            INCLUDED_FULLTEXT
        """
        # Filter studies if requested
        studies_to_include = self.studies
        if final_studies_only:
            studies_to_include = [
                study for study in self.studies
                if study.status == StudyStatus.INCLUDED_FULLTEXT
            ]
        
        return {
            "config": self.config.to_dict(),
            "studies": [study.to_dict() for study in studies_to_include],
            "abstract_screening_results": [
                result.to_dict() for result in self.abstract_screening_results
            ],
            "fulltext_screening_results": [
                result.to_dict() for result in self.fulltext_screening_results
            ],
            "execution_stats": self.execution_stats,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }