"""Type definitions and data models for Autonima."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class StudyStatus(Enum):
    """Status of a study in the systematic review pipeline."""
    PENDING = "pending"
    INCLUDED = "included"
    EXCLUDED = "excluded"
    RETRIEVAL_FAILED = "retrieval_failed"
    SCREENING_FAILED = "screening_failed"
    FULLTEXT_RETRIEVED = "fulltext_retrieved"
    FULLTEXT_UNAVAILABLE = "fulltext_unavailable"
    FULLTEXT_CACHED = "fulltext_cached"


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
        }
    
    def load_full_text(self, output_dir: str) -> str:
        """Load the full text content for this study.
        
        Args:
            output_dir: Output directory where pubget data is stored
            
        Returns:
            The full text content as a string, or None if not found
            
        Raises:
            ValueError: If output_dir is not provided
            FileNotFoundError: If the text file doesn't exist at the expected location
        """
        # Import here to avoid circular imports
        from ..retrieval.utils import _load_full_text
        
        # Use the _load_full_text function with the output directory
        return _load_full_text(self, output_dir=output_dir)


@dataclass
class SearchConfig:
    """Configuration for the search phase."""
    database: str = "pubmed"
    query: str = ""
    max_results: int = 1000
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    email: Optional[str] = None  # Required for NCBI API


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
    # Optional full text source configurations
    full_text_sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Configuration for the output phase."""
    directory: str = "results"
    prisma_diagram: bool = True
    formats: List[str] = field(default_factory=lambda: ["csv", "json"])
    include_metadata: bool = True
    compress_outputs: bool = False


@dataclass
class PipelineConfig:
    """Main configuration for the Autonima pipeline."""
    search: SearchConfig
    screening: ScreeningConfig
    retrieval: RetrievalConfig
    output: OutputConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return {
            "search": {
                "database": self.search.database,
                "query": self.search.query,
                "max_results": self.search.max_results,
                "date_from": self.search.date_from,
                "date_to": self.search.date_to,
                "email": self.search.email,
            },
            "screening": {
                "abstract": self.screening.abstract,
                "fulltext": self.screening.fulltext,
            },
            "retrieval": {
                "sources": self.retrieval.sources,
                "timeout": self.retrieval.timeout,
                "max_retries": self.retrieval.max_retries,
                "download_directory": self.retrieval.download_directory,
                "n_jobs": self.retrieval.n_jobs,
                "full_text_sources": self.retrieval.full_text_sources,
            },
            "output": {
                "directory": self.output.directory,
                "prisma_diagram": self.output.prisma_diagram,
                "formats": self.output.formats,
                "include_metadata": self.output.include_metadata,
                "compress_outputs": self.output.compress_outputs,
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
            final_studies_only: If True, only include studies with status INCLUDED
        """
        # Filter studies if requested
        studies_to_include = self.studies
        if final_studies_only:
            studies_to_include = [
                study for study in self.studies
                if study.status == StudyStatus.INCLUDED
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