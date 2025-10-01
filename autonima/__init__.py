"""
Autonima: Automated Neuroimaging Meta-Analysis

An LLM-powered framework for automating systematic literature reviews
and meta-analyses in neuroimaging, following the PRISMA framework.

This package provides automated tools for:
- Literature search via PubMed API
- Abstract and full-text screening using LLMs
- Full-text retrieval via PubGet and ACE
- PRISMA-compliant workflow management
- Output generation for downstream meta-analysis

Example:
    >>> from autonima import AutonimaPipeline
    >>> pipeline = AutonimaPipeline("config.yaml")
    >>> pipeline.run()
"""

__version__ = "0.1.0"
__author__ = "Autonima Development Team"
__description__ = "LLM-powered automated systematic review and meta-analysis"

from .pipeline import AutonimaPipeline
from .config import PipelineConfig

# Import retrieval module components
from .retrieval import BaseRetriever, PubGetRetriever, ACERetriever

# Import coordinate parsing module components
from .coordinates import CoordinatePoint, Analysis, ParseAnalysesOutput, parse_tables, CoordinateParsingClient

# Import LLM module components
from .llm.client import GenericLLMClient

# Import meta-analysis module components
from .meta import run_meta_analyses

__all__ = [
    "AutonimaPipeline",
    "PipelineConfig",
    "BaseRetriever",
    "PubGetRetriever",
    "ACERetriever",
    "CoordinatePoint",
    "Analysis",
    "ParseAnalysesOutput",
    "parse_tables",
    "CoordinateParsingClient",
    "GenericLLMClient",
    "run_meta_analyses"
]