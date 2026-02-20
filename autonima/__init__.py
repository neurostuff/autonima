"""
Autonima: Automated Neuroimaging Meta-Analysis

An LLM-powered framework for automating systematic literature reviews
and meta-analyses in neuroimaging, following the PRISMA framework.

This package provides automated tools for:
- Literature search via PubMed API
- Abstract and full-text screening using LLMs
- Full-text retrieval via PubGet
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

__all__ = []

# Core pipeline/config imports may rely on optional dependencies in submodules.
try:
    from .pipeline import AutonimaPipeline
    from .config import PipelineConfig

    __all__.extend(["AutonimaPipeline", "PipelineConfig"])
except ImportError:
    pass

# Retrieval module components (optional dependency: pubget and related stack)
try:
    from .retrieval import BaseRetriever, PubGetRetriever

    __all__.extend(["BaseRetriever", "PubGetRetriever"])
except ImportError:
    pass

# Coordinate parsing module components
try:
    from .coordinates import (
        CoordinatePoint,
        Analysis,
        ParseAnalysesOutput,
        parse_tables,
        CoordinateParsingClient,
    )

    __all__.extend(
        [
            "CoordinatePoint",
            "Analysis",
            "ParseAnalysesOutput",
            "parse_tables",
            "CoordinateParsingClient",
        ]
    )
except ImportError:
    pass

# LLM client (optional dependency: openai)
try:
    from .llm.client import GenericLLMClient

    __all__.append("GenericLLMClient")
except ImportError:
    pass

# Meta-analysis module components (optional dependency: nimare)
try:
    from .meta import run_meta_analyses

    HAS_META = True
    __all__.append("run_meta_analyses")
except ImportError:
    HAS_META = False
