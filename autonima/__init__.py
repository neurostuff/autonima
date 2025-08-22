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

__all__ = ["AutonimaPipeline", "PipelineConfig"]