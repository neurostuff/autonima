"""Search module for literature database integration."""

from .base import SearchEngine
from .pubmed import PubMedSearch

__all__ = ["SearchEngine", "PubMedSearch"]