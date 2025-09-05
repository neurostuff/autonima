"""Base interface for search engines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models.types import Study, SearchConfig


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    def __init__(self, config: SearchConfig):
        """Initialize the search engine with configuration."""
        self.config = config

    @abstractmethod
    async def search(self, query: str) -> List[Study]:
        """
        Execute search and return list of studies.

        Args:
            query: Search query string

        Returns:
            List of Study objects found by the search
        """
        pass

    @abstractmethod
    def get_search_info(self) -> Dict[str, Any]:
        """
        Get information about the search engine and current configuration.

        Returns:
            Dictionary with search engine metadata
        """
        pass

    def build_query(
        self,
        base_query: str,
    ) -> str:
        """
        Build a complete query string from base query.
        
        Args:
            base_query: Base search query
            
        Returns:
            Complete query string
        """
        query_parts = [base_query]
        
        # Add date filters if specified
        if self.config.date_from:
            date_from = self.config.date_from
            date_filter = (
                f'("{date_from}"[Date - Publication] : '
                f'"3000"[Date - Publication])'
            )
            query_parts.append(date_filter)
            
        if self.config.date_to:
            date_filter = (
                f'("1900"[Date - Publication] : '
                f'"{self.config.date_to}"[Date - Publication])'
            )
            query_parts.append(date_filter)
            
        return " AND ".join(query_parts)