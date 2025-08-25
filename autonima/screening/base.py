"""Base interface for screening engines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models.types import Study, ScreeningConfig, ScreeningResult


class ScreeningEngine(ABC):
    """Abstract base class for screening engines."""

    def __init__(self, config: ScreeningConfig):
        """Initialize the screening engine with configuration."""
        self.config = config

    @abstractmethod
    async def screen_abstracts(
        self, 
        studies: List[Study]
    ) -> List[ScreeningResult]:
        """
        Screen study abstracts for inclusion/exclusion.

        Args:
            studies: List of studies to screen

        Returns:
            List of screening results
        """
        pass

    @abstractmethod
    async def screen_fulltexts(
        self, 
        studies: List[Study]
    ) -> List[ScreeningResult]:
        """
        Screen full-text articles for inclusion/exclusion.

        Args:
            studies: List of studies with full text to screen

        Returns:
            List of screening results
        """
        pass

    @abstractmethod
    def get_screening_info(self) -> Dict[str, Any]:
        """
        Get information about the screening engine and current configuration.

        Returns:
            Dictionary with screening engine metadata
        """
        pass