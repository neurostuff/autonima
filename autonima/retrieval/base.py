"""Abstract base class for retrieval modules."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from ..models.types import Study


class BaseRetriever(ABC):
    """Abstract base class for full-text article retrieval."""

    @abstractmethod
    def retrieve(
        self,
        studies: List[Study],
        output_dir: Path,
        **kwargs
    ) -> List[Study]:
        """
        Retrieve full-text articles for a list of studies.
        
        Args:
            studies: List of studies that passed screening
            output_dir: Directory to store retrieved articles
            **kwargs: Additional parameters for retrieval
            
        Returns:
            List of studies with updated full_text_path attributes
        """
        pass

    @abstractmethod
    def validate_retrieval(
        self,
        studies: List[Study],
        output_dir: Path
    ) -> List[Study]:
        """
        Validate which studies have been successfully retrieved.
        
        Args:
            studies: List of studies to check
            output_dir: Directory where articles were retrieved
            
        Returns:
            List of studies with updated status
        """
        pass