"""ACE (Academic Citation Engine) integration stub.

This module is a placeholder for future ACE integration.
Currently, only PubGet is implemented for full-text retrieval.
"""

import logging
from pathlib import Path
from typing import List

from ..models.types import Study
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class ACERetriever(BaseRetriever):
    """Stub for ACE (Academic Citation Engine) integration.
    
    This is a placeholder implementation. Currently, only PubGet is supported
    for full-text retrieval from the PubMed Central Open Access subset.
    """

    def __init__(self):
        """Initialize the ACE retriever stub."""
        logger.warning(
            "ACE retriever is not implemented yet. "
            "Using PubGet for full-text retrieval instead."
        )

    def retrieve(
        self,
        studies: List[Study],
        output_dir: Path,
        **kwargs
    ) -> List[Study]:
        """
        Stub for ACE retrieval functionality.
        
        Args:
            studies: List of studies that passed screening
            output_dir: Directory to store retrieved articles
            **kwargs: Additional parameters
            
        Returns:
            List of studies (unchanged)
        """
        logger.warning(
            "ACE retrieval is not implemented. "
            "Please use PubGetRetriever instead."
        )
        return studies

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
            List of studies (unchanged)
        """
        logger.warning(
            "ACE validation is not implemented. "
            "Please use PubGetRetriever instead."
        )
        return studies