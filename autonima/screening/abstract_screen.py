"""Abstract screening implementation for systematic reviews."""

import logging
from typing import List, Dict, Any
from .llm_screener import LLMScreener
from ..models.types import Study, ScreeningResult, ScreeningConfig

logger = logging.getLogger(__name__)


class AbstractScreener(LLMScreener):
    """Specialized screener for abstract-level screening."""

    def __init__(self, config: ScreeningConfig):
        """Initialize the abstract screener."""
        super().__init__(config)
        self.screening_type = "abstract"

    async def screen_abstracts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Screen study abstracts for inclusion/exclusion.

        Args:
            studies: List of studies to screen (abstracts only)

        Returns:
            List of screening results
        """
        logger.info(f"Starting abstract screening for {len(studies)} studies")

        # Filter to only studies that have abstracts
        screenable_studies = [s for s in studies if s.abstract and s.abstract.strip()]

        if len(screenable_studies) < len(studies):
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} studies without abstracts"
            )

        # Use parent class method for actual screening
        return await super().screen_abstracts(screenable_studies)

    async def screen_fulltexts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Abstract screener doesn't handle full-text screening.

        Args:
            studies: List of studies (ignored)

        Returns:
            Empty list with warning
        """
        logger.warning("AbstractScreener does not support full-text screening")
        return []

    def build_abstract_screening_prompt(
        self,
        study: Study,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str]
    ) -> str:
        """
        Build a specialized prompt for abstract screening.

        Args:
            study: Study to screen
            inclusion_criteria: List of inclusion criteria
            exclusion_criteria: List of exclusion criteria

        Returns:
            Formatted prompt string
        """
        prompt = ".1f"f"""
You are a systematic review abstract screener. Your task is to evaluate whether a study should be INCLUDED or EXCLUDED in a systematic review based on its abstract and the provided criteria.

**IMPORTANT**: This is ABSTRACT screening only. You should NOT require information that would only be available in the full text. If you're unsure based on the abstract alone, err on the side of INCLUSION to allow full-text review.

STUDY ABSTRACT:
Title: {study.title}
Abstract: {study.abstract}
Authors: {', '.join(study.authors)}
Journal: {study.journal}
Publication Date: {study.publication_date}
DOI: {study.doi or 'Not available'}

INCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in inclusion_criteria)}

EXCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in exclusion_criteria)}

SCREENING INSTRUCTIONS:
1. Carefully evaluate the abstract against each criterion
2. If ANY exclusion criterion is clearly met, EXCLUDE the study
3. If the abstract provides INSUFFICIENT information to determine inclusion, INCLUDE for full-text review
4. Only EXCLUDE if you are highly confident based on the abstract alone
5. Provide a confidence score (0.0-1.0) reflecting how certain you are about the decision
6. Give a brief reason (max 100 words) explaining your decision

RESPONSE FORMAT:
DECISION: [INCLUDED/EXCLUDED]
CONFIDENCE: [0.0-1.0]
REASON: [Brief explanation]
"""

        return prompt.strip()

    def get_screening_info(self) -> Dict[str, Any]:
        """Get information about the abstract screening configuration."""
        base_info = super().get_screening_info()
        base_info.update({
            "screening_type": "abstract",
            "specialized_prompt": True,
            "abstract_only": True
        })
        return base_info