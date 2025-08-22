"""Full-text screening implementation for systematic reviews."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from .llm_screener import LLMScreener
from ..models.types import Study, ScreeningResult, ScreeningConfig

logger = logging.getLogger(__name__)


class FullTextScreener(LLMScreener):
    """Specialized screener for full-text screening."""

    def __init__(self, config: ScreeningConfig):
        """Initialize the full-text screener."""
        super().__init__(config)
        self.screening_type = "fulltext"

    async def screen_abstracts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Full-text screener doesn't handle abstract screening.

        Args:
            studies: List of studies (ignored)

        Returns:
            Empty list with warning
        """
        logger.warning("FullTextScreener does not support abstract screening")
        return []

    async def screen_fulltexts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Screen full-text articles for inclusion/exclusion.

        Args:
            studies: List of studies with full text to screen

        Returns:
            List of screening results
        """
        logger.info(f"Starting full-text screening for {len(studies)} studies")

        # Filter to only studies that have full text
        screenable_studies = [
            s for s in studies
            if s.full_text_path and Path(s.full_text_path).exists()
        ]

        if len(screenable_studies) < len(studies):
            logger.warning(
                f"Skipping {len(studies) - len(screenable_studies)} studies without full text"
            )

        # Use parent class method for actual screening
        return await super().screen_fulltexts(screenable_studies)

    def build_fulltext_screening_prompt(
        self,
        study: Study,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str]
    ) -> str:
        """
        Build a specialized prompt for full-text screening.

        Args:
            study: Study to screen
            inclusion_criteria: List of inclusion criteria
            exclusion_criteria: List of exclusion criteria

        Returns:
            Formatted prompt string
        """
        # In a real implementation, you would load and extract text from study.full_text_path
        full_text_content = f"[Full text content would be loaded from {study.full_text_path}]"

        prompt = ".1f"f"""
You are a systematic review full-text screener. Your task is to evaluate whether a study should be INCLUDED or EXCLUDED in a systematic review based on its full text and the provided criteria.

**IMPORTANT**: This is FULL-TEXT screening. You have access to the complete article and should make definitive decisions based on all available information.

STUDY INFORMATION:
Title: {study.title}
Abstract: {study.abstract}
Authors: {', '.join(study.authors)}
Journal: {study.journal}
Publication Date: {study.publication_date}
DOI: {study.doi or 'Not available'}

FULL TEXT CONTENT:
{full_text_content}

INCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in inclusion_criteria)}

EXCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in exclusion_criteria)}

SCREENING INSTRUCTIONS:
1. Carefully evaluate the full text against each inclusion criterion
2. Verify that ALL inclusion criteria are met
3. Check that NO exclusion criteria are violated
4. Pay special attention to study design, methods, participants, and outcomes
5. If the study meets all criteria, INCLUDE it
6. If ANY criterion is not met, EXCLUDE it
7. Provide a confidence score (0.0-1.0) reflecting your certainty
8. Give a detailed reason (max 200 words) explaining your decision

RESPONSE FORMAT:
DECISION: [INCLUDED/EXCLUDED]
CONFIDENCE: [0.0-1.0]
REASON: [Detailed explanation]
"""

        return prompt.strip()

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text content from PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            # This is a placeholder for PDF text extraction
            # In a real implementation, you would use PyPDF2, pdfplumber, or similar
            logger.info(f"Extracting text from {pdf_path}")

            # Mock extraction for now
            return f"[Extracted text from {pdf_path} would appear here]"

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return None

    def get_screening_info(self) -> Dict[str, Any]:
        """Get information about the full-text screening configuration."""
        base_info = super().get_screening_info()
        base_info.update({
            "screening_type": "fulltext",
            "specialized_prompt": True,
            "requires_full_text": True,
            "pdf_extraction": "mock"  # Would be "pypdf2" or similar in real implementation
        })
        return base_info