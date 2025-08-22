"""Base interface for screening engines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models.types import Study, ScreeningConfig, ScreeningResult, StudyStatus


class ScreeningEngine(ABC):
    """Abstract base class for screening engines."""

    def __init__(self, config: ScreeningConfig):
        """Initialize the screening engine with configuration."""
        self.config = config

    @abstractmethod
    async def screen_abstracts(self, studies: List[Study]) -> List[ScreeningResult]:
        """
        Screen study abstracts for inclusion/exclusion.

        Args:
            studies: List of studies to screen

        Returns:
            List of screening results
        """
        pass

    @abstractmethod
    async def screen_fulltexts(self, studies: List[Study]) -> List[ScreeningResult]:
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

    def build_screening_prompt(
        self,
        study: Study,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str],
        screening_type: str = "abstract"
    ) -> str:
        """
        Build a screening prompt for the LLM.

        Args:
            study: Study to screen
            inclusion_criteria: List of inclusion criteria
            exclusion_criteria: List of exclusion criteria
            screening_type: Type of screening ("abstract" or "fulltext")

        Returns:
            Formatted prompt string
        """
        content = study.abstract if screening_type == "abstract" else f"[Full text would be here for {study.pmid}]"

        prompt = f"""
You are a systematic review screening assistant. Your task is to evaluate whether a study should be INCLUDED or EXCLUDED based on the provided criteria.

STUDY INFORMATION:
Title: {study.title}
Abstract: {content}
Authors: {', '.join(study.authors)}
Journal: {study.journal}
Publication Date: {study.publication_date}
DOI: {study.doi or 'Not available'}

INCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in inclusion_criteria)}

EXCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in exclusion_criteria)}

INSTRUCTIONS:
1. Carefully analyze the study against each inclusion and exclusion criterion
2. If ANY exclusion criterion applies, EXCLUDE the study
3. If ALL inclusion criteria are met AND no exclusion criteria apply, INCLUDE the study
4. Provide a confidence score between 0.0 and 1.0
5. Give a brief reason for your decision (max 100 words)

RESPONSE FORMAT:
DECISION: [INCLUDED/EXCLUDED]
CONFIDENCE: [0.0-1.0]
REASON: [Brief explanation]

Please provide your response in the exact format specified above.
"""

        return prompt.strip()

    def parse_screening_response(self, response: str) -> Tuple[StudyStatus, float, str]:
        """
        Parse LLM response into structured result.

        Args:
            response: Raw LLM response string

        Returns:
            Tuple of (decision, confidence, reason)
        """
        lines = response.strip().split('\n')
        decision = StudyStatus.EXCLUDED
        confidence = 0.5
        reason = "Unable to parse response"

        for line in lines:
            line = line.strip()
            if line.startswith('DECISION:'):
                decision_text = line.replace('DECISION:', '').strip().upper()
                if 'INCLUDE' in decision_text:
                    decision = StudyStatus.INCLUDED
                else:
                    decision = StudyStatus.EXCLUDED
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except ValueError:
                    confidence = 0.5
            elif line.startswith('REASON:'):
                reason = line.replace('REASON:', '').strip()

        return decision, confidence, reason