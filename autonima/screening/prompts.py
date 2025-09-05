"""Prompt library for LLM-powered systematic review screening."""

from typing import List
from ..models.types import Study, StudyStatus


class PromptLibrary:
    """Library of prompts for systematic review screening."""
    
    @staticmethod
    def get_base_prompt() -> str:
        """Get the base prompt template for screening."""
        base_prompt = """
You are a systematic review screener. Your task is to evaluate whether a study
should be INCLUDED or EXCLUDED in a systematic review based on its content
and the provided criteria.

IMPORTANT GUIDELINES:
1. Be objective and consistent in your evaluations
2. Base your decisions solely on the provided study information and criteria
3. When in doubt, provide a detailed explanation of your reasoning
4. Use the exact response format specified
""".strip()
        
        return base_prompt

    @staticmethod
    def get_abstract_screening_prompt(
        study: Study,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str],
        objective: str = None
    ) -> str:
        """Get the prompt for abstract screening."""
        base_prompt = PromptLibrary.get_base_prompt()
        
        content = study.abstract or "No abstract available"
        
        instructions = """
INSTRUCTIONS FOR ABSTRACT SCREENING:
1. Carefully evaluate the abstract against each criterion
2. If ANY exclusion criterion is clearly met, EXCLUDE the study
3. If the abstract provides INSUFFICIENT information to determine inclusion, 
   INCLUDE for full-text review
4. Only EXCLUDE if you are highly confident based on the abstract alone
5. Provide a confidence score (0.0-1.0) reflecting how certain you are about 
   the decision
6. Give a brief reason (max 100 words) explaining your decision
""".strip()

        prompt = f"""
{base_prompt}

STUDY INFORMATION:
Title: {study.title}
Abstract: {content}
Authors: {', '.join(study.authors)}
Journal: {study.journal}
Publication Date: {study.publication_date}
DOI: {study.doi or 'Not available'}

OBJECTIVE:
{objective or 'Not provided'}

INCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in inclusion_criteria)}

EXCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in exclusion_criteria)}

{instructions}
""".strip()

        return prompt

    @staticmethod
    def get_fulltext_screening_prompt(
        study: Study,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str],
        output_dir: str = "test_output",
        objective: str = None
    ) -> str:
        """Get the prompt for full-text screening."""
        base_prompt = PromptLibrary.get_base_prompt()
        
        # For full-text screening, we load the full text content
        if study.status in (
            [StudyStatus.FULLTEXT_RETRIEVED, StudyStatus.FULLTEXT_CACHED]
        ):
            try:
                
                content = study.load_full_text(output_dir=output_dir)

            except Exception:
                raise RuntimeError(
                    f"Failed to load full text for study {study.pmid}"
                )
        else:
            content = "No full text available"
        
        instructions = """
INSTRUCTIONS FOR FULL-TEXT SCREENING:
1. Carefully evaluate the full text against each inclusion criterion
2. Verify that ALL inclusion criteria are met
3. Check that NO exclusion criteria are violated
4. Pay special attention to study design, methods, participants, and outcomes
5. If the study meets all criteria, INCLUDE it
6. If ANY criterion is not met, EXCLUDE it
7. Provide a confidence score (0.0-1.0) reflecting your certainty
8. Give a detailed reason (max 200 words) explaining your decision
""".strip()

        prompt = f"""
{base_prompt}

STUDY INFORMATION:
Title: {study.title}
Full Text Content: {content}
Authors: {', '.join(study.authors)}
Journal: {study.journal}
Publication Date: {study.publication_date}
DOI: {study.doi or 'Not available'}

OBJECTIVE:
{objective or 'Not provided'}

INCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in inclusion_criteria)}

EXCLUSION CRITERIA:
{chr(10).join(f"- {criterion}" for criterion in exclusion_criteria)}

{instructions}
""".strip()

        return prompt