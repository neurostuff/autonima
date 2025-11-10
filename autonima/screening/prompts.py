"""Prompt library for LLM-powered systematic review screening."""

from typing import List
from ..models.types import Study, StudyStatus
from ..utils.criteria import CriteriaMapping


class PromptLibrary:
    """Library of prompts for systematic review screening."""
    
    @staticmethod
    def get_base_prompt() -> str:
        """Get the base prompt template for screening."""
        base_prompt = """
You are a systematic review screener. Your task is to evaluate whether a study
should be INCLUDED or EXCLUDED in a systematic review based on its content,
the meta-analysis objective, and the provided criteria.

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
        criteria_mapping: CriteriaMapping = None,
        objective: str = None,
        confidence_reporting: bool = False,
        additional_instructions: str = None
    ) -> str:
        """Get the prompt for abstract screening."""
        base_prompt = PromptLibrary.get_base_prompt()
        
        content = study.abstract or "No abstract available"
        
        # Format criteria with IDs if mapping is provided
        if criteria_mapping:
            inclusion_text = "\n".join([
                f"{id}: {text}" 
                for id, text in criteria_mapping.inclusion.items()
            ])
            exclusion_text = "\n".join([
                f"{id}: {text}" 
                for id, text in criteria_mapping.exclusion.items()
            ])
        else:
            inclusion_text = "\n".join(f"- {criterion}" for criterion in inclusion_criteria)
            exclusion_text = "\n".join(f"- {criterion}" for criterion in exclusion_criteria)
        
        # Check if confidence reporting is enabled
        if confidence_reporting:
            confidence_instruction = (
                "6. Provide a confidence score (0.0-1.0) reflecting how certain "
                "you are about\n   the decision\n"
            )
            reason_instruction = (
                "7. Give a brief reason (max 100 words) explaining your "
                "decision"
            )
        else:
            confidence_instruction = ""
            reason_instruction = (
                "6. Give a brief reason (max 100 words) explaining your "
                "decision"
            )
        
        # Add additional instructions if provided
        additional_instructions_text = ""
        if additional_instructions:
            additional_instructions_text = f"\n{additional_instructions}\n"
        
        instructions = f"""
INSTRUCTIONS FOR ABSTRACT SCREENING:
1. Ensure the study addresses the review objective
2. Carefully evaluate the abstract against each criterion
3. If ANY exclusion criterion is clearly met, EXCLUDE the study
4. If the abstract provides INSUFFICIENT information to determine inclusion,
   INCLUDE for full-text review
5. Only EXCLUDE if you are highly confident based on the abstract alone
{confidence_instruction}{reason_instruction}
{additional_instructions_text}

IMPORTANT: In your response, you must specify which specific criteria IDs apply to this study.
- For included studies: List the inclusion criteria IDs that are satisfied (e.g., ["I1", "I2"])
- For excluded studies: List the exclusion criteria IDs that apply (e.g., ["E1"])
Respond with the exact JSON format specified, including the inclusion_criteria_applied and exclusion_criteria_applied fields.
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

META-ANALYSIS OBJECTIVE:
{objective or 'Not provided'}

INCLUSION CRITERIA:
{inclusion_text}

EXCLUSION CRITERIA:
{exclusion_text}

{instructions}
""".strip()

        return prompt

    @staticmethod
    def get_fulltext_screening_prompt(
        study: Study,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str],
        output_dir: str,
        criteria_mapping: CriteriaMapping = None,
        objective: str = None,
        confidence_reporting: bool = False,
        additional_instructions: str = None
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
        
        # Format criteria with IDs if mapping is provided
        if criteria_mapping:
            inclusion_text = "\n".join([
                f"{id}: {text}" 
                for id, text in criteria_mapping.inclusion.items()
            ])
            exclusion_text = "\n".join([
                f"{id}: {text}" 
                for id, text in criteria_mapping.exclusion.items()
            ])
        else:
            inclusion_text = "\n".join(f"- {criterion}" for criterion in inclusion_criteria)
            exclusion_text = "\n".join(f"- {criterion}" for criterion in exclusion_criteria)
        
        # Check if confidence reporting is enabled
        if confidence_reporting:
            confidence_instruction = (
                "8. Provide a confidence score (0.0-1.0) reflecting your "
                "certainty\n"
            )
            reason_instruction = (
                "9. Give a detailed reason (max 200 words) explaining your "
                "decision"
            )
        else:
            confidence_instruction = ""
            reason_instruction = (
                "8. Give a detailed reason (max 200 words) explaining your "
                "decision"
            )
        
        # Add additional instructions if provided
        additional_instructions_text = ""
        if additional_instructions:
            additional_instructions_text = f"\n{additional_instructions}\n"
        
        instructions = f"""
INSTRUCTIONS FOR FULL-TEXT SCREENING:
1. Ensure the study addresses the review objective
2. Carefully evaluate the full text against each inclusion criterion
3. Verify that ALL inclusion criteria are met
4. Check that NO exclusion criteria are violated
5. Pay special attention to study design, methods, participants, and outcomes
6. If the study meets all criteria, INCLUDE it
7. If ANY criterion is not met, EXCLUDE it
{confidence_instruction}{reason_instruction}
{additional_instructions_text}

IMPORTANT: In your response, you must specify which specific criteria IDs apply to this study.
- For included studies: List the inclusion criteria IDs that are satisfied (e.g., ["I1", "I2"])
- For excluded studies: List the exclusion criteria IDs that apply (e.g., ["E1"])
Respond with the exact JSON format specified, including the inclusion_criteria_applied and exclusion_criteria_applied fields.
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

META-ANALYSIS OBJECTIVE:
{objective or 'Not provided'}

INCLUSION CRITERIA:
{inclusion_text}

EXCLUSION CRITERIA:
{exclusion_text}

{instructions}
""".strip()

        return prompt