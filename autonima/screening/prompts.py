"""Prompt library for LLM-powered systematic review screening."""

from typing import List
from ..models.types import Study, StudyStatus
from ..utils.criteria import CriteriaMapping
from ..cache_versions import (
    ABSTRACT_SCREENING_PROMPT_VERSION,
    FULLTEXT_SCREENING_PROMPT_VERSION,
)


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
2. Carefully evaluate the abstract against each inclusion and exclusion criterion
3. Apply this decision rule:
   - EXCLUDE if ANY exclusion criterion is clearly met, OR if one or more
     inclusion criteria are clearly not met.
   - INCLUDE for full-text review when exclusion criteria are not met and
     inclusion status is uncertain from abstract-only evidence.
4. If the abstract provides INSUFFICIENT information to determine inclusion,
   INCLUDE for full-text review
5. Only EXCLUDE if you are highly confident based on the abstract alone
{confidence_instruction}{reason_instruction}
{additional_instructions_text}

IMPORTANT: In your response, you must fully encode criteria coverage using IDs.
- inclusion_criteria_applied: include ALL inclusion criteria IDs that are satisfied.
- exclusion_criteria_applied: include ALL exclusion criteria IDs that are met.
- Always populate BOTH arrays for every decision (included or excluded). Use []
  only when none apply.
- For excluded studies, the reason must explicitly name:
  (a) exclusion IDs met (if any),
  (b) inclusion IDs met, and
  (c) inclusion IDs not met or not demonstrated.
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
        if study.fulltext_available:
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
                "9. Provide a confidence score (0.0-1.0) reflecting your "
                "certainty\n"
            )
            reason_instruction = (
                "10. Give a detailed reason (max 200 words) explaining your "
                "decision"
            )
        else:
            confidence_instruction = ""
            reason_instruction = (
                "9. Give a detailed reason (max 200 words) explaining your "
                "decision"
            )
        
        # Add additional instructions if provided
        additional_instructions_text = ""
        if additional_instructions:
            additional_instructions_text = f"\n{additional_instructions}\n"

        # A study read from a document source is screened from that document, and the prompt
        # must not tell the model it is reading the article.
        document = getattr(study, "document", None)
        if document is not None:
            return PromptLibrary._get_document_screening_prompt(
                study=study,
                content=content,
                document_description=document.description,
                inclusion_text=inclusion_text,
                exclusion_text=exclusion_text,
                objective=objective,
                confidence_instruction=confidence_instruction,
                reason_instruction=reason_instruction,
                additional_instructions_text=additional_instructions_text,
            )
        
        instructions = f"""
INSTRUCTIONS FOR FULL-TEXT SCREENING:
1. Ensure the study addresses the review objective
2. Carefully evaluate the full text against each inclusion criterion
3. Verify that ALL inclusion criteria are met
4. Check that NO exclusion criteria are violated
5. Pay special attention to study design, methods, participants, and outcomes
6. If the study meets all criteria, INCLUDE it
7. EXCLUDE if ANY exclusion criterion is met OR if ANY inclusion criterion is not met
8. Determine whether the provided "full text" is actually complete.
   Set fulltext_incomplete=true when content is incomplete and only provides
   title/abstract/metadata/references or another truncated subset.
   Consider full text complete only when core article content is present
   (e.g., introduction/background, methods, results, and discussion).
   Missing supplemental material is acceptable.
{confidence_instruction}{reason_instruction}
{additional_instructions_text}

IMPORTANT: In your response, you must fully encode criteria coverage using IDs.
- inclusion_criteria_applied: include ALL inclusion criteria IDs that are satisfied.
- exclusion_criteria_applied: include ALL exclusion criteria IDs that are met.
- Always populate BOTH arrays for every decision (included or excluded). Use []
  only when none apply.
- For excluded studies, the reason must explicitly name:
  (a) exclusion IDs met (if any),
  (b) inclusion IDs met, and
  (c) inclusion IDs not met.
When fulltext_incomplete=true, this flag takes precedence in downstream logic.
Respond with the exact JSON format specified, including fulltext_incomplete,
inclusion_criteria_applied, and exclusion_criteria_applied fields.
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

    @staticmethod
    def _get_document_screening_prompt(
        study: Study,
        content: str,
        document_description: str,
        inclusion_text: str,
        exclusion_text: str,
        objective: str,
        confidence_instruction: str,
        reason_instruction: str,
        additional_instructions_text: str,
    ) -> str:
        """Full-text screening prompt for a study read from a document source.

        The article prompt's instruction 8 asks whether introduction, methods, results and
        discussion are present. A derived document has none of them and would be flagged
        incomplete every time, so here incompleteness means only that the document itself
        reports the article could not be read.
        """
        base_prompt = PromptLibrary.get_base_prompt()

        instructions = f"""
INSTRUCTIONS FOR FULL-TEXT SCREENING FROM A STUDY DOCUMENT:
1. Ensure the study addresses the review objective
2. Carefully evaluate the study document against each inclusion criterion
3. Verify that ALL inclusion criteria are met
4. Check that NO exclusion criteria are violated
5. Pay special attention to study design, methods, participants, and outcomes
6. If the study meets all criteria, INCLUDE it
7. EXCLUDE if ANY exclusion criterion is met OR if ANY inclusion criterion is not met
8. Set fulltext_incomplete=true ONLY when the study document itself shows that the
   article could not be read -- for example, it reports a publisher access notice, or
   states that the article body was unavailable to whatever produced the document.
   Do NOT set it because the document has no introduction, methods, results or
   discussion sections, or because a field is empty or marked as not reported: the
   document is a derived representation of the article, and it is as complete as the
   process that produced it.
{confidence_instruction}{reason_instruction}
{additional_instructions_text}

IMPORTANT: In your response, you must fully encode criteria coverage using IDs.
- inclusion_criteria_applied: include ALL inclusion criteria IDs that are satisfied.
- exclusion_criteria_applied: include ALL exclusion criteria IDs that are met.
- Always populate BOTH arrays for every decision (included or excluded). Use []
  only when none apply.
- For excluded studies, the reason must explicitly name:
  (a) exclusion IDs met (if any),
  (b) inclusion IDs met, and
  (c) inclusion IDs not met.
When fulltext_incomplete=true, this flag takes precedence in downstream logic.
Respond with the exact JSON format specified, including fulltext_incomplete,
inclusion_criteria_applied, and exclusion_criteria_applied fields.
""".strip()

        prompt = f"""
{base_prompt}

ABOUT THE STUDY DOCUMENT:
The study document below is not the article's full text. It stands in for the full
text in this review, so judge every criterion from what the document reports.
Document: {document_description}

STUDY INFORMATION:
Title: {study.title}
Study Document: {content}
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
