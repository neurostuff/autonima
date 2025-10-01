"""Prompt templates for annotation decisions."""

from typing import List, Optional
from .schema import AnalysisMetadata, AnnotationCriteriaConfig


def create_annotation_prompt(
    metadata: AnalysisMetadata,
    criteria: AnnotationCriteriaConfig
) -> str:
    """
    Create a prompt for the LLM to decide if an analysis should be included in an annotation.
    
    Args:
        metadata: Analysis metadata to include in the prompt
        criteria: Annotation criteria configuration
        
    Returns:
        Formatted prompt string
    """
    # Build the metadata section based on configured fields
    metadata_lines = []
    
    if "analysis_name" in criteria.metadata_fields and metadata.analysis_name:
        metadata_lines.append(f"- Name: {metadata.analysis_name}")
        
    if "analysis_description" in criteria.metadata_fields and metadata.analysis_description:
        metadata_lines.append(f"- Description: {metadata.analysis_description}")
        
    if "table_caption" in criteria.metadata_fields and metadata.table_caption:
        metadata_lines.append(f"- Table Caption: {metadata.table_caption}")
        
    if "table_footer" in criteria.metadata_fields and metadata.table_footer:
        metadata_lines.append(f"- Table Footer: {metadata.table_footer}")
        
    if "study_title" in criteria.metadata_fields and metadata.study_title:
        metadata_lines.append(f"- Study Title: {metadata.study_title}")
        
    if "study_abstract" in criteria.metadata_fields and metadata.study_abstract:
        metadata_lines.append(f"- Study Abstract: {metadata.study_abstract}")
        
    if "study_authors" in criteria.metadata_fields and metadata.study_authors:
        metadata_lines.append(f"- Study Authors: {', '.join(metadata.study_authors)}")
        
    if "study_journal" in criteria.metadata_fields and metadata.study_journal:
        metadata_lines.append(f"- Study Journal: {metadata.study_journal}")
        
    if "study_publication_date" in criteria.metadata_fields and metadata.study_publication_date:
        metadata_lines.append(f"- Study Publication Date: {metadata.study_publication_date}")
    
    # Add any custom fields
    for field_name, field_value in metadata.custom_fields.items():
        if field_name in criteria.metadata_fields and field_value:
            metadata_lines.append(f"- {field_name.replace('_', ' ').title()}: {field_value}")
    
    # Format inclusion and exclusion criteria
    inclusion_criteria_text = "\n".join([f"  - {c}" for c in criteria.inclusion_criteria])
    exclusion_criteria_text = "\n".join([f"  - {c}" for c in criteria.exclusion_criteria])
    
    # Create the prompt
    prompt = f"""
You are a neuroimaging meta-analysis expert evaluating whether an analysis meets specific inclusion criteria.

STUDY CONTEXT:
{chr(10).join(metadata_lines) if metadata_lines else "No metadata available"}

ANNOTATION CRITERIA: "{criteria.name}"
Description: {criteria.description or "No description provided"}

INCLUSION CRITERIA:
{inclusion_criteria_text or "No inclusion criteria specified"}

EXCLUSION CRITERIA:
{exclusion_criteria_text or "No exclusion criteria specified"}

Based on the provided information, should this analysis be included in the "{criteria.name}" annotation?

Respond with JSON:
{{
  "include": true/false,
  "reasoning": "Brief explanation of decision"
}}
"""
    
    return prompt.strip()