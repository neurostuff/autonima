"""Prompt templates for annotation decisions."""

from typing import List
from collections import defaultdict
from .schema import AnalysisMetadata, AnnotationCriteriaConfig, TableMetadata, StudyAnalysisGroup


def create_study_multi_annotation_prompt(
    study_group: StudyAnalysisGroup,
    criteria_list: List[AnnotationCriteriaConfig],
    metadata_fields: List[str] = None
) -> str:
    """
    Create a prompt for annotating all analyses in a study for a single annotation criteria.
    
    Args:
        study_group: The study group containing study metadata, tables, and analyses.
        criteria: Annotation criteria configuration.
        metadata_fields: List of metadata fields to include.
        
    Returns:
        Formatted prompt string.
    """
    # Use the provided metadata_fields or fall back to first criteria's
    fields_to_use = metadata_fields or (
        getattr(criteria_list[0], 'metadata_fields', None)
        if criteria_list else None
    ) or [
        "analysis_name",
        "analysis_description",
        "study_title"
    ]
    
    # Format study-level metadata
    study_metadata = []
    if "study_title" in fields_to_use and study_group.study_title:
        study_metadata.append(f"Study Title: {study_group.study_title}")
    if "study_abstract" in fields_to_use and study_group.study_abstract:
        study_metadata.append(f"Study Abstract: {study_group.study_abstract}")
    if "study_authors" in fields_to_use and study_group.study_authors:
        authors = ', '.join(study_group.study_authors)
        study_metadata.append(f"Study Authors: {authors}")
    if "study_journal" in fields_to_use and study_group.study_journal:
        study_metadata.append(f"Study Journal: {study_group.study_journal}")
    if "study_publication_date" in fields_to_use and study_group.study_publication_date:
        study_metadata.append(
            f"Publication Date: {study_group.study_publication_date}"
        )
    
    # Group analyses by table
    table_map = {t.table_id: t for t in study_group.tables}
    table_analyses = defaultdict(list)
    for analysis in study_group.analyses:
        table_analyses[analysis.table_id].append(analysis)
    
    # Format table and analysis sections
    table_sections = []
    for table_id, analyses in table_analyses.items():
        table = table_map.get(table_id, TableMetadata(table_id=table_id))
        table_section = [f"TABLE: {table.caption or 'Unnamed Table'}"]
        if table.footer:
            table_section.append(f"Footer: {table.footer}")
        
        for i, analysis in enumerate(analyses):
            analysis_lines = [f"Analysis {i+1}:"]
            if "analysis_name" in fields_to_use and analysis.analysis_name:
                analysis_lines.append(f"- Name: {analysis.analysis_name}")
            if ("analysis_description" in fields_to_use and
                    analysis.analysis_description):
                analysis_lines.append(
                    f"- Description: {analysis.analysis_description}"
                )
            if "study_title" in fields_to_use and analysis.study_title:
                analysis_lines.append(f"- Study Title: {analysis.study_title}")
            table_section.append("\n".join(analysis_lines))
        
        table_sections.append("\n".join(table_section))
    
    # Format criteria sections for all annotations
    criteria_sections = []
    for i, criteria in enumerate(criteria_list):
        # Format criteria with IDs if mapping is provided
        if criteria.criteria_mapping:
            inclusion_items = criteria.criteria_mapping.get(
                'inclusion', {}
            ).items()
            inclusion_list = [f"{id}: {text}" for id, text in inclusion_items]
            inclusion_text = "\n".join(inclusion_list)
            
            exclusion_items = criteria.criteria_mapping.get(
                'exclusion', {}
            ).items()
            exclusion_list = [f"{id}: {text}" for id, text in exclusion_items]
            exclusion_text = "\n".join(exclusion_list)
        else:
            inclusion_list = [f"  - {c}" for c in criteria.inclusion_criteria]
            inclusion_text = "\n".join(inclusion_list)
            exclusion_list = [f"  - {c}" for c in criteria.exclusion_criteria]
            exclusion_text = "\n".join(exclusion_list)
        
        criteria_section = f"""
            ANNOTATION {i+1}: "{criteria.name}"
            Description: {criteria.description or "No description provided"}

            INCLUSION CRITERIA:
            {inclusion_text or "No inclusion criteria specified"}

            EXCLUSION CRITERIA:
            {exclusion_text or "No exclusion criteria specified"}
            """
        criteria_sections.append(criteria_section)
    
    # Prepare context strings
    study_context_str = "\n".join(study_metadata) if study_metadata else "No study context available"
    tables_str = "\n\n".join(table_sections) if table_sections else "No tables available"
    criteria_str = "\n".join(criteria_sections)
    
    # Create the prompt
    prompt = """
You are a neuroimaging meta-analysis expert evaluating all analyses from an entire study for multiple annotations.

STUDY CONTEXT:
{study_context_str}

TABLES AND ANALYSES:
{tables_str}

{criteria_str}

For EACH analysis and EACH annotation, provide:
- Include decision (true/false)
- Reasoning
- Applied inclusion/exclusion criteria

Output JSON format:
{{
  "study_id": "{study_id}",
  "decisions": [
    {{
      "analysis_id": "id1",
      "annotations": [
        {{
          "annotation_name": "annotation_1",
          "include": true,
          "reasoning": "...",
          "inclusion_criteria_applied": ["I1"],
          "exclusion_criteria_applied": []
        }},
        {{
          "annotation_name": "annotation_2",
          "include": false,
          "reasoning": "...",
          "inclusion_criteria_applied": [],
          "exclusion_criteria_applied": ["E3"]
        }}
      ]
    }},
    {{
      "analysis_id": "id2",
      "annotations": [
        {{
          "annotation_name": "annotation_1",
          "include": true,
          "reasoning": "...",
          "inclusion_criteria_applied": ["I1"],
          "exclusion_criteria_applied": []
        }},
        {{
          "annotation_name": "annotation_2",
          "include": true,
          "reasoning": "...",
          "inclusion_criteria_applied": ["I2"],
          "exclusion_criteria_applied": []
        }}
      ]
    }}
  ]
}}
""".format(
        study_context_str=study_context_str,
        tables_str=tables_str,
        criteria_str=criteria_str,
        study_id=study_group.study_id
    ).strip()

    from pdb import set_trace; set_trace()

    return prompt


def create_multi_annotation_prompt(
    metadata: AnalysisMetadata,
    criteria_list: List[AnnotationCriteriaConfig],
    metadata_fields: List[str] = None
) -> str:
    """
    Create a prompt for the LLM to decide if an analysis should be included 
    in multiple annotations at once.
    
    Args:
        metadata: Analysis metadata to include in the prompt
        criteria_list: List of annotation criteria configurations
        metadata_fields: List of metadata fields to include
        
    Returns:
        Formatted prompt string
    """
    # Use the provided metadata_fields or fall back to first criteria's
    fields_to_use = metadata_fields or (
        getattr(criteria_list[0], 'metadata_fields', None)
        if criteria_list else None
    ) or [
        "analysis_name",
        "analysis_description",
        "study_title"
    ]
    
    # Build the metadata section based on configured fields
    metadata_lines = []
    
    if "analysis_name" in fields_to_use and metadata.analysis_name:
        metadata_lines.append(f"- Name: {metadata.analysis_name}")
        
    if ("analysis_description" in fields_to_use and
            metadata.analysis_description):
        metadata_lines.append(
            f"- Description: {metadata.analysis_description}"
        )
        
    if "study_title" in fields_to_use and metadata.study_title:
        metadata_lines.append(f"- Study Title: {metadata.study_title}")
        
    if "study_abstract" in fields_to_use and metadata.study_abstract:
        metadata_lines.append(f"- Study Abstract: {metadata.study_abstract}")
        
    if "study_authors" in fields_to_use and metadata.study_authors:
        authors = ', '.join(metadata.study_authors)
        metadata_lines.append(f"- Study Authors: {authors}")
        
    if "study_journal" in fields_to_use and metadata.study_journal:
        metadata_lines.append(f"- Study Journal: {metadata.study_journal}")
        
    if ("study_publication_date" in fields_to_use and
            metadata.study_publication_date):
        date_str = metadata.study_publication_date
        metadata_lines.append(f"- Study Publication Date: {date_str}")
    
    if "study_fulltext" in fields_to_use and metadata.study_fulltext:
        metadata_lines.append(
            f"- Study Full Text: {metadata.study_fulltext}"
        )
    
    # Add any custom fields
    for field_name, field_value in metadata.custom_fields.items():
        if field_name in fields_to_use and field_value:
            formatted_name = field_name.replace('_', ' ').title()
            metadata_lines.append(f"- {formatted_name}: {field_value}")
    
    # Build criteria sections for all annotations
    criteria_sections = []
    for i, criteria in enumerate(criteria_list):
        # Format criteria with IDs if mapping is provided
        if criteria.criteria_mapping:
            inclusion_items = criteria.criteria_mapping.get(
                'inclusion', {}
            ).items()
            inclusion_list = [f"{id}: {text}" for id, text in inclusion_items]
            inclusion_text = "\n".join(inclusion_list)
            
            exclusion_items = criteria.criteria_mapping.get(
                'exclusion', {}
            ).items()
            exclusion_list = [f"{id}: {text}" for id, text in exclusion_items]
            exclusion_text = "\n".join(exclusion_list)
        else:
            inclusion_list = [f"  - {c}" for c in criteria.inclusion_criteria]
            inclusion_text = "\n".join(inclusion_list)
            exclusion_list = [f"  - {c}" for c in criteria.exclusion_criteria]
            exclusion_text = "\n".join(exclusion_list)
        
        criteria_section = f"""
            ANNOTATION {i+1}: "{criteria.name}"
            Description: {criteria.description or "No description provided"}

            INCLUSION CRITERIA:
            {inclusion_text or "No inclusion criteria specified"}

            EXCLUSION CRITERIA:
            {exclusion_text or "No exclusion criteria specified"}
            """
        criteria_sections.append(criteria_section)
    
    # Create the prompt
    prompt = f"""
You are a neuroimaging meta-analysis expert evaluating whether an analysis meets 
specific inclusion criteria for multiple annotations simultaneously.

The following analysis has been extracted from within a table of a published 
fMRI/neuroimaging article. You will be provided with metadata about the 
analysis, the table it was extracted from, and the study it belongs to. Note 
that since each table may have contained multiple analyses, the table caption 
may describe multiple analyses that are not relevant to this specific analysis.
As such, while taking into account the table caption, please focus primarily on
the analysis name and description for your decision.

STUDY CONTEXT:
{chr(10).join(metadata_lines) if metadata_lines else "No metadata available"}

{chr(10).join(criteria_sections)}

Based on the provided information, should this analysis be included in each of the 
annotations listed above?

IMPORTANT: For each annotation, you must specify which specific criteria IDs apply
to this analysis.
- For included analyses: List the inclusion criteria IDs that are satisfied
- For excluded analyses: List the exclusion criteria IDs that apply

Respond with JSON array, one object for each annotation in the same order:
[
  {{
    "annotation_name": "annotation_1_name",
    "include": true/false,
    "reasoning": "Brief explanation of decision",
    "inclusion_criteria_applied": ["I1", "I2"],
    "exclusion_criteria_applied": []
  }},
  {{
    "annotation_name": "annotation_2_name",
    "include": true/false,
    "reasoning": "Brief explanation of decision",
    "inclusion_criteria_applied": ["I3", "I4"],
    "exclusion_criteria_applied": []
  }}
]
"""
    
    return prompt.strip()