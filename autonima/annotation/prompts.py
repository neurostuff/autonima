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
    Create a prompt for annotating all analyses in a study for multiple annotation criteria.
    High-impact reliability fixes:
      1) Remove placeholder annotation labels (annotation_1, etc.) from the output example.
      2) Add an explicit allowed-values ENUM for annotation_name.
      3) Add a fixed mapping from ANNOTATION N -> canonical criteria.name.
      4) Add a lightweight self-check to prevent placeholders.
    """
    if metadata_fields is None:
        metadata_fields = []

    # -------------------------
    # Study-level metadata
    # -------------------------
    study_metadata = []

    if "study_title" in metadata_fields and study_group.study_title:
        study_metadata.append(
            f"STUDY (APPLIES TO ALL ANALYSES):\nStudy Title: {study_group.study_title}"
        )

    if ("study_abstract" in metadata_fields
        and study_group.study_abstract
    ):
        study_metadata.append(
            f"Study Abstract: {study_group.study_abstract}"
        )
    if ("study_fulltext" in metadata_fields
          and study_group.study_fulltext):
        study_metadata.append(
            f"Study Full Text: {study_group.study_fulltext}"
        )

    # -------------------------
    # Group analyses by table
    # -------------------------
    table_map = {t.table_id: t for t in study_group.tables}
    table_analyses = defaultdict(list)
    for analysis in study_group.analyses:
        table_analyses[analysis.table_id].append(analysis)

    # -------------------------
    # Tables and analyses section
    # -------------------------
    table_sections = []
    for table_id, analyses in table_analyses.items():
        table = table_map.get(table_id, TableMetadata(table_id=table_id))
        table_section = [f"TABLE: {table.caption or 'Unnamed Table'}"]
        if table.footer:
            table_section.append(f"Footer: {table.footer}")

        for analysis in analyses:
            analysis_lines = [f"Analysis ID: {analysis.analysis_id}"]
            if "analysis_name" in metadata_fields and analysis.analysis_name:
                analysis_lines.append(f"- Name: {analysis.analysis_name}")
            if "analysis_description" in metadata_fields and analysis.analysis_description:
                analysis_lines.append(f"- Description: {analysis.analysis_description}")
            table_section.append("\n".join(analysis_lines))

        table_sections.append("\n".join(table_section))

    # -------------------------
    # Criteria sections + mapping/ENUM
    # -------------------------
    criteria_sections = []
    allowed_annotation_names = [c.name for c in criteria_list]

    mapping_lines = [f'ANNOTATION {i+1} → "{c.name}"' for i, c in enumerate(criteria_list)]
    annotation_mapping_str = "\n".join(mapping_lines)
    allowed_names_str = "\n".join([f'- "{name}"' for name in allowed_annotation_names])

    # Build global criteria block once. These IDs are shared across annotations.
    global_inclusion_map = {}
    global_exclusion_map = {}
    for criteria in criteria_list:
        mapping = criteria.criteria_mapping or {}
        global_inclusion_map.update(mapping.get("global_inclusion", {}))
        global_exclusion_map.update(mapping.get("global_exclusion", {}))

    if global_inclusion_map:
        global_inclusion_text = "\n".join(
            [f"{cid}: {text}" for cid, text in global_inclusion_map.items()]
        )
    else:
        global_inclusion_text = "No global inclusion criteria specified"

    if global_exclusion_map:
        global_exclusion_text = "\n".join(
            [f"{cid}: {text}" for cid, text in global_exclusion_map.items()]
        )
    else:
        global_exclusion_text = "No global exclusion criteria specified"

    for i, criteria in enumerate(criteria_list):
        mapping = criteria.criteria_mapping or {}
        local_inclusion = mapping.get("local_inclusion")
        local_exclusion = mapping.get("local_exclusion")

        # Fallback for unit tests and configs that do not include explicit local blocks.
        if local_inclusion is None:
            local_inclusion = mapping.get("inclusion", {})
        if local_exclusion is None:
            local_exclusion = mapping.get("exclusion", {})

        if local_inclusion:
            local_inclusion_text = "\n".join(
                [f"{cid}: {text}" for cid, text in local_inclusion.items()]
            )
        else:
            local_inclusion_text = "\n".join(
                [f"  - {c}" for c in criteria.inclusion_criteria]
            ) or "No local inclusion criteria specified"

        if local_exclusion:
            local_exclusion_text = "\n".join(
                [f"{cid}: {text}" for cid, text in local_exclusion.items()]
            )
        else:
            local_exclusion_text = "\n".join(
                [f"  - {c}" for c in criteria.exclusion_criteria]
            ) or "No local exclusion criteria specified"

        criteria_section = f"""
ANNOTATION {i+1}: "{criteria.name}"
Description: {criteria.description or "No description provided"}

LOCAL INCLUSION CRITERIA:
{local_inclusion_text}

LOCAL EXCLUSION CRITERIA:
{local_exclusion_text}
""".strip()
        criteria_sections.append(criteria_section)

    # -------------------------
    # Prepare context strings
    # -------------------------
    study_context_str = "\n".join(study_metadata) if study_metadata else "No study context available"
    tables_str = "\n\n".join(table_sections) if table_sections else "No tables available"
    criteria_str = "\n\n".join(criteria_sections)

    # -------------------------
    # Output examples: canonical names + namespaced criteria IDs
    # -------------------------
    global_inc_example_id = next(iter(global_inclusion_map.keys()), "GLOBAL_I1")
    global_exc_example_id = next(iter(global_exclusion_map.keys()), "GLOBAL_E1")

    example_annotations = []
    for i, criteria in enumerate(criteria_list):
        mapping = criteria.criteria_mapping or {}
        local_inc = mapping.get("local_inclusion") or mapping.get("inclusion") or {}
        local_exc = mapping.get("local_exclusion") or mapping.get("exclusion") or {}
        local_inc_example_id = next(
            iter(local_inc.keys()),
            f'{criteria.name.upper().replace("-", "_").replace(" ", "_")}_I1'
        )
        local_exc_example_id = next(iter(local_exc.keys()), "")

        if i == 0:
            example_annotations.append(f"""{{
          "annotation_name": "{criteria.name}",
          "include": true,
          "reasoning": "Global and local inclusion criteria are met for this construct.",
          "inclusion_criteria_applied": ["{global_inc_example_id}", "{local_inc_example_id}"],
          "exclusion_criteria_applied": []
        }}""")
            continue

        if local_exc_example_id:
            exclusion_example_ids = f'["{global_exc_example_id}", "{local_exc_example_id}"]'
            example_reasoning = (
                f"Excluded because {global_exc_example_id} and/or {local_exc_example_id} apply."
            )
        else:
            exclusion_example_ids = "[]"
            example_reasoning = (
                f"Excluded because missing inclusion criteria: "
                f"{global_inc_example_id}, {local_inc_example_id}."
            )

        example_annotations.append(f"""{{
          "annotation_name": "{criteria.name}",
          "include": false,
          "reasoning": "{example_reasoning}",
          "inclusion_criteria_applied": [],
          "exclusion_criteria_applied": {exclusion_example_ids}
        }}""")

    example_annotations_str = ",\n        ".join(example_annotations)

    prompt = f"""
You are a neuroimaging meta-analysis expert evaluating all analyses from an entire study for multiple annotations.

STUDY CONTEXT:
NOTE:
The Study Title and Abstract above apply to ALL analyses below and should be used
as shared context when making annotation decisions. Prioritize analysis-level metadata
(analysis name and analysis description) over broad study context when they conflict.


{study_context_str}

TABLES AND ANALYSES:
{tables_str}

ANNOTATION NAME MAPPING (FIXED IDENTIFIERS):
{annotation_mapping_str}

IMPORTANT CONSTRAINT (ENUM):
The field "annotation_name" MUST be one of the following exact strings and MUST NOT use placeholder labels:
{allowed_names_str}

GLOBAL CRITERIA (APPLIES TO ALL ANNOTATIONS; HARD GATE)
GLOBAL INCLUSION CRITERIA:
{global_inclusion_text}

GLOBAL EXCLUSION CRITERIA:
{global_exclusion_text}

{criteria_str}

For EACH analysis and EACH annotation, provide:
- Include decision (true/false)
- Reasoning
- Applied inclusion/exclusion criteria

IMPORTANT:
- Use the exact "Analysis ID" shown above for each analysis.
- Do NOT output placeholder annotation labels. Use the canonical annotation names listed in the ENUM.
- Decision policy:
  1) Apply global criteria first.
  2) If any global exclusion applies OR required global inclusion criteria are not met, set include=false.
  3) Only if global criteria pass, evaluate local annotation criteria.
- For include=true: include both global/local inclusion IDs that justify inclusion.
- For include=false: provide exclusion IDs when applicable; if none apply, explicitly name missing inclusion IDs in reasoning.

Output JSON format:
{{
  "study_id": "{study_group.study_id}",
  "decisions": [
    {{
      "analysis_id": "<use exact Analysis ID from above>",
      "annotations": [
        {example_annotations_str}
      ]
    }}
  ]
}}

FINAL SELF-CHECK (before you respond):
- Every annotation_name exactly matches one of the allowed strings in the ENUM.
- No placeholder annotation labels appear anywhere.
- For include=true decisions, inclusion_criteria_applied is not empty.
- For include=false decisions with empty exclusion_criteria_applied, reasoning names missing inclusion criteria IDs.
""".strip()

    return prompt


def create_single_study_annotation_prompt(
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
        metadata_fields: List of metadata fields to include (already filtered by config)
        
    Returns:
        Formatted prompt string
    """
    if metadata_fields is None:
        metadata_fields = []
    
    # Build the metadata section based on configured fields
    metadata_lines = []
    
    if "analysis_name" in metadata_fields and metadata.analysis_name:
        metadata_lines.append(f"- Analysis Name: {metadata.analysis_name}")
        
    if ("analysis_description" in metadata_fields and
            metadata.analysis_description):
        metadata_lines.append(
            f"- Analysis Description: {metadata.analysis_description}"
        )
    
    if "table_caption" in metadata_fields and metadata.table_caption:
        metadata_lines.append(f"- Table Caption: {metadata.table_caption}")
    
    if "table_footer" in metadata_fields and metadata.table_footer:
        metadata_lines.append(f"- Table Footer: {metadata.table_footer}")
        
    if "study_title" in metadata_fields and metadata.study_title:
        metadata_lines.append(f"- Study Title: {metadata.study_title}")
        
    if "study_abstract" in metadata_fields and metadata.study_abstract:
        metadata_lines.append(f"- Study Abstract: {metadata.study_abstract}")
        
    if "study_authors" in metadata_fields and metadata.study_authors:
        authors = ', '.join(metadata.study_authors)
        metadata_lines.append(f"- Study Authors: {authors}")
        
    if "study_journal" in metadata_fields and metadata.study_journal:
        metadata_lines.append(f"- Study Journal: {metadata.study_journal}")
        
    if ("study_publication_date" in metadata_fields and
            metadata.study_publication_date):
        date_str = metadata.study_publication_date
        metadata_lines.append(f"- Study Publication Date: {date_str}")
    
    if "study_fulltext" in metadata_fields and metadata.study_fulltext:
        metadata_lines.append(
            f"- Study Full Text: {metadata.study_fulltext}"
        )
    
    # Add any custom fields
    for field_name, field_value in metadata.custom_fields.items():
        if field_name in metadata_fields and field_value:
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
