"""Coordinate parsing processor for the pipeline."""

import logging
import csv
from pathlib import Path
from typing import List

from .openai_client import CoordinateParsingClient
from .schema import Analysis

logger = logging.getLogger(__name__)


class CoordinateProcessor:
    """Processor for parsing coordinates from activation tables."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the coordinate processor.
        
        Args:
            model: The model to use for parsing
        """
        self.model = model
        self.client = CoordinateParsingClient()
    
    def process_study(self, study):
        """
        Process all activation tables for a study and extract analyses.
        
        Args:
            study: The study to process
            
        Returns:
            List of analyses extracted from the study's tables
        """
        # Import locally to avoid circular imports
        from autonima.models.types import Study, ActivationTable
        
        if not study.activation_tables:
            return []
        
        all_analyses = []
        
        for table in study.activation_tables:
            try:
                # Load the table data
                table_path = Path(table.table_path)
                if not table_path.exists():
                    logger.warning(f"Table file not found: {table_path}")
                    continue
                
                # Read the table as text
                with open(table_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    table_text = "\n".join([",".join(r) for r in rows])
                
                # Create a prompt for the table
                prompt = self._create_table_prompt(
                    table_text,
                    table_caption=table.table_caption or "",
                    table_foot=table.table_foot or ""
                )
                
                # Parse the table
                result = self.client.parse_analyses(prompt, model=self.model)
                
                # Add the analyses to our list
                all_analyses.extend(result.analyses)
                
            except Exception as e:
                logger.warning(f"Error processing table {table.table_path}: {e}")
                continue
        
        return all_analyses
    
    def process_single_table(self, table):
        """
        Process a single activation table and extract analyses.
        
        Args:
            table: The ActivationTable to process
            
        Returns:
            List of analyses extracted from the table
        """
        try:
            # Load the table data
            table_path = Path(table.table_path)
            if not table_path.exists():
                logger.warning(f"Table file not found: {table_path}")
                return []
            
            # Read the table as text
            with open(table_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                table_text = "\n".join([",".join(r) for r in rows])
            
            # Create a prompt for the table
            prompt = self._create_table_prompt(
                table_text,
                table_caption=table.table_caption or "",
                table_foot=table.table_foot or ""
            )
            
            # Parse the table
            result = self.client.parse_analyses(prompt, model=self.model)
            
            return result.analyses
            
        except Exception as e:
            logger.warning(f"Error processing table {table.table_path}: {e}")
            return []
    
    def _create_table_prompt(self, table_text: str, table_caption: str = "", table_foot: str = "") -> str:
        """
        Create a prompt for parsing a table.
        
        Args:
            table_text: The text content of the table
            table_caption: The caption of the table
            table_foot: The footer of the table
            
        Returns:
            The prompt for the LLM
        """
        prompt = f"""
        You are a neuroimaging data curation assistant.

        You will receive a CSV table extracted from a published fMRI/neuroimaging article.
        The table reports statistical activation results, usually organized by *analysis* or *contrast*
        (e.g., "Athletes: motor imagery", "Non-athletes: motor imagery"). Each analysis may contain multiple rows of
        activation foci, with region names, MNI/TAL coordinates, and statistics.

        Table Caption: {table_caption}
        Table Foot: {table_foot}

        Your task is to output JSON strictly matching the schema of the `parse_analyses` function:

        {{
        "analyses": [
            {{
            "name": <string or null>,
            "description": <string or null>,
            "points": [
                {{
                "coordinates": [x, y, z],
                "space": <"MNI" | "TAL" | null>
                "values": [
                    {{
                    "value": <float or string or null>,
                    "kind": <string or null>
                    }},
                    ...
                ]  # Omit this field if no statistical values are available
                }},
                ...
            ]
            }}
        ]
        }}

        ⚠️ CRITICAL RULES for coordinates:
        - Coordinates **must come ONLY from the X, Y, Z columns** (or an equivalent labeled "MNI coordinates").
        - Do NOT use any values from other numeric columns (e.g., Cluster, Volume, Brodmann area, ALE, T, Z).
        - If a row does not contain all three values under X, Y, Z → exclude that row.
        - Coordinates must be exactly three numeric values, extracted in order: [X, Y, Z].

        Other rules:
        1. **Analyses/contrasts**
        - Start a new analysis whenever a distinct label is present (e.g., "Athletes: motor imagery").
        - If no explicit contrasts, treat the whole table as a single analysis.
        - Use only names that explicitly appear in the provided table, caption, or footnotes. Never invent.

        2. **Space**
        - If the table mentions MNI or Talairach, set `"space"` accordingly.
        - If unclear, use `"space": null`.

        3. **Values**
        - If the table has statistical values (e.g., T, Z), include them in `"values"`
        - For the `"kind"` field, you MUST use ONLY these exact values:
          * "z-statistic" for Z-scores
          * "t-statistic" for T-values
          * "f-statistic" for F-values
          * "p-value" for p-values (including FDR-corrected)
          * "beta" for beta coefficients
          * "correlation" for correlation coefficients
          * "other" for any other statistical measures
        - If no statistical columns, omit the `"values"` field entirely
        - Do NOT include values from non-statistical columns (e.g., Cluster, Volume, Brodmann area, ALE).
        - Each value must correspond to the same row as its X, Y, Z coordinates

        4. **Filtering**
        - Ignore all other columns (cluster size, Brodmann area, ALE, etc.).
        - Only extract X, Y, Z → nothing else.

        5. **Null handling**
        - Missing analysis names → `"name": null`.
        - No valid coordinates in an analysis → keep `"points": []`.

        6. **Consistency**
        - Ensure coordinates are always `[float, float, float]`.
        - Do not include fields outside the schema.
        - Do not fabricate analysis names from prompt examples.

        ---
        
        Now apply these rules to the following table:

        {table_text}
        """
        
        return prompt