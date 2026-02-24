"""Coordinate parsing processor for the pipeline."""

import logging
import threading
from typing import Any, Dict, List

from .boundary_rules import (
    compute_name_quality_score,
    is_generic_analysis_name,
)
from .openai_client import CoordinateParsingClient

logger = logging.getLogger(__name__)


class CoordinateProcessor:
    """Processor for parsing coordinates from activation tables."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        path_preference: List[str] = None,
    ):
        """
        Initialize the coordinate processor.
        
        Args:
            model: The model to use for parsing
        """
        if path_preference is None:
            path_preference = [
                "table_html_path",
                "table_raw_path",
                "table_data_path",
                "table_xml_path",
            ]
        self.model = model
        self.path_preference = path_preference
        self.client = CoordinateParsingClient()
        self._diagnostic_lock = threading.Lock()
        self._table_diagnostics: List[Dict[str, Any]] = []

    def get_table_diagnostics(self) -> List[Dict[str, Any]]:
        """Return recorded per-table diagnostics for coordinate parsing."""
        with self._diagnostic_lock:
            return list(self._table_diagnostics)

    def _record_table_diagnostic(self, diagnostic: Dict[str, Any]) -> None:
        with self._diagnostic_lock:
            self._table_diagnostics.append(diagnostic)

    
    def process_single_table(self, table):
        """
        Process a single activation table and extract analyses.
        
        Args:
            table: The ActivationTable to process
            
        Returns:
            List of analyses extracted from the table
        """
        table_diag: Dict[str, Any] = {
            "table_id": table.table_id,
            "table_source_type": getattr(table, "table_source_type", None),
            "source_parser": "csv_fallback",
            "structure_confidence": 0.5,
            "boundary_count": 1,
            "analysis_count": 0,
            "failure_tags": [],
            "analysis_diagnostics": [],
        }

        try:
            analyses = self._parse_with_fallback_prompt(table, table_diag)
            self._record_table_diagnostic(table_diag)
            return analyses
            
        except Exception as e:
            logger.warning(f"Error processing table {table.table_id}: {e}")
            table_diag["failure_tags"].append("table_structure_misread")
            self._record_table_diagnostic(table_diag)
            return []

    def _parse_with_fallback_prompt(
        self,
        table: Any,
        table_diag: Dict[str, Any],
    ) -> List[Any]:
        # Load the raw table content using the table's method.
        table.load_raw_table()
        if table.raw_table is None:
            logger.warning(f"No valid table path found for table: {table.table_id}")
            table_diag["failure_tags"].append("table_structure_misread")
            return []

        prompt = self._create_table_prompt(
            table.raw_table,
            table_caption=table.table_caption or "",
            table_foot=table.table_foot or "",
        )
        result = self.client.parse_analyses(prompt, model=self.model)
        analyses: List[Any] = []
        for analysis in result.analyses:
            analysis.table_id = table.table_id
            analysis_diag = self._apply_name_quality_policy(analysis, table.table_caption or "")
            table_diag["analysis_diagnostics"].append(analysis_diag)
            analyses.append(analysis)

        table_diag["analysis_count"] = len(analyses)
        if any(
            isinstance(d, dict) and d.get("generic_labeling_detected")
            for d in table_diag["analysis_diagnostics"]
        ):
            table_diag["failure_tags"].append("generic_labeling")
        table_diag["failure_tags"] = list(dict.fromkeys(table_diag["failure_tags"]))
        return analyses

    def _apply_name_quality_policy(self, analysis: Any, fallback_context: str) -> Dict[str, Any]:
        name_quality_score = compute_name_quality_score(analysis.name)
        generic_labeling_detected = is_generic_analysis_name(analysis.name)

        if generic_labeling_detected and not analysis.description:
            analysis.description = fallback_context or None

        return {
            "analysis_name": analysis.name,
            "point_count": len(analysis.points),
            "name_quality_score": round(name_quality_score, 3),
            "generic_labeling_detected": generic_labeling_detected,
        }
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
