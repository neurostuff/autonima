"""Coordinate parsing processor for the pipeline."""

import logging
import threading
from typing import Any, Dict, List, Optional

from .boundary_rules import (
    compute_name_quality_score,
    is_generic_analysis_name,
)
from .openai_client import CoordinateParsingClient
from .table_normalizer import normalize_table

logger = logging.getLogger(__name__)


class CoordinateProcessor:
    """Processor for parsing coordinates from activation tables."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        path_preference: List[str] = None,
        use_canonical_table_json: bool = True,
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
        self.use_canonical_table_json = use_canonical_table_json
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
            if self.use_canonical_table_json:
                try:
                    canonical = normalize_table(table)
                    table.table_normalized_json = canonical
                except Exception as e:
                    logger.warning(
                        "Canonical normalization failed for table %s: %s. Falling back.",
                        table.table_id,
                        e,
                    )
                    table_diag["failure_tags"].append("table_structure_misread")
                else:
                    try:
                        analyses = self._parse_from_canonical(canonical, table, table_diag)
                        self._record_table_diagnostic(table_diag)
                        return analyses
                    except Exception as e:
                        logger.warning(
                            "Canonical parse path failed for table %s: %s. Falling back.",
                            table.table_id,
                            e,
                        )
                        table_diag["failure_tags"].append("coordinate_block_misattribution")

            analyses = self._parse_with_fallback_prompt(table, table_diag)
            self._record_table_diagnostic(table_diag)
            return analyses
            
        except Exception as e:
            logger.warning(f"Error processing table {table.table_id}: {e}")
            table_diag["failure_tags"].append("table_structure_misread")
            self._record_table_diagnostic(table_diag)
            return []

    def _parse_from_canonical(
        self,
        canonical: Dict[str, Any],
        table: Any,
        table_diag: Dict[str, Any],
    ) -> List[Any]:
        source_parser = canonical.get("source_parser", "csv_fallback")
        boundary_markers = canonical.get("boundary_markers", {})
        segments = boundary_markers.get("segments", [])
        rows = canonical.get("rows", [])
        coordinate_row_count = sum(1 for row in rows if row.get("has_coordinates"))

        table_diag["source_parser"] = source_parser
        table_diag["boundary_count"] = int(boundary_markers.get("segment_count", 0) or 0)
        table_diag["structure_confidence"] = self._compute_structure_confidence(source_parser)

        if source_parser in {"pubget_xml_original", "pubget_xml_transformed", "ace_html"}:
            table_diag["failure_tags"] = []

        analyses: List[Any] = []
        parse_segments = segments or [{"segment_index": 0, "boundary_key": "default"}]

        for segment in parse_segments:
            prompt = self._create_canonical_prompt(canonical, segment=segment)
            result = self.client.parse_analyses(prompt, model=self.model)
            segment_context = self._build_segment_context(canonical, segment)
            for analysis in result.analyses:
                analysis.table_id = table.table_id
                analysis_diag = self._apply_name_quality_policy(
                    analysis, segment_context
                )
                analysis_diag["segment_index"] = segment.get("segment_index")
                analysis_diag["boundary_key"] = segment.get("boundary_key")
                table_diag["analysis_diagnostics"].append(analysis_diag)
                analyses.append(analysis)

        table_diag["analysis_count"] = len(analyses)
        generic_count = sum(
            1
            for diag in table_diag["analysis_diagnostics"]
            if diag.get("generic_labeling_detected")
        )
        if generic_count:
            table_diag["failure_tags"].append("generic_labeling")

        if segments and len(analyses) < len(segments):
            table_diag["failure_tags"].append("contrast_overmerge")

        if coordinate_row_count > 0 and len(analyses) == 0:
            table_diag["failure_tags"].append("coordinate_block_misattribution")

        # De-duplicate while preserving order.
        table_diag["failure_tags"] = list(dict.fromkeys(table_diag["failure_tags"]))
        return analyses

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
        if any(d.get("generic_labeling_detected") for d in table_diag["analysis_diagnostics"]):
            table_diag["failure_tags"].append("generic_labeling")
        table_diag["failure_tags"] = list(dict.fromkeys(table_diag["failure_tags"]))
        return analyses

    def _compute_structure_confidence(self, source_parser: str) -> float:
        if source_parser == "pubget_xml_original":
            return 0.95
        if source_parser in {"pubget_xml_transformed", "ace_html"}:
            return 0.85
        return 0.5

    def _build_segment_context(
        self,
        canonical: Dict[str, Any],
        segment: Optional[Dict[str, Any]],
    ) -> str:
        identity = canonical.get("table_identity", {})
        caption = identity.get("table_caption") or ""
        boundary_key = ""
        if segment:
            boundary_key = segment.get("boundary_key") or ""
        context_parts = []
        if boundary_key and boundary_key != "default":
            context_parts.append(boundary_key)
        if caption:
            context_parts.append(caption)
        return " | ".join(context_parts)

    def _apply_name_quality_policy(self, analysis: Any, fallback_context: str) -> Dict[str, Any]:
        name_quality_score = compute_name_quality_score(analysis.name)
        generic_labeling_detected = is_generic_analysis_name(analysis.name)

        if generic_labeling_detected:
            analysis.name = None
            if not analysis.description:
                analysis.description = fallback_context or None

        return {
            "analysis_name": analysis.name,
            "point_count": len(analysis.points),
            "name_quality_score": round(name_quality_score, 3),
            "generic_labeling_detected": generic_labeling_detected,
        }

    def _create_canonical_prompt(
        self,
        canonical: Dict[str, Any],
        segment: Optional[Dict[str, Any]] = None,
    ) -> str:
        identity = canonical.get("table_identity", {})
        headers = canonical.get("header_hierarchy", [])
        rows = canonical.get("rows", [])
        coord_detection = canonical.get("coordinate_column_detection", {})
        boundary_markers = canonical.get("boundary_markers", {})
        space_hints = canonical.get("space_hints", [])

        segment_index = None if segment is None else segment.get("segment_index")
        row_start = 0 if segment is None else segment.get("row_start", 0)
        row_end = len(rows) - 1 if segment is None else segment.get("row_end", len(rows) - 1)

        selected_rows = [
            row
            for row in rows
            if row_start <= row.get("row_index", -1) <= row_end
            and (
                row.get("row_type") == "header"
                or row.get("has_coordinates")
                or row.get("section_label")
            )
        ]
        if not selected_rows:
            selected_rows = rows

        header_lines = []
        for col_idx, path in enumerate(headers):
            header_lines.append(f"- Col {col_idx}: {' > '.join(path) if path else '(empty)'}")
        header_block = "\n".join(header_lines) if header_lines else "- No headers detected"

        row_lines = []
        for row in selected_rows:
            coord_values = row.get("coordinate_values", {})
            stat_values = row.get("stat_values", [])
            stats_text = "; ".join(
                f"{entry.get('kind')}={entry.get('raw')}"
                for entry in stat_values
                if entry.get("raw")
            )
            row_lines.append(
                " | ".join(
                    [
                        f"row={row.get('row_index')}",
                        f"section={row.get('section_label') or '-'}",
                        f"boundary={row.get('boundary_key') or '-'}",
                        f"cells={row.get('cell_text_by_col')}",
                        f"x={coord_values.get('x')}",
                        f"y={coord_values.get('y')}",
                        f"z={coord_values.get('z')}",
                        f"stats={stats_text or '-'}",
                    ]
                )
            )
        rows_block = "\n".join(row_lines) if row_lines else "- No rows available"

        segment_constraints = []
        if segment is not None:
            segment_constraints.append(
                f"Parse ONLY segment {segment_index} with boundary key '{segment.get('boundary_key')}'."
            )
            segment_constraints.append(
                "Do NOT merge analyses across different segment indices or boundary keys."
            )
        else:
            segment_constraints.append(
                "Do NOT merge analyses across rows with different boundary keys."
            )
        constraint_lines = "\n".join(f"- {line}" for line in segment_constraints)

        coordinate_column_text = (
            f"x_col={coord_detection.get('x_col')}, "
            f"y_col={coord_detection.get('y_col')}, "
            f"z_col={coord_detection.get('z_col')}"
        )

        prompt = f"""
        You are a neuroimaging data curation assistant.

        You are given a CANONICAL TABLE JSON projection with expanded table structure
        (rowspan/colspan already resolved) and deterministic boundary markers.
        Use this structured view to extract analyses and coordinates.

        Table ID: {identity.get('table_id')}
        Table Label: {identity.get('table_label')}
        Table Caption: {identity.get('table_caption')}
        Table Foot: {identity.get('table_foot')}
        Space hints: {space_hints}
        Coordinate columns: {coordinate_column_text}
        Boundary segments: {boundary_markers.get('segments', [])}

        Hard constraints:
        - Coordinates must come only from detected x/y/z columns.
        - Exclude rows missing any of x, y, z.
        {constraint_lines}
        - Start a new analysis when contrast/subgroup/direction boundary changes.
        - Names should include discriminative contrast semantics whenever available.
        - Use only table/caption/footer text; do not invent names.
        - If values are provided, "kind" MUST be one of:
          "z-statistic", "t-statistic", "f-statistic", "correlation", "p-value", "beta", "other".
        - Do NOT include non-statistical columns as values (e.g., cluster size, volume, BA, ALE).
        - If kind is unclear, use "other" (never invent labels like "stat1", "cluster_size", etc.).

        Header hierarchy:
        {header_block}

        Structured rows:
        {rows_block}

        Output JSON for parse_analyses schema:
        {{
          "analyses": [
            {{
              "name": <string or null>,
              "description": <string or null>,
              "points": [
                {{
                  "coordinates": [x, y, z],
                  "space": <"MNI" | "TAL" | null>,
                  "values": [{{"value": <float|string|null>, "kind": <allowed kind>}}]
                }}
              ]
            }}
          ]
        }}
        """

        return prompt

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
