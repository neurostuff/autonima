"""LLM client for coordinate parsing tasks."""

import logging
import re
from typing import Optional

from ..llm.client import GenericLLMClient
from .schema import ParseAnalysesOutput


class CoordinateParsingClient(GenericLLMClient):
    """LLM client specifically for coordinate parsing tasks."""

    _ALLOWED_KINDS = {
        "z-statistic",
        "t-statistic",
        "f-statistic",
        "correlation",
        "p-value",
        "beta",
        "other",
    }

    @staticmethod
    def _normalize_kind(kind: Optional[str]) -> Optional[str]:
        """Normalize model-emitted kind labels to schema-allowed values."""
        if kind is None:
            return None

        text = str(kind).strip().lower()
        text = text.replace("_", "-")
        text = re.sub(r"\s+", "-", text)
        text = text.replace("–", "-").replace("—", "-")

        if text in CoordinateParsingClient._ALLOWED_KINDS:
            return text

        # Explicit non-statistics that should be dropped.
        if any(
            marker in text
            for marker in (
                "cluster",
                "size",
                "volume",
                "brodmann",
                "ba",
                "ale",
                "extent",
            )
        ):
            return None

        if text in {"z", "zscore", "z-score"} or "zstat" in text:
            return "z-statistic"
        if text in {"t", "tscore", "t-score"} or "tstat" in text or text == "stat1":
            return "t-statistic"
        if text in {"f", "fscore", "f-score"} or "fstat" in text:
            return "f-statistic"
        if "pvalue" in text or text.startswith("p-") or text == "p":
            return "p-value"
        if "corr" in text or text in {"r", "pearson-r"}:
            return "correlation"
        if "beta" in text:
            return "beta"
        if text == "stat2":
            # Conservative fallback for unlabeled second statistic columns.
            return "other"
        return "other"
    
    def parse_analyses(
        self,
        prompt: str,
        model: str = "gpt-4o-mini"
    ) -> ParseAnalysesOutput:
        """Parse neuroimaging results table into distinct analyses with metadata.
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            
        Returns:
            ParseAnalysesOutput: The parsed analyses
        """
        # Generate function schema from Pydantic model
        func_name = "parse_analyses"
        function_schema = self._generate_function_schema(
            ParseAnalysesOutput,
            func_name
        )

        # Call the LLM API
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that parses neuroimaging results tables "
                        "into structured JSON for downstream analysis. "
                        "Respond using the parse_analyses function."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            functions=[function_schema],
            function_call={"name": func_name}
        )
        
        # Extract the function call result
        function_call = response.choices[0].message.function_call
        if not function_call:
            raise ValueError("No function call returned from API")
        
        # Parse the result
        import json
        result_dict = json.loads(function_call.arguments)
        
        # Preprocess the result to filter out points without valid coordinates
        # and sanitize value kinds to schema-allowed literals.
        for analysis in result_dict.get("analyses", []):
            valid_points = []
            for point in analysis.get("points", []):
                # Check if the point has valid coordinates
                coordinates = point.get("coordinates")
                if (isinstance(coordinates, list) and
                    len(coordinates) == 3 and
                    all(isinstance(coord, (int, float)) for coord in coordinates)):
                    cleaned_values = []
                    for value in point.get("values", []) or []:
                        normalized_kind = self._normalize_kind(value.get("kind"))
                        if normalized_kind is None:
                            # Drop explicitly non-statistical value columns.
                            continue
                        value["kind"] = normalized_kind
                        cleaned_values.append(value)

                    if cleaned_values:
                        point["values"] = cleaned_values
                    else:
                        point.pop("values", None)

                    valid_points.append(point)
            analysis["points"] = valid_points
        
        # Validate and return the result
        try:
            return ParseAnalysesOutput(**result_dict)
        except Exception as e:
            # Log the error and the result that failed validation
            logger = logging.getLogger(__name__)
            logger.error(f"Validation error: {e}")
            logger.error(f"Result that failed validation: {result_dict}")
            # Re-raise the exception
            raise
