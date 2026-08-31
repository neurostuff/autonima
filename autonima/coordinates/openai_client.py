"""LLM client for coordinate parsing tasks."""

import math
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel
from ..llm.client import GenericLLMClient, resolve_model_name, resolve_model_kwargs
from .schema import ParseAnalysesOutput


def _sanitize_parse_result(result_dict: object) -> dict:
    """Drop malformed nested items without discarding an entire table."""
    if not isinstance(result_dict, dict):
        return {"analyses": []}

    sanitized_analyses = []
    raw_analyses = result_dict.get("analyses", [])
    if not isinstance(raw_analyses, list):
        raw_analyses = []

    for analysis in raw_analyses:
        if not isinstance(analysis, dict):
            continue
        sanitized_analysis = dict(analysis)
        valid_points = []
        raw_points = analysis.get("points", [])
        if not isinstance(raw_points, list):
            raw_points = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            coordinates = point.get("coordinates")
            if not (
                isinstance(coordinates, list)
                and len(coordinates) == 3
                and all(
                    isinstance(coord, (int, float))
                    and not isinstance(coord, bool)
                    and math.isfinite(coord)
                    and abs(coord) <= 200
                    for coord in coordinates
                )
            ):
                continue

            sanitized_point = dict(point)
            raw_values = point.get("values")
            if isinstance(raw_values, list):
                valid_values = [
                    value
                    for value in raw_values
                    if isinstance(value, dict)
                ]
                if valid_values:
                    sanitized_point["values"] = valid_values
                else:
                    sanitized_point.pop("values", None)
            elif raw_values is not None:
                sanitized_point.pop("values", None)
            valid_points.append(sanitized_point)

        sanitized_analysis["points"] = valid_points
        sanitized_analyses.append(sanitized_analysis)

    return {"analyses": sanitized_analyses}


class CoordinateParsingClient(GenericLLMClient):
    """LLM client specifically for coordinate parsing tasks."""
    
    def parse_analyses(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        model_params: Optional[Dict[str, Any]] = None
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
            model=resolve_model_name(model),
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
            function_call={"name": func_name},
                **resolve_model_kwargs(model, model_params)
        )
        
        # Extract the function call result
        function_call = response.choices[0].message.function_call
        if not function_call:
            raise ValueError("No function call returned from API")
        
        # Parse the result
        import json
        result_dict = _sanitize_parse_result(json.loads(function_call.arguments))
        
        # Validate and return the result
        try:
            return ParseAnalysesOutput(**result_dict)
        except Exception as e:
            # Log the error and the result that failed validation
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Validation error: {e}")
            logger.error(f"Result that failed validation: {result_dict}")
            # Re-raise the exception
            raise
