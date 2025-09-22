"""LLM client for coordinate parsing tasks."""

from typing import Type
from pydantic import BaseModel
from ..llm.client import GenericLLMClient
from .schema import ParseAnalysesOutput


class CoordinateParsingClient(GenericLLMClient):
    """LLM client specifically for coordinate parsing tasks."""
    
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
        for analysis in result_dict.get("analyses", []):
            valid_points = []
            for point in analysis.get("points", []):
                # Check if the point has valid coordinates
                coordinates = point.get("coordinates")
                if (isinstance(coordinates, list) and
                    len(coordinates) == 3 and
                    all(isinstance(coord, (int, float)) for coord in coordinates)):
                    valid_points.append(point)
            analysis["points"] = valid_points
        
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