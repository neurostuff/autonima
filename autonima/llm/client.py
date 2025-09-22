"""Generic LLM API client for systematic review tasks."""

import os
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
import openai


class GenericLLMClient:
    """Generic LLM API client for various tasks."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the generic LLM client.
        
        Args:
            api_key: API key. If not provided, will use appropriate environment 
                     variable based on the base_url.
            base_url: Base URL for the API. If not provided, defaults to 
                      OpenAI's API.
        """
        self.base_url = base_url or None
        
        # Determine the appropriate API key based on the base URL
        if api_key:
            self.api_key = api_key
        elif self.base_url and "openrouter" in self.base_url:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API key must be provided either as an argument or "
                "through the appropriate environment variable"
            )
        
        # Initialize the client with the base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _generate_function_schema(
        self, 
        model_class: Type[BaseModel], 
        function_name: str
    ) -> Dict[str, Any]:
        """Generate OpenAI function schema from Pydantic model.
        
        Args:
            model_class: Pydantic model class
            function_name: Name for the function
            
        Returns:
            Dict representing the OpenAI function schema
        """
        schema = model_class.model_json_schema()
        
        # Convert JSON schema to OpenAI function schema
        properties = {}
        required = []
        
        for field_name, field_info in schema.get("properties", {}).items():
            properties[field_name] = {
                "type": field_info["type"],
                "description": field_info.get("description", "")
            }
            
            # Handle array items
            if "items" in field_info:
                properties[field_name]["items"] = field_info["items"]
            
            # Handle array constraints
            if "minItems" in field_info:
                properties[field_name]["minItems"] = field_info["minItems"]
            if "maxItems" in field_info:
                properties[field_name]["maxItems"] = field_info["maxItems"]
            
            # Handle enum values
            if "enum" in field_info:
                properties[field_name]["enum"] = field_info["enum"]
            
            # Handle numeric constraints
            if field_info["type"] == "number":
                if "minimum" in field_info:
                    properties[field_name]["minimum"] = field_info["minimum"]
                if "maximum" in field_info:
                    properties[field_name]["maximum"] = field_info["maximum"]
        
        # Get required fields
        required = schema.get("required", [])
        
        return {
            "name": function_name,
            "description": f"Process a task using the {function_name} function",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }