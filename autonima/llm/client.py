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
            api_key: API key. If not provided, OPENAI_API_KEY is used.
            base_url: Base URL for the API. If not provided, defaults to 
                      OpenAI's API, or OPENAI_API_GATEWAY when set.
        """
        gateway_base_url = os.getenv("OPENAI_API_GATEWAY")
        self.base_url = base_url or gateway_base_url or None
        
        if api_key:
            self.api_key = api_key
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
        """Generate OpenAI function schema from a Pydantic model with nested handling."""
        
        def convert_field(field_info: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively convert Pydantic JSON schema fields to OpenAI function schema."""
            result = {}
            
            # Handle type
            if "type" in field_info:
                result["type"] = field_info["type"]
            
            # Description
            if "description" in field_info:
                result["description"] = field_info["description"]
            
            # Enum
            if "enum" in field_info:
                result["enum"] = field_info["enum"]
            
            # Handle anyOf/oneOf (e.g., Optional[Literal[...]])
            if "anyOf" in field_info:
                # Keep only enum if present
                for option in field_info["anyOf"]:
                    if "enum" in option:
                        result["enum"] = option["enum"]
                        result["type"] = option.get("type", "string")
                        break  # take first enum found
            
            # Arrays
            if "items" in field_info:
                result["items"] = convert_field(field_info["items"])
            if "minItems" in field_info:
                result["minItems"] = field_info["minItems"]
            if "maxItems" in field_info:
                result["maxItems"] = field_info["maxItems"]
            
            # Numeric constraints
            if field_info.get("type") == "number":
                if "minimum" in field_info:
                    result["minimum"] = field_info["minimum"]
                if "maximum" in field_info:
                    result["maximum"] = field_info["maximum"]
            
            # Nested object via $ref
            if "$ref" in field_info:
                result["$ref"] = field_info["$ref"]
            
            return result
        
        # Generate full JSON schema including definitions
        full_schema = model_class.model_json_schema(ref_template="#/$defs/{model}")
        
        properties = {}
        for field_name, field_info in full_schema.get("properties", {}).items():
            properties[field_name] = convert_field(field_info)
        
        required = full_schema.get("required", [])
        
        return {
            "name": function_name,
            "description": f"Process a task using the {function_name} function",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                # include all definitions so nested refs work
                "$defs": full_schema.get("$defs", {})
            }
        }
