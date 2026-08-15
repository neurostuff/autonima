"""Generic LLM API client for systematic review tasks."""

import os
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
import openai


MODEL_PREFIX_ENV = "AUTONIMA_MODEL_PREFIX"


def resolve_model_name(model: Optional[str]) -> Optional[str]:
    """Qualify a bare model name with the gateway prefix from the environment.

    Some gateways (e.g. Portkey) route on a provider-qualified model name such as
    ``@my-provider-slug/gpt-5-mini-2025-08-07``. Keeping that prefix in config files
    hard-codes one deployment into every config and, because the model string is part
    of each stage's cache signature, changing it invalidates otherwise-valid cached
    screening results. So the prefix is applied here, at request time only, and never
    written back into the config object.

    A model name that is already provider-qualified (contains ``/``) is returned
    unchanged, so configs that pin a full name keep working and are never double-prefixed.

    Only the leading provider segment of the environment value is used, so both
    ``@my-provider-slug`` and a full ``@my-provider-slug/some-model`` are accepted. The
    latter is a common way to write it down, and taking the first segment keeps each
    config's own model choice intact instead of overriding it.
    """
    if not model:
        return model
    prefix = os.getenv(MODEL_PREFIX_ENV, "").strip().strip("/").split("/")[0]
    if not prefix or "/" in model:
        return model
    return f"{prefix}/{model}"


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
