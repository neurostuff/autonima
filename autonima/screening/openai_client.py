"""Generic LLM API client for systematic review screening."""

import os
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
import openai
from .schema import AbstractScreeningOutput, FullTextScreeningOutput


class GenericLLMClient:
    """Generic LLM API client for screening tasks."""
    
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
        elif "openrouter" in self.base_url:
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
        
        # Create description for the function
        screen_type = function_name.replace('screen_', '')
        description = f"Screen a study {screen_type} for inclusion/exclusion"
        
        return {
            "name": function_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def screen_abstract(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> AbstractScreeningOutput:
        """Screen an abstract using LLM API with function calling.
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            AbstractScreeningOutput: The screening result
        """
        # Generate function schema from Pydantic model
        func_name = "screen_abstract"
        function_schema = self._generate_function_schema(
            AbstractScreeningOutput, 
            func_name
        )
        
        # Call the LLM API
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a systematic review screener. "
                        "Respond using the screen_abstract function."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            functions=[function_schema],
            function_call={"name": func_name},
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract the function call result
        function_call = response.choices[0].message.function_call
        if not function_call:
            raise ValueError("No function call returned from API")
        
        # Parse the result
        import json
        result_dict = json.loads(function_call.arguments)
        return AbstractScreeningOutput(**result_dict)
    
    def screen_fulltext(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> FullTextScreeningOutput:
        """Screen a full-text using LLM API with function calling.
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            FullTextScreeningOutput: The screening result
        """
        # Generate function schema from Pydantic model
        func_name = "screen_fulltext"
        function_schema = self._generate_function_schema(
            FullTextScreeningOutput, 
            func_name
        )
        
        # Call the LLM API
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a systematic review screener. "
                        "Respond using the screen_fulltext function."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            functions=[function_schema],
            function_call={"name": func_name},
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract the function call result
        function_call = response.choices[0].message.function_call
        if not function_call:
            raise ValueError("No function call returned from API")
        
        # Parse the result
        import json
        result_dict = json.loads(function_call.arguments)
        return FullTextScreeningOutput(**result_dict)