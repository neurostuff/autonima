"""LLM client for annotation decisions."""

import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
import sys
import os
from ..llm.client import GenericLLMClient
from .schema import AnalysisMetadata, AnnotationCriteriaConfig, AnnotationDecision

logger = logging.getLogger(__name__)


class AnnotationDecisionOutput(BaseModel):
    """Output schema for annotation decision."""
    include: bool
    reasoning: str


class AnnotationClient:
    """LLM client for making annotation decisions."""
    
    def __init__(self):
        """Initialize the annotation client."""
        self._client = GenericLLMClient()
    
    def make_decision(
        self,
        metadata: AnalysisMetadata,
        criteria: AnnotationCriteriaConfig,
        model: str = "gpt-4o-mini"
    ) -> AnnotationDecision:
        """
        Make a decision about whether an analysis should be included in an annotation.
        
        Args:
            metadata: Analysis metadata
            criteria: Annotation criteria configuration
            model: LLM model to use
            
        Returns:
            Annotation decision with inclusion boolean and reasoning
        """
        try:
            # Create the prompt
            from .prompts import create_annotation_prompt
            # Get metadata_fields from the criteria or use default
            metadata_fields = getattr(criteria, 'metadata_fields', None)
            prompt = create_annotation_prompt(metadata, criteria, metadata_fields)
            
            # Get the response from the LLM
            response_text = self.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a neuroimaging meta-analysis expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            try:
                response_data = json.loads(response_text)
                decision_output = AnnotationDecisionOutput(**response_data)
            except Exception as e:
                logger.warning(f"Failed to parse JSON response: {e}. Response: {response}")
                # Try to extract the information manually
                decision_output = self._parse_response_manually(response)
            
            # Create the annotation decision
            decision = AnnotationDecision(
                annotation_name=criteria.name,
                analysis_id=metadata.analysis_id,
                study_id=metadata.study_id,
                include=decision_output.include,
                reasoning=decision_output.reasoning,
                model_used=model
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making annotation decision: {e}")
            # Return a default decision (exclude) in case of error
            return AnnotationDecision(
                annotation_name=criteria.name,
                analysis_id=metadata.analysis_id,
                study_id=metadata.study_id,
                include=False,
                reasoning=f"Error in decision making: {str(e)}",
                model_used=model
            )
    
    def _parse_response_manually(self, response: str) -> AnnotationDecisionOutput:
        """
        Attempt to parse the response manually if JSON parsing fails.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Annotation decision output
        """
        # Default to excluding if we can't parse
        include = False
        reasoning = "Failed to parse response"
        
        # Simple heuristics to extract information
        response_lower = response.lower()
        
        # Look for inclusion indicators
        if '"include": true' in response_lower or '"include":true' in response_lower:
            include = True
        elif '"include": false' in response_lower or '"include":false' in response_lower:
            include = False
            
        # Try to extract reasoning
        if '"reasoning":' in response:
            try:
                # Find the reasoning part
                start = response.find('"reasoning":') + len('"reasoning":')
                if response[start] == '"':
                    start += 1
                    end = response.find('"', start)
                    reasoning = response[start:end]
                else:
                    # Handle non-string reasoning (shouldn't happen with our prompt)
                    end = response.find('}', start)
                    if end == -1:
                        end = len(response)
                    reasoning = response[start:end].strip()
                    # Remove trailing comma if present
                    if reasoning.endswith(','):
                        reasoning = reasoning[:-1]
            except Exception:
                pass
                
        return AnnotationDecisionOutput(include=include, reasoning=reasoning)
    
    def chat_completion(self, messages, model, response_format=None):
        """
        Get a chat completion from the LLM.
        
        Args:
            messages: List of messages for the conversation
            model: Model to use
            temperature: Temperature for generation
            response_format: Response format (e.g., {"type": "json_object"})
            
        Returns:
            String response from the LLM
        """
        try:
            response = self._client.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise