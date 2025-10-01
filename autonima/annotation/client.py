"""LLM client for annotation decisions."""

import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from ..llm.client import OpenAIClient
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
        self._client = OpenAIClient()
    
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
            prompt = create_annotation_prompt(metadata, criteria)
            
            # Get the response from the LLM
            response = self._client.chat_completion(
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
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            try:
                response_data = json.loads(response)
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