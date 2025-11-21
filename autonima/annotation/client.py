"""LLM client for annotation decisions."""

import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel
from ..llm.client import GenericLLMClient
from .schema import AnalysisMetadata, AnnotationCriteriaConfig, AnnotationDecision

logger = logging.getLogger(__name__)


class AnnotationDecisionOutput(BaseModel):
    """Output schema for annotation decision."""
    include: bool
    reasoning: str
    inclusion_criteria_applied: List[str] = []
    exclusion_criteria_applied: List[str] = []


class MultiAnnotationDecisionOutput(BaseModel):
    """Output schema for multiple annotation decisions."""
    annotation_name: str
    include: bool
    reasoning: str
    inclusion_criteria_applied: List[str] = []
    exclusion_criteria_applied: List[str] = []


class MultiAnnotationDecisionOutputList(BaseModel):
    """Output schema for list of multiple annotation decisions."""
    decisions: List[MultiAnnotationDecisionOutput]


class AnnotationClient:
    """LLM client for making annotation decisions."""
    
    def __init__(self):
        """Initialize the annotation client."""
        self._client = GenericLLMClient()
    
    def _generate_function_schema(
        self,
        model_class: BaseModel,
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
            
            # Handle array items
            if field_info["type"] == "array" and "items" in field_info:
                properties[field_name]["items"] = field_info["items"]
            
            # Handle numeric constraints
            if field_info["type"] == "number":
                if "minimum" in field_info:
                    properties[field_name]["minimum"] = field_info["minimum"]
                if "maximum" in field_info:
                    properties[field_name]["maximum"] = field_info["maximum"]
        
        # Get required fields
        required = schema.get("required", [])
        
        # Create description for the function
        description = "Make an annotation decision for a neuroimaging analysis"
        
        return {
            "name": function_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
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
            
            # Generate function schema from Pydantic model
            func_name = "make_annotation_decision"
            function_schema = self._generate_function_schema(
                AnnotationDecisionOutput,
                func_name
            )
            
            # Call the LLM API with function calling
            response = self._client.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a neuroimaging meta-analysis expert. "
                            "Respond using the make_annotation_decision function."
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
            result_dict = json.loads(function_call.arguments)
            decision_output = AnnotationDecisionOutput(**result_dict)
            
            # Create the annotation decision
            decision = AnnotationDecision(
                annotation_name=criteria.name,
                analysis_id=metadata.analysis_id,
                study_id=metadata.study_id,
                include=decision_output.include,
                reasoning=decision_output.reasoning,
                model_used=model,
                inclusion_criteria_applied=decision_output.inclusion_criteria_applied,
                exclusion_criteria_applied=decision_output.exclusion_criteria_applied
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
    
    def make_multi_decision(
        self,
        metadata: AnalysisMetadata,
        criteria_list: List[AnnotationCriteriaConfig],
        model: str = "gpt-4o-mini"
    ) -> List[AnnotationDecision]:
        """
        Make decisions about whether an analysis should be included in multiple annotations.
        
        Args:
            metadata: Analysis metadata
            criteria_list: List of annotation criteria configurations
            model: LLM model to use
            
        Returns:
            List of annotation decisions
        """
        if not criteria_list:
            return []
        
        try:
            # Create the prompt for all annotations at once
            from .prompts import create_multi_annotation_prompt
            # Get metadata_fields from the first criteria or use default
            metadata_fields = getattr(criteria_list[0], 'metadata_fields', None)
            prompt = create_multi_annotation_prompt(metadata, criteria_list, metadata_fields)
            
            # Generate function schema from Pydantic model
            func_name = "make_multi_annotation_decisions"
            function_schema = self._generate_function_schema(
                MultiAnnotationDecisionOutputList,
                func_name
            )
            
            # Call the LLM API with function calling
            response = self._client.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a neuroimaging meta-analysis expert. "
                            "Respond using the make_multi_annotation_decisions function."
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
            result_dict = json.loads(function_call.arguments)
            decision_list_output = MultiAnnotationDecisionOutputList(**result_dict)
            decision_outputs = decision_list_output.decisions

            # Create the annotation decisions
            decisions = []
            for i, decision_output in enumerate(decision_outputs):
                if i < len(criteria_list):
                    criteria = criteria_list[i]
                    decision = AnnotationDecision(
                        annotation_name=decision_output.annotation_name or criteria.name,
                        analysis_id=metadata.analysis_id,
                        study_id=metadata.study_id,
                        include=decision_output.include,
                        reasoning=decision_output.reasoning,
                        model_used=model,
                        inclusion_criteria_applied=decision_output.inclusion_criteria_applied,
                        exclusion_criteria_applied=decision_output.exclusion_criteria_applied
                    )
                    decisions.append(decision)
            
            # If we didn't get enough responses, fill in with individual decisions
            while len(decisions) < len(criteria_list):
                criteria = criteria_list[len(decisions)]
                decision = self.make_decision(metadata, criteria, model)
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error making multi annotation decisions: {e}")
            # Return individual decisions as fallback
            return [self.make_decision(metadata, criteria, model) for criteria in criteria_list]
    
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