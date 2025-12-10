"""LLM client for annotation decisions."""

import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from ..llm.client import GenericLLMClient
from .schema import AnalysisMetadata, AnnotationConfig, AnnotationCriteriaConfig, AnnotationDecision, StudyAnalysisGroup
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


class AnnotationDecisionOutput(BaseModel):
    """Output schema for annotation decision."""
    include: bool = Field(..., description="Boolean decision: true to include, false to exclude")
    reasoning: str = Field(..., description="Brief explanation for the decision")
    inclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of inclusion criterion IDs that apply")
    exclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of exclusion criterion IDs that apply")
    
    @field_validator('include', mode='before')
    @classmethod
    def validate_include(cls, v):
        """Ensure include is a valid boolean, not None."""
        if v is None:
            raise ValueError("'include' field cannot be None - must be true or false")
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in ('true', 'yes', '1'):
                return True
            elif v_lower in ('false', 'no', '0'):
                return False
            raise ValueError(f"Invalid string value for 'include': {v}")
        return bool(v)


class MultiAnnotationDecisionOutput(BaseModel):
    """Output schema for multiple annotation decisions."""
    annotation_name: str = Field(..., description="Name of the annotation")
    include: bool = Field(..., description="Boolean decision: true to include, false to exclude")
    reasoning: str = Field(..., description="Brief explanation for the decision")
    inclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of inclusion criterion IDs that apply")
    exclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of exclusion criterion IDs that apply")
    
    @field_validator('include', mode='before')
    @classmethod
    def validate_include(cls, v):
        """Ensure include is a valid boolean, not None."""
        if v is None:
            raise ValueError("'include' field cannot be None - must be true or false")
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in ('true', 'yes', '1'):
                return True
            elif v_lower in ('false', 'no', '0'):
                return False
            raise ValueError(f"Invalid string value for 'include': {v}")
        return bool(v)


class MultiAnnotationDecisionOutputList(BaseModel):
    """Output schema for list of multiple annotation decisions."""
    decisions: List[MultiAnnotationDecisionOutput]


class AnalysisAnnotations(BaseModel):
    """Annotations for a single analysis in study-level response."""
    annotation_name: str = Field(..., description="Name of the annotation")
    include: bool = Field(..., description="Boolean decision: true to include, false to exclude")
    reasoning: str = Field(..., description="Brief explanation for the decision")
    inclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of inclusion criterion IDs that apply")
    exclusion_criteria_applied: List[str] = Field(default_factory=list, description="List of exclusion criterion IDs that apply")
    
    @field_validator('include', mode='before')
    @classmethod
    def validate_include(cls, v):
        """Ensure include is a valid boolean, not None."""
        if v is None:
            raise ValueError("'include' field cannot be None - must be true or false")
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in ('true', 'yes', '1'):
                return True
            elif v_lower in ('false', 'no', '0'):
                return False
            raise ValueError(f"Invalid string value for 'include': {v}")
        return bool(v)


class StudyAnalysisDecision(BaseModel):
    """Decision for a single analysis in study-level response."""
    analysis_id: str
    annotations: List[AnalysisAnnotations]


class StudyMultiAnnotationOutput(BaseModel):
    """Output schema for study-level multi-annotation decisions."""
    study_id: str
    decisions: List[StudyAnalysisDecision]


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
    
    def make_multi_decision(
        self,
        metadata: AnalysisMetadata,
        criteria_list: List[AnnotationCriteriaConfig],
        model: str = "gpt-4o-mini",
        prompt_type: str = "single_analysis",
        study_group: StudyAnalysisGroup = None
    ) -> List[AnnotationDecision]:
        """
        Make decisions about whether an analysis should be included in multiple annotations.
        
        Args:
            metadata: Analysis metadata
            criteria_list: List of annotation criteria configurations
            model: LLM model to use
            prompt_type: Type of prompt ("single_analysis" or "multi_analysis")
            study_group: Study analysis group (required for multi_analysis)
            
        Returns:
            List of annotation decisions
        """
        if not criteria_list:
            return []
        
        try:
            # Extract metadata fields that are actually present in the metadata object
            # (non-None values, excluding required fields)
            metadata_fields = []
            metadata_dict = metadata.model_dump()
            for field_name, field_value in metadata_dict.items():
                # Skip required fields and None values
                if field_name not in ['analysis_id', 'study_id', 'table_id', 'custom_fields'] and field_value is not None:
                    metadata_fields.append(field_name)
            
            # Select the appropriate prompt based on prompt_type
            if prompt_type == "multi_analysis":
                if study_group is None:
                    raise ValueError(
                        "study_group is required for multi_analysis prompt type"
                    )
                from .prompts import create_study_multi_annotation_prompt
                prompt = create_study_multi_annotation_prompt(
                    study_group, criteria_list, metadata_fields
                )
            else:  # Default to single_analysis
                from .prompts import create_single_study_annotation_prompt
                prompt = create_single_study_annotation_prompt(
                    metadata, criteria_list, metadata_fields
                )
            
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
            
            # Validate that all 'include' fields are present and boolean
            self._validate_include_fields(result_dict, prompt_type)
            
            # Handle different response formats
            decisions = []
            if prompt_type == "multi_analysis":
                # Parse study-level response
                study_output = StudyMultiAnnotationOutput(**result_dict)
                for analysis_decision in study_output.decisions:
                    for annotation_output in analysis_decision.annotations:
                        decision = AnnotationDecision(
                            annotation_name=annotation_output.annotation_name,
                            analysis_id=analysis_decision.analysis_id,
                            study_id=study_output.study_id,
                            include=annotation_output.include,
                            reasoning=annotation_output.reasoning,
                            model_used=model,
                            inclusion_criteria_applied=annotation_output.inclusion_criteria_applied,
                            exclusion_criteria_applied=annotation_output.exclusion_criteria_applied
                        )
                        decisions.append(decision)
            else:
                # Parse table-level response
                decision_list_output = MultiAnnotationDecisionOutputList(
                    **result_dict
                )
                decision_outputs = decision_list_output.decisions

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
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error making multi annotation decisions: {e}")
            logger.error(f"Prompt used: {prompt[:500]}...")
            if 'result_dict' in locals():
                logger.error(f"Response received: {json.dumps(result_dict, indent=2)[:1000]}")
            raise
    
    def _validate_include_fields(self, result_dict: Dict[str, Any], prompt_type: str) -> None:
        """
        Validate that all 'include' fields in the response are present and boolean.
        
        Args:
            result_dict: The parsed response dictionary
            prompt_type: Type of prompt used
            
        Raises:
            ValueError: If any 'include' field is None or invalid
        """
        if prompt_type == "multi_analysis":
            # Validate study-level response
            if 'decisions' in result_dict:
                for i, analysis_decision in enumerate(result_dict['decisions']):
                    if 'annotations' in analysis_decision:
                        for j, annotation in enumerate(analysis_decision['annotations']):
                            if 'include' not in annotation or annotation['include'] is None:
                                raise ValueError(
                                    f"Missing or None 'include' field in decisions[{i}].annotations[{j}]. "
                                    f"The LLM must provide a boolean value (true/false) for every annotation."
                                )
        else:
            # Validate single-analysis response
            if 'decisions' in result_dict:
                for i, decision in enumerate(result_dict['decisions']):
                    if 'include' not in decision or decision['include'] is None:
                        raise ValueError(
                            f"Missing or None 'include' field in decisions[{i}]. "
                            f"The LLM must provide a boolean value (true/false) for every annotation."
                        )
    
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
            log_error_with_debug(logger, f"Error in chat completion: {e}")
            raise
