"""LLM client for annotation decisions."""

import json
import logging
from types import SimpleNamespace
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from ..llm.client import GenericLLMClient
from .schema import AnalysisMetadata, AnnotationCriteriaConfig, AnnotationDecision, StudyAnalysisGroup, build_dynamic_multi_annotation_models
from ..utils import log_error_with_debug

logger = logging.getLogger(__name__)


class _MissingAPIKeyCompletions:
    """Placeholder completions API used when no API key is configured."""

    def __init__(self, error: Exception):
        self._error = error

    def create(self, *args, **kwargs):
        raise self._error


class _MissingAPIKeyClient:
    """Lightweight client shape compatible with test patching."""

    def __init__(self, error: Exception):
        self.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=_MissingAPIKeyCompletions(error)
            )
        )


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
    
    def __init__(self, max_retries: int = 3):
        """Initialize the annotation client.
        
        Args:
            max_retries: Maximum number of retries for malformed responses
        """
        try:
            self._client = GenericLLMClient()
        except ValueError as e:
            if "API key must be provided" in str(e):
                # Keep constructor usable for tests that patch
                # `client._client.client.chat.completions.create`.
                self._client = _MissingAPIKeyClient(e)
            else:
                raise
        self.max_retries = max_retries
    
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
        
        # Create description for the function
        description = "Make an annotation decision for a neuroimaging analysis"
        
        return {
            "name": function_name,
            "description": description,
            "parameters": schema,
        }
    
    def make_decision(
        self,
        metadata: Union[AnalysisMetadata, StudyAnalysisGroup],
        criteria_list: List[AnnotationCriteriaConfig],
        metadata_fields: List[str],
        model: str = "gpt-4o-mini",
        prompt_type: str = "single_analysis",
    ) -> List[AnnotationDecision]:
        """
        Make decisions about whether an analysis should be included in multiple annotations.
        
        Args:
            metadata: Either AnalysisMetadata (for single_analysis) or
                     StudyAnalysisGroup (for multi_analysis)
            criteria_list: List of annotation criteria configurations
            model: LLM model to use
            prompt_type: Type of prompt ("single_analysis" or "multi_analysis")
            metadata_fields: List of metadata field names to include in prompt.
            
        Returns:
            List of annotation decisions
        """
        if not criteria_list:
            return []
        
        # Retry loop to handle malformed LLM responses
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return self._make_decision_attempt(
                    metadata, criteria_list, metadata_fields, model, prompt_type
                )
            except (ValueError, KeyError, TypeError) as e:
                last_exception = e
                error_msg = str(e)
                
                # Check if it's a validation error we should retry
                if any(keyword in error_msg.lower() for keyword in [
                    'validation error', 'field required', 'missing', 'none',
                    'hallucinated', 'invalid analysis_id'
                ]):
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed with validation error: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying LLM request...")
                        continue
                else:
                    # Don't retry for other types of errors
                    raise
        
        # All retries exhausted
        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def _make_decision_attempt(
        self,
        metadata: Union[AnalysisMetadata, StudyAnalysisGroup],
        criteria_list: List[AnnotationCriteriaConfig],
        metadata_fields: List[str],
        model: str,
        prompt_type: str,
    ) -> List[AnnotationDecision]:
        """
        Single attempt at making annotation decisions.
        
        Args:
            metadata: Either AnalysisMetadata (for single_analysis) or
                     StudyAnalysisGroup (for multi_analysis)
            criteria_list: List of annotation criteria configurations
            metadata_fields: List of metadata field names to include in prompt
            model: LLM model to use
            prompt_type: Type of prompt ("single_analysis" or "multi_analysis")
            
        Returns:
            List of annotation decisions
            
        Raises:
            ValueError: If response validation fails
        """
        try:
            # Extract valid analysis_ids from metadata for validation
            valid_analysis_ids = self._get_valid_analysis_ids(metadata)
            
            # Select the appropriate prompt based on metadata type
            if isinstance(metadata, StudyAnalysisGroup):
                from .prompts import create_study_multi_annotation_prompt
                prompt = create_study_multi_annotation_prompt(
                    metadata, criteria_list, metadata_fields
                )
            else:
                from .prompts import create_single_study_annotation_prompt
                prompt = create_single_study_annotation_prompt(
                    metadata, criteria_list, metadata_fields
                )
            
            # Generate function schema from dynamic Pydantic models
            DecisionModel, OutputListModel = build_dynamic_multi_annotation_models(criteria_list)
            FunctionModel = OutputListModel
            if prompt_type == "multi_analysis":
                allowed_annotation_names = [c.name for c in criteria_list]

                class StudyAnalysisDecisionDynamic(BaseModel):
                    analysis_id: str
                    annotations: List[DecisionModel]

                    @model_validator(mode="after")
                    def validate_annotation_coverage(self):
                        names = [a.annotation_name for a in self.annotations]
                        if len(set(names)) != len(names):
                            raise ValueError(
                                f"Duplicate annotation_name values found in analysis {self.analysis_id}: {names}"
                            )
                        missing = [
                            name for name in allowed_annotation_names
                            if name not in set(names)
                        ]
                        if missing:
                            raise ValueError(
                                f"Missing annotation decisions for analysis {self.analysis_id}: {missing}"
                            )
                        return self

                class StudyMultiAnnotationOutputDynamic(BaseModel):
                    study_id: str
                    decisions: List[StudyAnalysisDecisionDynamic]

                FunctionModel = StudyMultiAnnotationOutputDynamic

            func_name = "make_multi_annotation_decisions"
            function_schema = self._generate_function_schema(
                FunctionModel,
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
            
            message = response.choices[0].message
            function_call = getattr(message, "function_call", None)
            function_args = (
                getattr(function_call, "arguments", None)
                if function_call is not None
                else None
            )

            raw_payload = None
            if isinstance(function_args, (str, bytes, bytearray)):
                raw_payload = function_args
            else:
                message_content = getattr(message, "content", None)
                if isinstance(message_content, (str, bytes, bytearray)):
                    raw_payload = message_content

            if raw_payload is None:
                raise ValueError(
                    "No parseable function_call arguments or message content "
                    "returned from API"
                )

            # Parse the result
            result_dict = json.loads(raw_payload)

            # Backward-compatible single-decision response format.
            if (
                prompt_type != "multi_analysis"
                and "decisions" not in result_dict
                and "include" in result_dict
            ):
                if len(criteria_list) != 1:
                    raise ValueError(
                        "Single-object response format is only supported "
                        "when exactly one annotation criterion is requested"
                    )
                result_dict = {
                    "decisions": [
                        {
                            "annotation_name": criteria_list[0].name,
                            "include": result_dict.get("include"),
                            "reasoning": result_dict.get("reasoning", ""),
                            "inclusion_criteria_applied": result_dict.get(
                                "inclusion_criteria_applied", []
                            ),
                            "exclusion_criteria_applied": result_dict.get(
                                "exclusion_criteria_applied", []
                            ),
                        }
                    ]
                }
            
            # Validate the response structure before parsing
            self._validate_response_structure(result_dict, prompt_type, metadata)
            
            # Validate that all 'include' fields are present and boolean
            self._validate_include_fields(result_dict, prompt_type)
            
            # Validate that analysis_ids in response match input metadata (prevents hallucination)
            self._validate_analysis_ids(result_dict, valid_analysis_ids, prompt_type)
            
            # Handle different response formats
            decisions = []
            if prompt_type == "multi_analysis":
                # Inject study_id from metadata if not in response
                if 'study_id' not in result_dict and isinstance(metadata, StudyAnalysisGroup):
                    result_dict['study_id'] = metadata.study_id
                
                # Parse study-level response with strict dynamic validation.
                study_output = FunctionModel(**result_dict)
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
                # Parse single-analysis response with strict dynamic validation.
                decision_list_output = OutputListModel(**result_dict)
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
            logger.debug(f"Error in decision attempt: {e}")
            logger.debug(f"Prompt used: {prompt[:500]}...")
            if 'result_dict' in locals():
                logger.debug(f"Response received: {json.dumps(result_dict, indent=2)[:1000]}")
            raise
    
    def _validate_response_structure(
        self,
        result_dict: Dict[str, Any],
        prompt_type: str,
        metadata: Union[AnalysisMetadata, StudyAnalysisGroup]
    ) -> None:
        """
        Validate the structure of the LLM response before parsing.
        
        Args:
            result_dict: The parsed response dictionary
            prompt_type: Type of prompt used
            metadata: Metadata object to get expected structure
            
        Raises:
            ValueError: If response structure is invalid
        """
        if prompt_type == "multi_analysis":
            # Validate study-level response structure
            if 'decisions' not in result_dict:
                raise ValueError(
                    "Missing 'decisions' field in study-level response. "
                    f"Response keys: {list(result_dict.keys())}"
                )
            
            if not isinstance(result_dict['decisions'], list):
                raise ValueError(
                    f"'decisions' field must be a list, got {type(result_dict['decisions'])}"
                )
            
            # Check each decision has required fields
            for i, decision in enumerate(result_dict['decisions']):
                if not isinstance(decision, dict):
                    raise ValueError(
                        f"decisions[{i}] must be a dict, got {type(decision)}"
                    )
                
                # Check for analysis_id field
                if 'analysis_id' not in decision:
                    raise ValueError(
                        f"Missing 'analysis_id' field in decisions[{i}]. "
                        f"Available keys: {list(decision.keys())}"
                    )
                
                # Check for annotations field
                if 'annotations' not in decision:
                    raise ValueError(
                        f"Missing 'annotations' field in decisions[{i}]. "
                        f"Available keys: {list(decision.keys())}"
                    )
                
                if not isinstance(decision['annotations'], list):
                    raise ValueError(
                        f"decisions[{i}].annotations must be a list, "
                        f"got {type(decision['annotations'])}"
                    )
        else:
            # Validate single-analysis response structure
            if 'decisions' not in result_dict:
                raise ValueError(
                    "Missing 'decisions' field in response. "
                    f"Response keys: {list(result_dict.keys())}"
                )
            
            if not isinstance(result_dict['decisions'], list):
                raise ValueError(
                    f"'decisions' field must be a list, got {type(result_dict['decisions'])}"
                )
    
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
    
    def _get_valid_analysis_ids(
        self,
        metadata: Union[AnalysisMetadata, StudyAnalysisGroup]
    ) -> set:
        """
        Extract the set of valid analysis_ids from input metadata.
        
        Args:
            metadata: Either AnalysisMetadata (single) or StudyAnalysisGroup (multiple)
            
        Returns:
            Set of valid analysis_id strings
        """
        if isinstance(metadata, StudyAnalysisGroup):
            return {analysis.analysis_id for analysis in metadata.analyses}
        else:
            # Single analysis metadata
            return {metadata.analysis_id}

    def _validate_analysis_ids(
        self,
        result_dict: Dict[str, Any],
        valid_analysis_ids: set,
        prompt_type: str
    ) -> None:
        """
        Validate that all analysis_ids in the response exist in input metadata.
        
        This prevents the LLM from hallucinating analysis_ids that don't correspond
        to actual analyses in the study.
        
        Args:
            result_dict: The parsed response dictionary
            valid_analysis_ids: Set of valid analysis_ids from input metadata
            prompt_type: Type of prompt used ("single_analysis" or "multi_analysis")
            
        Raises:
            ValueError: If any analysis_id is hallucinated (not in valid set)
        """
        if prompt_type == "multi_analysis":
            # Extract all analysis_ids from the response
            returned_ids = set()
            if 'decisions' in result_dict:
                for decision in result_dict['decisions']:
                    if isinstance(decision, dict) and 'analysis_id' in decision:
                        # Strip whitespace to handle minor formatting issues
                        analysis_id = str(decision['analysis_id']).strip()
                        returned_ids.add(analysis_id)
            
            # Check for hallucinated IDs (in response but not in valid set)
            hallucinated = returned_ids - valid_analysis_ids
            
            if hallucinated:
                logger.warning(
                    f"LLM hallucinated analysis_id(s): {sorted(hallucinated)}. "
                    f"Valid IDs: {sorted(valid_analysis_ids)}. "
                    f"Returned IDs: {sorted(returned_ids)}"
                )
                raise ValueError(
                    f"LLM hallucinated analysis_id(s) that don't exist in input metadata: {sorted(hallucinated)}. "
                    f"Valid IDs were: {sorted(valid_analysis_ids)}. "
                    f"LLM returned: {sorted(returned_ids)}"
                )
            
            logger.debug(
                f"Analysis ID validation passed. All {len(returned_ids)} returned IDs are valid."
            )
        else:
            # For single_analysis mode, validation is less critical since we inject the ID
            # But we can still check if present in the response
            if 'decisions' in result_dict:
                for decision in result_dict['decisions']:
                    if isinstance(decision, dict):
                        # In single_analysis mode, the analysis_id is typically not in the decision
                        # It gets injected later, so we don't need to validate here
                        pass

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
