"""Test retry mechanism for annotation client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic_core._pydantic_core import ValidationError

from autonima.annotation.client import AnnotationClient
from autonima.annotation.schema import AnnotationCriteriaConfig, StudyAnalysisGroup, AnalysisMetadata, TableMetadata


def test_retry_on_malformed_response():
    """Test that the client retries on malformed LLM responses."""
    client = AnnotationClient(max_retries=3)
    
    # Create test metadata
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[
            TableMetadata(table_id="table_1", caption="Test table")
        ],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1",
                analysis_name="Test analysis"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Has activation coordinates"],
            exclusion_criteria=[]
        )
    ]
    
    # Mock the LLM client to return malformed responses first, then a valid one
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # First attempt: malformed response (missing analysis_id and annotations)
        malformed_response = Mock()
        malformed_response.choices = [Mock()]
        malformed_response.choices[0].message.function_call = Mock()
        malformed_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "study_id": "12345",
                    "decisions": []
                }
            ]
        }
        '''
        
        # Second attempt: valid response
        valid_response = Mock()
        valid_response.choices = [Mock()]
        valid_response.choices[0].message.function_call = Mock()
        valid_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "analysis_id": "12345_analysis_0",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Has coordinates",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                }
            ]
        }
        '''
        
        # Set up the mock to return malformed first, then valid
        mock_create.side_effect = [malformed_response, valid_response]
        
        # Make the decision - should succeed on second attempt
        decisions = client.make_decision(
            metadata=metadata,
            criteria_list=criteria_list,
            metadata_fields=["analysis_name"],
            model="gpt-4o-mini",
            prompt_type="multi_analysis"
        )
        
        # Verify the request was called twice (1 failed, 1 succeeded)
        assert mock_create.call_count == 2
        
        # Verify we got the expected decision
        assert len(decisions) == 1
        assert decisions[0].analysis_id == "12345_analysis_0"
        assert decisions[0].include is True
        assert decisions[0].annotation_name == "test_annotation"


def test_retry_exhaustion():
    """Test that the client raises an error after exhausting retries."""
    client = AnnotationClient(max_retries=2)
    
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Test"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # Always return malformed response
        malformed_response = Mock()
        malformed_response.choices = [Mock()]
        malformed_response.choices[0].message.function_call = Mock()
        malformed_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "study_id": "12345",
                    "decisions": []
                }
            ]
        }
        '''
        
        mock_create.return_value = malformed_response
        
        # Should raise ValueError after exhausting retries
        with pytest.raises(ValueError, match="Missing 'analysis_id' field"):
            client.make_decision(
                metadata=metadata,
                criteria_list=criteria_list,
                metadata_fields=["analysis_name"],
                model="gpt-4o-mini",
                prompt_type="multi_analysis"
            )
        
        # Verify the request was called max_retries times
        assert mock_create.call_count == 2


def test_no_retry_on_non_validation_error():
    """Test that non-validation errors are not retried."""
    client = AnnotationClient(max_retries=3)
    
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Test"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # Simulate a network error or other non-validation error
        mock_create.side_effect = RuntimeError("Network error")
        
        # Should raise the error immediately without retrying
        with pytest.raises(RuntimeError, match="Network error"):
            client.make_decision(
                metadata=metadata,
                criteria_list=criteria_list,
                metadata_fields=["analysis_name"],
                model="gpt-4o-mini",
                prompt_type="multi_analysis"
            )
        
        # Verify the request was called only once
        assert mock_create.call_count == 1


def test_retry_on_hallucinated_analysis_id():
    """Test that the client retries when LLM hallucinates analysis_ids."""
    client = AnnotationClient(max_retries=3)
    
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[
            TableMetadata(table_id="table_1", caption="Test table")
        ],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1",
                analysis_name="Real analysis"
            ),
            AnalysisMetadata(
                analysis_id="12345_analysis_1",
                study_id="12345",
                table_id="table_1",
                analysis_name="Another real analysis"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Has activation coordinates"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # First attempt: hallucinated analysis_id
        hallucinated_response = Mock()
        hallucinated_response.choices = [Mock()]
        hallucinated_response.choices[0].message.function_call = Mock()
        hallucinated_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "analysis_id": "12345_analysis_0",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Has coordinates",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                },
                {
                    "analysis_id": "HALLUCINATED_ID_999",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": false,
                            "reasoning": "Made up analysis",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                }
            ]
        }
        '''
        
        # Second attempt: valid response with correct IDs
        valid_response = Mock()
        valid_response.choices = [Mock()]
        valid_response.choices[0].message.function_call = Mock()
        valid_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "analysis_id": "12345_analysis_0",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Has coordinates",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                },
                {
                    "analysis_id": "12345_analysis_1",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Also has coordinates",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                }
            ]
        }
        '''
        
        mock_create.side_effect = [hallucinated_response, valid_response]
        
        # Make the decision - should succeed on second attempt
        decisions = client.make_decision(
            metadata=metadata,
            criteria_list=criteria_list,
            metadata_fields=["analysis_name"],
            model="gpt-4o-mini",
            prompt_type="multi_analysis"
        )
        
        # Verify the request was called twice (1 failed, 1 succeeded)
        assert mock_create.call_count == 2
        
        # Verify we got the expected decisions
        assert len(decisions) == 2
        assert decisions[0].analysis_id == "12345_analysis_0"
        assert decisions[1].analysis_id == "12345_analysis_1"
        assert all(d.include is True for d in decisions)


def test_hallucinated_id_exhausts_retries():
    """Test that persistent hallucination exhausts retries and raises error."""
    client = AnnotationClient(max_retries=2)
    
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Test"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # Always return hallucinated analysis_id
        hallucinated_response = Mock()
        hallucinated_response.choices = [Mock()]
        hallucinated_response.choices[0].message.function_call = Mock()
        hallucinated_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "analysis_id": "WRONG_ID",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Hallucinated",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                }
            ]
        }
        '''
        
        mock_create.return_value = hallucinated_response
        
        # Should raise ValueError after exhausting retries
        with pytest.raises(ValueError, match="hallucinated analysis_id"):
            client.make_decision(
                metadata=metadata,
                criteria_list=criteria_list,
                metadata_fields=["analysis_name"],
                model="gpt-4o-mini",
                prompt_type="multi_analysis"
            )
        
        # Verify the request was called max_retries times
        assert mock_create.call_count == 2


def test_multiple_hallucinated_ids():
    """Test detection of multiple hallucinated IDs in a single response."""
    client = AnnotationClient(max_retries=2)
    
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Test"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # Return multiple hallucinated IDs
        hallucinated_response = Mock()
        hallucinated_response.choices = [Mock()]
        hallucinated_response.choices[0].message.function_call = Mock()
        hallucinated_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "analysis_id": "FAKE_1",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Hallucinated",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                },
                {
                    "analysis_id": "FAKE_2",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": false,
                            "reasoning": "Also hallucinated",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                }
            ]
        }
        '''
        
        mock_create.return_value = hallucinated_response
        
        # Should raise ValueError mentioning both hallucinated IDs
        with pytest.raises(ValueError) as exc_info:
            client.make_decision(
                metadata=metadata,
                criteria_list=criteria_list,
                metadata_fields=["analysis_name"],
                model="gpt-4o-mini",
                prompt_type="multi_analysis"
            )
        
        error_msg = str(exc_info.value)
        assert "FAKE_1" in error_msg or "FAKE_2" in error_msg
        assert "hallucinated" in error_msg.lower()


def test_valid_analysis_ids_pass_validation():
    """Test that valid analysis_ids pass validation without retry."""
    client = AnnotationClient(max_retries=3)
    
    metadata = StudyAnalysisGroup(
        study_id="12345",
        tables=[],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table_1"
            ),
            AnalysisMetadata(
                analysis_id="12345_analysis_1",
                study_id="12345",
                table_id="table_1"
            )
        ]
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Test"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # Valid response with correct IDs
        valid_response = Mock()
        valid_response.choices = [Mock()]
        valid_response.choices[0].message.function_call = Mock()
        valid_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "analysis_id": "12345_analysis_0",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": true,
                            "reasoning": "Valid",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                },
                {
                    "analysis_id": "12345_analysis_1",
                    "annotations": [
                        {
                            "annotation_name": "test_annotation",
                            "include": false,
                            "reasoning": "Also valid",
                            "inclusion_criteria_applied": [],
                            "exclusion_criteria_applied": []
                        }
                    ]
                }
            ]
        }
        '''
        
        mock_create.return_value = valid_response
        
        # Should succeed on first attempt
        decisions = client.make_decision(
            metadata=metadata,
            criteria_list=criteria_list,
            metadata_fields=["analysis_name"],
            model="gpt-4o-mini",
            prompt_type="multi_analysis"
        )
        
        # Verify the request was called only once (no retries)
        assert mock_create.call_count == 1
        
        # Verify decisions were created correctly
        assert len(decisions) == 2
        assert decisions[0].analysis_id == "12345_analysis_0"
        assert decisions[1].analysis_id == "12345_analysis_1"


def test_single_analysis_mode_skips_validation():
    """Test that single_analysis mode doesn't strictly validate analysis_ids."""
    client = AnnotationClient(max_retries=3)
    
    metadata = AnalysisMetadata(
        analysis_id="12345_analysis_0",
        study_id="12345",
        table_id="table_1",
        analysis_name="Test analysis"
    )
    
    criteria_list = [
        AnnotationCriteriaConfig(
            name="test_annotation",
            inclusion_criteria=["Test"],
            exclusion_criteria=[]
        )
    ]
    
    with patch.object(client._client.client.chat.completions, 'create') as mock_create:
        # In single_analysis mode, the response doesn't include analysis_id
        valid_response = Mock()
        valid_response.choices = [Mock()]
        valid_response.choices[0].message.function_call = Mock()
        valid_response.choices[0].message.function_call.arguments = '''
        {
            "decisions": [
                {
                    "annotation_name": "test_annotation",
                    "include": true,
                    "reasoning": "Valid",
                    "inclusion_criteria_applied": [],
                    "exclusion_criteria_applied": []
                }
            ]
        }
        '''
        
        mock_create.return_value = valid_response
        
        # Should succeed without validation issues
        decisions = client.make_decision(
            metadata=metadata,
            criteria_list=criteria_list,
            metadata_fields=["analysis_name"],
            model="gpt-4o-mini",
            prompt_type="single_analysis"
        )
        
        # Verify the request was called only once
        assert mock_create.call_count == 1
        
        # Verify decision was created with injected analysis_id
        assert len(decisions) == 1
        assert decisions[0].analysis_id == "12345_analysis_0"
