"""Pytest tests for multi-annotation functionality."""

from unittest.mock import patch, MagicMock
from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.schema import (AnnotationConfig,
                                        AnnotationCriteriaConfig)


def test_multi_annotation_prompt_creation():
    """Test that multi-annotation prompts are created correctly."""
    from autonima.annotation.prompts import create_multi_annotation_prompt
    from autonima.annotation.schema import AnalysisMetadata
    
    # Create mock metadata
    metadata = AnalysisMetadata(
        analysis_id="12345_analysis_0",
        study_id="12345",
        analysis_name="Test Analysis",
        analysis_description="Test description",
        study_title="Test Study"
    )
    
    # Create mock criteria
    criteria1 = AnnotationCriteriaConfig(
        name="emotion",
        description="Emotion-related analyses",
        inclusion_criteria=["Emotion processing", "Affective stimuli"],
        exclusion_criteria=["Non-emotional tasks"]
    )
    
    criteria2 = AnnotationCriteriaConfig(
        name="cognitive_control",
        description="Cognitive control analyses",
        inclusion_criteria=["Conflict monitoring", "Inhibitory control"],
        exclusion_criteria=["Passive tasks"]
    )
    
    # Create multi-annotation prompt
    prompt = create_multi_annotation_prompt(metadata, [criteria1, criteria2])
    
    # Check that prompt contains information about both annotations
    assert "emotion" in prompt.lower()
    assert "cognitive_control" in prompt.lower()
    assert "Emotion processing" in prompt
    assert "Conflict monitoring" in prompt
    assert "ANNOTATION 1" in prompt
    assert "ANNOTATION 2" in prompt


def test_annotation_processor_multi_decision():
    """Test that annotation processor can handle multiple annotations."""
    # Create annotation config with multiple annotations
    config = AnnotationConfig(
        create_all_included_annotation=True,
        annotations=[
            AnnotationCriteriaConfig(
                name="emotion",
                description="Emotion-related analyses",
                inclusion_criteria=["Emotion processing"],
                exclusion_criteria=["Non-emotional tasks"]
            ),
            AnnotationCriteriaConfig(
                name="cognitive_control",
                description="Cognitive control analyses",
                inclusion_criteria=["Conflict monitoring"],
                exclusion_criteria=["Passive tasks"]
            )
        ]
    )
    
    # Mock the LLM client
    with patch('autonima.annotation.client.GenericLLMClient') as \
            mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the function calling response
        mock_function_call = MagicMock()
        mock_function_call.arguments = (
            '{"decisions": [{"annotation_name": "emotion", "include": true, '
            '"reasoning": "Meets emotion criteria"}, {"annotation_name": '
            '"cognitive_control", "include": false, "reasoning": "Does not '
            'meet cognitive control criteria"}]}'
        )
        mock_message = MagicMock()
        mock_message.function_call = mock_function_call
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]
        mock_client.client.chat.completions.create.return_value = (
            mock_api_response
        )
        
        # Create processor
        processor = AnnotationProcessor(config)
        
        # Verify that the processor was created successfully
        assert processor is not None
        assert len(processor.config.annotations) == 2


if __name__ == "__main__":
    # Run the tests
    test_multi_annotation_prompt_creation()
    test_annotation_processor_multi_decision()
    print("All multi-annotation tests passed!")