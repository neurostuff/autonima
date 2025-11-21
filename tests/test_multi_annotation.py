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
        table_id="table1",  # Required field
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


def test_study_multi_annotation_prompt_creation():
    """Test that study-level multi-annotation prompts are created correctly."""
    from autonima.annotation.prompts import (
        create_study_multi_annotation_prompt
    )
    from autonima.annotation.schema import (
        AnalysisMetadata, TableMetadata, StudyAnalysisGroup
    )
    
    # Create study group with 2 tables, 4 analyses
    study_group = StudyAnalysisGroup(
        study_id="12345",
        study_title="Test fMRI Study",
        study_abstract="A study about emotion processing",
        tables=[
            TableMetadata(
                table_id="table1",
                caption="Behavioral results",
                footer="N=30 participants"
            ),
            TableMetadata(
                table_id="table2",
                caption="Neural activation patterns"
            )
        ],
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table1",
                analysis_name="Accuracy by condition",
                analysis_description="Mean accuracy scores"
            ),
            AnalysisMetadata(
                analysis_id="12345_analysis_1",
                study_id="12345",
                table_id="table1",
                analysis_name="Response times",
                analysis_description="Mean RT in milliseconds"
            ),
            AnalysisMetadata(
                analysis_id="12345_analysis_2",
                study_id="12345",
                table_id="table2",
                analysis_name="Fear vs Neutral contrast",
                analysis_description="BOLD activation differences"
            ),
            AnalysisMetadata(
                analysis_id="12345_analysis_3",
                study_id="12345",
                table_id="table2",
                analysis_name="Amygdala ROI analysis",
                analysis_description="Region of interest analysis"
            )
        ]
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
    
    # Create study-level multi-annotation prompt
    prompt = create_study_multi_annotation_prompt(
        study_group, [criteria1, criteria2]
    )
    
    # Check that prompt contains information about both annotations
    assert "emotion" in prompt.lower()
    assert "cognitive_control" in prompt.lower()
    
    # Check that all tables are mentioned by their captions
    assert ("Behavioral results" in prompt or
            "behavioral results" in prompt.lower())
    assert ("Neural activation patterns" in prompt or
            "neural activation patterns" in prompt.lower())
    
    # Check that all analyses are mentioned
    assert "12345_analysis_0" in prompt
    assert "12345_analysis_1" in prompt
    assert "12345_analysis_2" in prompt
    assert "12345_analysis_3" in prompt
    
    # Check table grouping structure
    assert "TABLE 1" in prompt or "Table 1" in prompt
    assert "TABLE 2" in prompt or "Table 2" in prompt


def test_study_analysis_group_creation():
    """Test StudyAnalysisGroup schema validation."""
    from autonima.annotation.schema import (
        StudyAnalysisGroup, TableMetadata, AnalysisMetadata
    )
    
    # Test valid creation
    study_group = StudyAnalysisGroup(
        study_id="12345",
        tables=[
            TableMetadata(table_id="t1", caption="Table 1")
        ],
        analyses=[
            AnalysisMetadata(
                analysis_id="a1",
                study_id="12345",
                table_id="t1"
            )
        ]
    )
    
    assert study_group.study_id == "12345"
    assert len(study_group.tables) == 1
    assert len(study_group.analyses) == 1
    assert study_group.tables[0].table_id == "t1"
    assert study_group.analyses[0].table_id == "t1"


def test_table_metadata_nan_handling():
    """Test that TableMetadata handles NaN values correctly."""
    from autonima.annotation.schema import TableMetadata
    import math
    
    # Test with None
    table1 = TableMetadata(table_id="t1", caption=None, footer=None)
    assert table1.caption is None
    assert table1.footer is None
    
    # Test with NaN float
    table2 = TableMetadata(table_id="t2", caption=math.nan, footer="footer")
    assert table2.caption is None
    assert table2.footer == "footer"
    
    # Test with string "nan"
    table3 = TableMetadata(table_id="t3", caption="nan", footer="NaN")
    assert table3.caption is None
    assert table3.footer is None
    
    # Test with valid strings
    table4 = TableMetadata(
        table_id="t4",
        caption="Valid caption",
        footer="Valid footer"
    )
    assert table4.caption == "Valid caption"
    assert table4.footer == "Valid footer"