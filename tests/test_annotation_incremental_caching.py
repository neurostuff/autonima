"""Pytest tests for incremental caching in AnnotationProcessor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.schema import (AnnotationConfig,
                                        AnnotationCriteriaConfig)
from autonima.models.types import Study
from autonima.coordinates.schema import Analysis


def create_mock_study(pmid, num_analyses=2, status="INCLUDED_FULLTEXT"):
    """Create a mock study with analyses for testing."""
    # Create Analysis objects
    analyses = []
    for i in range(num_analyses):
        analysis = Analysis(
            name=f"Analysis {i}",
            description=f"Description {i}",
            points=[]  # Empty points for testing
        )
        analyses.append(analysis)
    
    # Create a simple mock study object
    study = Study(
        pmid=pmid,
        title=f"Test Study {pmid}",
        abstract="This is a test abstract",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2023-01-01",
        status=status,
        analyses=analyses
    )
    return study


def test_study_based_caching_all_analyses_annotation():
    """Test study-based caching for 'all_analyses' annotation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create annotation config
        config = AnnotationConfig(
            create_all_included_annotations=True,
            annotations=[]
        )
        
        # Mock the LLM client
        with patch('autonima.annotation.client.GenericLLMClient') as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the function calling response for single decision
            mock_function_call = MagicMock()
            mock_function_call.arguments = (
                '{"include": true, "reasoning": "Meets criteria"}'
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
            
            # Mock the function calling response for multi decision
            mock_multi_function_call = MagicMock()
            mock_multi_function_call.arguments = (
                '{"decisions": [{"annotation_name": "test_annotation", '
                '"include": true, "reasoning": "Meets all inclusion '
                'criteria"}]}'
            )
            mock_multi_message = MagicMock()
            mock_multi_message.function_call = mock_multi_function_call
            mock_multi_choice = MagicMock()
            mock_multi_choice.message = mock_multi_message
            mock_multi_api_response = MagicMock()
            mock_multi_api_response.choices = [mock_multi_choice]
            # We need to mock the second call to create
            mock_client.client.chat.completions.create.side_effect = [
                mock_api_response, mock_multi_api_response
            ]
            
            # Create processor
            processor = AnnotationProcessor(config)
            
            # Create test studies
            studies = [create_mock_study("12345", 2),
                       create_mock_study("67890", 1)]
            
            # Process studies for the first time
            results1 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Check that we got results
            assert len(results1) > 0
            assert all(r.annotation_name == "all_analyses" for r in results1)
            
            # Check that results were saved to cache
            cache_file = Path(temp_dir) / "outputs" / "annotation_results.json"
            assert cache_file.exists()
            
            # Load and verify cache contents
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            assert len(cached_data) == len(results1)
            
            # Process studies again - should load from cache
            results2 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Should have the same results
            assert len(results2) == len(results1)
            assert all(r1.study_id == r2.study_id and
                       r1.annotation_name == r2.annotation_name
                       for r1, r2 in zip(results1, results2))


def test_study_based_caching_custom_annotations():
    """Test study-based caching for custom annotations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create annotation config with custom annotations
        config = AnnotationConfig(
            create_all_included_annotations=False,
            annotations=[
                AnnotationCriteriaConfig(
                    name="test_annotation",
                    description="Test annotation for testing",
                    inclusion_criteria=["Test inclusion criterion"],
                    exclusion_criteria=[]
                )
            ]
        )
        
        # Mock the LLM client
        with patch('autonima.annotation.client.GenericLLMClient') as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the make_decision method to return a predefined response
            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            mock_response.criteria_ids = []
            
            # Mock the chat_completion method to return JSON
            mock_client.client.chat.completions.create.return_value.\
                choices.__getitem__.return_value.message.content = \
                '{"include": true, "reasoning": "Meets criteria"}'
            
            # Mock the make_multi_decision method to return a list of decisions
            mock_multi_response = MagicMock()
            mock_multi_response.decision = "INCLUDED"
            mock_multi_response.confidence = 0.95
            mock_multi_response.reason = "Meets all inclusion criteria"
            mock_multi_response.criteria_ids = []
            mock_client.make_multi_decision.return_value = [
                mock_multi_response]
            
            mock_client.make_decision.return_value = mock_response
            
            # Create processor
            processor = AnnotationProcessor(config)
            
            # Create test studies
            studies = [create_mock_study("12345", 2)]
            
            # Process studies for the first time
            results1 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Check that we got results
            assert len(results1) > 0
            assert all(r.annotation_name == "test_annotation"
                       for r in results1)
            
            # Check that results were saved to cache
            cache_file = Path(temp_dir) / "outputs" / "annotation_results.json"
            assert cache_file.exists()
            
            # Process studies again - should load from cache
            results2 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Should have the same results
            assert len(results2) == len(results1)
            assert all(r1.study_id == r2.study_id and
                       r1.annotation_name == r2.annotation_name
                       for r1, r2 in zip(results1, results2))


def test_study_based_caching_mixed_annotations():
    """Test study-based caching with both 'all_analyses' and
    custom annotations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create annotation config with both types
        config = AnnotationConfig(
            create_all_included_annotations=True,
            annotations=[
                AnnotationCriteriaConfig(
                    name="test_annotation",
                    description="Test annotation for testing",
                    inclusion_criteria=["Test inclusion criterion"],
                    exclusion_criteria=[]
                )
            ]
        )
        
        # Mock the LLM client
        with patch('autonima.annotation.client.GenericLLMClient') as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the make_decision method to return a predefined response
            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            mock_response.criteria_ids = []
            
            # Mock the chat_completion method to return JSON
            mock_client.client.chat.completions.create.return_value.\
                choices.__getitem__.return_value.message.content = \
                '{"include": true, "reasoning": "Meets criteria"}'
            
            # Mock the make_multi_decision method to return a list of decisions
            mock_multi_response = MagicMock()
            mock_multi_response.decision = "INCLUDED"
            mock_multi_response.confidence = 0.95
            mock_multi_response.reason = "Meets all inclusion criteria"
            mock_multi_response.criteria_ids = []
            mock_client.make_multi_decision.return_value = [
                mock_multi_response]
            
            mock_client.make_decision.return_value = mock_response
            
            # Create processor
            processor = AnnotationProcessor(config)
            
            # Create test studies
            studies = [create_mock_study("12345", 1)]
            
            # Process studies for the first time
            results1 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Check that we got results for both annotation types
            all_analyses_results = [r for r in results1
                                    if r.annotation_name == "all_analyses"]
            custom_results = [r for r in results1
                              if r.annotation_name == "test_annotation"]
            
            assert len(all_analyses_results) > 0
            assert len(custom_results) > 0
            
            # Process studies again - should load from cache
            results2 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Should have the same results
            assert len(results2) == len(results1)


def test_partial_cache_replacement():
    """Test that when a study is reprocessed, all its results are replaced."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create annotation config
        config = AnnotationConfig(
            create_all_included_annotations=True,
            annotations=[
                AnnotationCriteriaConfig(
                    name="test_annotation",
                    description="Test annotation",
                    inclusion_criteria=["Test criterion"],
                    exclusion_criteria=[]
                )
            ]
        )
        
        # Mock the LLM client
        with patch('autonima.annotation.client.GenericLLMClient') as \
                mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the make_decision method to return a predefined response
            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            mock_response.criteria_ids = []
            
            # Mock the chat_completion method to return JSON
            mock_client.client.chat.completions.create.return_value.\
                choices.__getitem__.return_value.message.content = \
                '{"include": true, "reasoning": "Meets criteria"}'
            
            # Mock the make_multi_decision method to return a list of decisions
            mock_multi_response = MagicMock()
            mock_multi_response.decision = "INCLUDED"
            mock_multi_response.confidence = 0.95
            mock_multi_response.reason = "Meets all inclusion criteria"
            mock_multi_response.criteria_ids = []
            mock_client.make_multi_decision.return_value = [
                mock_multi_response]
            
            mock_client.make_decision.return_value = mock_response
            
            # Create processor
            processor = AnnotationProcessor(config)
            
            # Create test studies
            studies = [create_mock_study("12345", 2)]
            
            # Process studies for the first time
            results1 = processor.process_studies(studies, output_dir=temp_dir)
            
            # Verify we have results for both annotation types
            all_analyses_count = len([r for r in results1
                                      if r.annotation_name == "all_analyses"])
            custom_count = len([r for r in results1
                                if r.annotation_name == "test_annotation"])
            assert all_analyses_count == 2  # 2 analyses
            assert custom_count == 2  # 2 analyses
            
            # Now modify the config to test cache replacement
            config2 = AnnotationConfig(
                create_all_included_annotations=True,
                annotations=[
                    AnnotationCriteriaConfig(
                        name="test_annotation_v2",  # Different annotation name
                        description="Test annotation version 2",
                        inclusion_criteria=["Test criterion v2"],
                        exclusion_criteria=[]
                    )
                ]
            )
            
            processor2 = AnnotationProcessor(config2)
            
            # Process with the new config - should replace all results
            # for study "12345"
            results2 = processor2.process_studies(studies, output_dir=temp_dir)
            
            # Should have results for "all_analyses" and "test_annotation_v2"
            # but not "test_annotation"
            all_analyses_results = [r for r in results2
                                    if r.annotation_name == "all_analyses"]
            custom_v2_results = [r for r in results2
                                 if r.annotation_name == "test_annotation_v2"]
            old_custom_results = [r for r in results2
                                  if r.annotation_name == "test_annotation"]
            
            assert len(all_analyses_results) == 2  # 2 analyses
            assert len(custom_v2_results) == 2  # 2 analyses
            assert len(old_custom_results) == 0  # Should be replaced


def test_create_all_included_annotations_generates_system_annotations():
    """Test that system annotations produce all_studies and all_abstract."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AnnotationConfig(
            create_all_included_annotations=True,
            annotations=[]
        )

        processor = AnnotationProcessor(config)

        excluded_abstract = create_mock_study(
            "10001", num_analyses=1, status="EXCLUDED_ABSTRACT"
        )
        included_abstract = create_mock_study(
            "10002", num_analyses=1, status="INCLUDED_ABSTRACT"
        )
        excluded_fulltext = create_mock_study(
            "10003", num_analyses=1, status="EXCLUDED_FULLTEXT"
        )

        results = processor.process_studies(
            included_studies=[included_abstract],
            all_studies=[
                excluded_abstract,
                included_abstract,
                excluded_fulltext,
            ],
            all_abstract_studies=[
                included_abstract,
                excluded_fulltext,
            ],
            output_dir=temp_dir,
        )

        all_studies_results = [
            r for r in results if r.annotation_name == "all_studies"
        ]
        all_abstract_results = [
            r for r in results
            if r.annotation_name == "all_abstract"
        ]

        assert len(all_studies_results) == 3
        assert len(all_abstract_results) == 2

        all_studies_study_ids = {r.study_id for r in all_studies_results}
        all_abstract_study_ids = {
            r.study_id for r in all_abstract_results
        }
        assert all_studies_study_ids == {"10001", "10002", "10003"}
        assert all_abstract_study_ids == {"10002", "10003"}


if __name__ == "__main__":
    # Run the tests
    test_study_based_caching_all_analyses_annotation()
    test_study_based_caching_custom_annotations()
    test_study_based_caching_mixed_annotations()
    test_partial_cache_replacement()
    test_create_all_included_annotations_generates_system_annotations()
    print("All tests passed!")
