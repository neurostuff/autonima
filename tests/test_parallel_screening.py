"""Pytest tests for the parallel screening functionality."""

import asyncio
import time
from unittest.mock import patch, MagicMock
from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener


def test_parallel_screening():
    """Test parallel screening functionality."""
    # Create test studies
    studies = []
    for i in range(5):
        study = Study(
            pmid=f"TEST{i:03d}",
            title=f"Test Study {i}",
            abstract=f"This is abstract number {i} for testing parallel processing.",
            authors=[f"Author {i}"],
            journal="Test Journal",
            publication_date="2023",
            status=StudyStatus.PENDING
        )
        studies.append(study)
    
    # Create screening config
    config = ScreeningConfig()
    
    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return a predefined response
        mock_response = MagicMock()
        mock_response.decision = "INCLUDED"
        mock_response.confidence = 0.95
        mock_response.reason = "Meets all inclusion criteria"
        mock_client.screen_abstract.return_value = mock_response
        
        # Create unified screener
        screener = LLMScreener(config, num_workers=3)
        
        # Clear cache to ensure we're testing the actual screening
        screener.clear_cache()
        
        # Test parallel screening
        start_time = time.time()
        results = asyncio.run(screener.screen_abstracts(studies, num_workers=3))
        end_time = time.time()
        parallel_time = end_time - start_time
        
        assert isinstance(results, list)
        # We should get exactly 5 results
        assert len(results) == 5
        
        # Verify that all results have the expected structure
        for i, result in enumerate(results):
            assert result.study_id == f"TEST{i:03d}"
            assert result.decision == StudyStatus.INCLUDED
            assert result.confidence == 0.95
            assert result.reason == "Meets all inclusion criteria"
            assert result.screening_type == "abstract"
        
        # Verify that the screen_abstract method was called the expected number of times
        assert mock_client.screen_abstract.call_count == 5
        
        # Test serial screening for comparison
        screener.clear_cache()
        start_time = time.time()
        results_serial = asyncio.run(screener.screen_abstracts(studies, num_workers=1))
        end_time = time.time()
        serial_time = end_time - start_time
        
        # Both should produce the same results
        assert len(results_serial) == len(results)
        
        print(f"Parallel screening time: {parallel_time:.2f}s")
        print(f"Serial screening time: {serial_time:.2f}s")


def test_parallel_screening_with_errors():
    """Test parallel screening with some errors."""
    # Create test studies
    studies = []
    for i in range(3):
        study = Study(
            pmid=f"TEST_ERR{i:03d}",
            title=f"Test Study {i}",
            abstract=f"This is abstract number {i} for testing error handling.",
            authors=[f"Author {i}"],
            journal="Test Journal",
            publication_date="2023",
            status=StudyStatus.PENDING
        )
        studies.append(study)
    
    # Create screening config
    config = ScreeningConfig()
    
    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return different responses
        # Make the second call fail
        def mock_screen_abstract(prompt, model):
            if "abstract number 1" in prompt:
                raise Exception("Simulated API error")
            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            return mock_response
        
        mock_client.screen_abstract.side_effect = mock_screen_abstract
        
        # Create unified screener
        screener = LLMScreener(config, num_workers=2)
        
        # Clear cache to ensure we're testing the actual screening
        screener.clear_cache()
        
        # Test parallel screening with errors
        results = asyncio.run(screener.screen_abstracts(studies, num_workers=2))
        
        assert isinstance(results, list)
        # We should get exactly 3 results (even with errors)
        assert len(results) == 3
        
        # First and third should be successful
        assert results[0].decision == StudyStatus.INCLUDED
        assert results[2].decision == StudyStatus.INCLUDED
        
        # Second should be a failure
        assert results[1].decision == StudyStatus.SCREENING_FAILED
        assert "Screening failed" in results[1].reason


if __name__ == "__main__":
    test_parallel_screening()
    test_parallel_screening_with_errors()
    print("All parallel screening tests passed!")