"""Pytest tests for the parallel screening functionality."""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener


def test_parallel_screening():
    """Test parallel screening functionality."""
    # Create a temporary directory for this test
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test studies
        studies = []
        for i in range(5):
            study = Study(
                pmid=f"PARALLEL_TEST_{i:03d}",
                title=f"Test Study {i}",
                abstract=(
                    f"This is abstract number {i} for testing parallel processing."
                ),
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
            
            # Create unified screener with temporary output directory
            screener = LLMScreener(config, num_workers=3, output_dir=str(temp_dir))
            
            # Test parallel screening
            start_time = time.time()
            results = asyncio.run(screener.screen_abstracts(studies, num_workers=3))
            end_time = time.time()
            parallel_time = end_time - start_time
            
            assert isinstance(results, list)
            # We should get exactly 5 results
            assert len(results) == 5
            
            # Verify that all results have the expected structure
            # Create a mapping of study_id to expected values for easier checking
            expected_results = {f"PARALLEL_TEST_{i:03d}": i for i in range(5)}
            
            # Check that we have the right number of results
            assert len(results) == 5
            
            # Check that each result has the expected structure
            for result in results:
                assert result.study_id in expected_results
                assert result.decision == StudyStatus.INCLUDED
                assert result.confidence == 0.95
                assert result.reason == "Meets all inclusion criteria"
                assert result.screening_type == "abstract"
                # Remove the study_id from expected_results to ensure no duplicates
                del expected_results[result.study_id]
                
            # Verify that all expected study_ids were found (expected_results should be empty)
            assert len(expected_results) == 0
            
            # Verify that the screen_abstract method was called the expected number of times
            assert mock_client.screen_abstract.call_count == 5
            
            # Test serial screening for comparison
            start_time = time.time()
            results_serial = asyncio.run(screener.screen_abstracts(studies, num_workers=1))
            end_time = time.time()
            serial_time = end_time - start_time
            
            # Both should produce the same results
            assert len(results_serial) == len(results)
            
            print(f"Parallel screening time: {parallel_time:.2f}s")
            print(f"Serial screening time: {serial_time:.2f}s")
            
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parallel_screening_with_errors():
    """Test parallel screening with some errors."""
    # Create a temporary directory for this test
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test studies
        studies = []
        for i in range(3):
            study = Study(
                pmid=f"PARALLEL_ERR_{i:03d}",
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
            
            # Create unified screener with temporary output directory
            screener = LLMScreener(config, num_workers=2, output_dir=str(temp_dir))
            
            # Test parallel screening with errors
            results = asyncio.run(screener.screen_abstracts(studies, num_workers=2))
            
            assert isinstance(results, list)
            # We should get exactly 3 results (even with errors)
            assert len(results) == 3
            
            # Create a mapping of results by study_id for easier checking
            results_by_id = {result.study_id: result for result in results}
            
            # Check that all expected study IDs are present
            expected_ids = {
                "PARALLEL_ERR_000",
                "PARALLEL_ERR_001",
                "PARALLEL_ERR_002"
            }
            assert set(results_by_id.keys()) == expected_ids
            
            # Check results for each study
            for study_id, result in results_by_id.items():
                if study_id == "PARALLEL_ERR_001":
                    # This one should have failed
                    assert result.decision == StudyStatus.SCREENING_FAILED
                    assert "Screening failed" in result.reason
                else:
                    # These should be successful
                    assert result.decision == StudyStatus.INCLUDED
                    assert result.confidence == 0.95
                    assert result.reason == "Meets all inclusion criteria"
                    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_parallel_screening()
    test_parallel_screening_with_errors()
    print("All parallel screening tests passed!")