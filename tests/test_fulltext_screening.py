"""Pytest tests for the full text screening functionality."""

import asyncio
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener


def test_fulltext_screening():
    """Test full text screening functionality."""
    # Create a test CSV file with full text content
    test_data = {
        'pmcid': ['PMC123456'],
        'text': ['This is the full text content for testing full text '
                 'screening.']
    }
    
    df = pd.DataFrame(test_data)
    # Use the standard path for pubget data
    test_dir = (Path(__file__).parent.parent / 'test_output' / 'retrieval' /
                'pubget_data')
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / 'text.csv'
    df.to_csv(test_file, index=False)
    
    try:
        # Create a test study with full text
        study = Study(
            pmid="TEST001",
            title="fMRI study of working memory in schizophrenia",
            abstract="This is an abstract.",
            authors=["Smith J", "Johnson A"],
            journal="Neuroimage",
            publication_date="2023",
            pmcid="PMC123456",
            status=StudyStatus.FULLTEXT_RETRIEVED
        )

        # Create screening config
        config = ScreeningConfig()

        # Mock the LLM client
        with patch('autonima.screening.screener.GenericLLMClient') as \
             mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the screen_fulltext method to return a predefined response
            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            mock_client.screen_fulltext.return_value = mock_response

            # Create unified screener
            screener = LLMScreener(config)
            
            # Clear cache to ensure we're testing the actual screening
            screener.clear_cache()

            # Test screening
            studies_list = [study]
            results = asyncio.run(screener.screen_fulltexts(studies_list))

            assert isinstance(results, list)
            # We should get exactly one result
            assert len(results) == 1
            
            result = results[0]
            assert result.study_id == "TEST001"
            assert hasattr(result, 'decision')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'reason')
            
            # Verify that the screen_fulltext method was called
            mock_client.screen_fulltext.assert_called_once()
            
    finally:
        # Clean up the test file
        if test_file.exists():
            test_file.unlink()


def test_fulltext_screening_with_missing_pmcid():
    """Test full text screening with missing pmcid."""
    # Create a test study without pmcid
    study = Study(
        pmid="TEST002",
        title="fMRI study of working memory in schizophrenia",
        abstract="This is an abstract.",
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.FULLTEXT_RETRIEVED
    )

    # Create screening config
    config = ScreeningConfig()

    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as \
         mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create unified screener
        screener = LLMScreener(config)
        
        # Clear cache to ensure we're testing the actual screening
        screener.clear_cache()

        # Test screening - should skip studies with missing pmcid
        studies_list = [study]
        results = asyncio.run(screener.screen_fulltexts(studies_list))

        assert isinstance(results, list)
        # We should get no results because the pmcid is missing
        assert len(results) == 0


if __name__ == "__main__":
    test_fulltext_screening()
    test_fulltext_screening_with_missing_pmcid()
    print("All full text screening tests passed!")