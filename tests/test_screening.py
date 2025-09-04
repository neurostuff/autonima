"""Pytest tests for the screening module."""

import asyncio
from unittest.mock import patch, MagicMock
from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener


def test_abstract_screening():
    """Test abstract screening functionality."""
    # Create a test study with a long abstract
    abstract_text = (
        "This study investigated the neural correlates of working memory "
        "deficits in schizophrenia using fMRI. Participants included 50 "
        "patients with schizophrenia and 50 healthy controls. Results showed "
        "altered activation in prefrontal cortex during working memory tasks."
    )
    
    study = Study(
        pmid="TEST001",
        title="fMRI study of working memory in schizophrenia",
        abstract=abstract_text,
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.PENDING
    )

    # Create screening config
    config = ScreeningConfig()

    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as \
         mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return a predefined response
        mock_response = MagicMock()
        mock_response.decision = "INCLUDED"
        mock_response.confidence = 0.95
        mock_response.reason = "Meets all inclusion criteria"
        mock_client.screen_abstract.return_value = mock_response

        # Create unified screener
        screener = LLMScreener(config)

        # Test screening
        studies_list = [study]
        results = asyncio.run(screener.screen_abstracts(studies_list))

        assert isinstance(results, list)
        # We should get exactly one result
        assert len(results) == 1
        
        result = results[0]
        assert result.study_id == "TEST001"
        assert hasattr(result, 'decision')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reason')


def test_screener_initialization():
    """Test screener initialization."""
    # Create screening config
    config = ScreeningConfig()

    # Create unified screener
    screener = LLMScreener(config)
    
    assert screener.config == config


def test_screener_get_info():
    """Test getting screener information."""
    # Create screening config
    config = ScreeningConfig()

    # Create unified screener
    screener = LLMScreener(config)

    # Test screening info
    info = screener.get_screening_info()
    
    assert isinstance(info, dict)
    assert 'engine' in info
    assert 'abstract_model' in info
    assert 'fulltext_model' in info


def test_abstract_screening_parallel():
    """Test abstract screening functionality with parallel processing."""
    # Create test studies with long abstracts
    abstract_text = (
        "This study investigated the neural correlates of working memory "
        "deficits in schizophrenia using fMRI. Participants included 50 "
        "patients with schizophrenia and 50 healthy controls. Results showed "
        "altered activation in prefrontal cortex during working memory tasks."
    )
    
    study1 = Study(
        pmid="TEST001",
        title="fMRI study of working memory in schizophrenia",
        abstract=abstract_text,
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.PENDING
    )
    
    study2 = Study(
        pmid="TEST002",
        title="Another fMRI study of working memory",
        abstract=abstract_text,
        authors=["Brown T", "Davis K"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.PENDING
    )

    # Create screening config
    config = ScreeningConfig()

    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as \
         mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return a predefined response
        mock_response = MagicMock()
        mock_response.decision = "INCLUDED"
        mock_response.confidence = 0.95
        mock_response.reason = "Meets all inclusion criteria"
        mock_client.screen_abstract.return_value = mock_response

        # Create unified screener
        screener = LLMScreener(config)

        # Test screening with parallel=True
        studies_list = [study1, study2]
        results = asyncio.run(
            screener.screen_abstracts(studies_list, parallel=True)
        )

        assert isinstance(results, list)
        # We should get exactly two results
        assert len(results) == 2
        
        # Check first result
        result1 = results[0]
        assert result1.study_id == "TEST001"
        assert hasattr(result1, 'decision')
        assert hasattr(result1, 'confidence')
        assert hasattr(result1, 'reason')
        
        # Check second result
        result2 = results[1]
        assert result2.study_id == "TEST002"
        assert hasattr(result2, 'decision')
        assert hasattr(result2, 'confidence')
        assert hasattr(result2, 'reason')


if __name__ == "__main__":
    test_abstract_screening()
    test_screener_initialization()
    test_screener_get_info()
    test_abstract_screening_parallel()
    print("All screening tests passed!")