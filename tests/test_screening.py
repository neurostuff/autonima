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