"""Pytest tests for the confidence reporting and threshold functionality."""

import asyncio
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_screener_with_confidence_reporting_enabled(temp_dir):
    """Test screener with confidence reporting enabled."""
    # Create a test study
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

    # Create screening config with confidence reporting enabled
    config = ScreeningConfig()
    config.abstract.update({
        "confidence_reporting": True,
        "threshold": 0.9,  # Explicitly set threshold
        "objective": "Test objective for screening",
        "inclusion_criteria": ["Test inclusion criterion"],
        "exclusion_criteria": ["Test exclusion criterion"]
    })

    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as \
         mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return a predefined response
        # with confidence below threshold
        mock_response = MagicMock()
        mock_response.decision = "INCLUDED"
        mock_response.confidence = 0.85  # Below threshold of 0.9
        mock_response.reason = "Meets all inclusion criteria"
        mock_client.screen_abstract.return_value = mock_response

        # Create unified screener
        screener = LLMScreener(
            config,
            output_dir=str(temp_dir)
        )

        # Test screening
        studies_list = [study]
        results = asyncio.run(screener.screen_abstracts(studies_list))

        assert isinstance(results, list)
        # We should get exactly one result
        assert len(results) == 1
        
        result = results[0]
        assert result.study_id == "TEST001"
        assert result.decision == StudyStatus.EXCLUDED
        assert "below threshold" in result.reason
        assert hasattr(result, 'confidence')


def test_screener_with_confidence_reporting_disabled(temp_dir):
    """Test screener with confidence reporting disabled."""
    # Create a test study
    abstract_text = (
        "This study investigated the neural correlates of working memory "
        "deficits in schizophrenia using fMRI. Participants included 50 "
        "patients with schizophrenia and 50 healthy controls. Results showed "
        "altered activation in prefrontal cortex during working memory tasks."
    )
    
    study = Study(
        pmid="TEST002",
        title="fMRI study of working memory in schizophrenia",
        abstract=abstract_text,
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.PENDING
    )

    # Create screening config with confidence reporting disabled (default)
    config = ScreeningConfig()
    config.abstract.update({
        "confidence_reporting": False,
        "threshold": 0.9,  # This should be ignored
        "objective": "Test objective for screening",
        "inclusion_criteria": ["Test inclusion criterion"],
        "exclusion_criteria": ["Test exclusion criterion"]
    })

    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as \
         mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return a predefined response
        mock_response = MagicMock()
        mock_response.decision = "INCLUDED"
        # Below threshold, but should be ignored
        mock_response.confidence = 0.85
        mock_response.reason = "Meets all inclusion criteria"
        mock_client.screen_abstract.return_value = mock_response

        # Create unified screener
        screener = LLMScreener(
            config,
            output_dir=str(temp_dir)
        )

        # Test screening
        studies_list = [study]
        results = asyncio.run(screener.screen_abstracts(studies_list))

        assert isinstance(results, list)
        # We should get exactly one result
        assert len(results) == 1
        
        result = results[0]
        assert result.study_id == "TEST002"
        # Should respect LLM's original decision since confidence
        # reporting is disabled
        assert result.decision == StudyStatus.INCLUDED
        assert "below threshold" not in result.reason
        assert hasattr(result, 'confidence')


def test_screener_with_no_threshold(temp_dir):
    """Test screener with confidence reporting enabled but no threshold."""
    # Create a test study
    abstract_text = (
        "This study investigated the neural correlates of working memory "
        "deficits in schizophrenia using fMRI. Participants included 50 "
        "patients with schizophrenia and 50 healthy controls. Results showed "
        "altered activation in prefrontal cortex during working memory tasks."
    )
    
    study = Study(
        pmid="TEST003",
        title="fMRI study of working memory in schizophrenia",
        abstract=abstract_text,
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.PENDING
    )

    # Create screening config with confidence reporting enabled but
    # no threshold
    config = ScreeningConfig()
    config.abstract.update({
        "confidence_reporting": True,
        "threshold": None,  # No threshold
        "objective": "Test objective for screening",
        "inclusion_criteria": ["Test inclusion criterion"],
        "exclusion_criteria": ["Test exclusion criterion"]
    })

    # Mock the LLM client
    with patch('autonima.screening.screener.GenericLLMClient') as \
         mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the screen_abstract method to return a predefined response
        mock_response = MagicMock()
        mock_response.decision = "EXCLUDED"
        mock_response.confidence = 0.85
        mock_response.reason = "Does not meet inclusion criteria"
        mock_client.screen_abstract.return_value = mock_response

        # Create unified screener
        screener = LLMScreener(
            config,
            output_dir=str(temp_dir)
        )

        # Test screening
        studies_list = [study]
        results = asyncio.run(screener.screen_abstracts(studies_list))

        assert isinstance(results, list)
        # We should get exactly one result
        assert len(results) == 1
        
        result = results[0]
        assert result.study_id == "TEST003"
        # Should respect LLM's original decision since no threshold is set
        assert result.decision == StudyStatus.EXCLUDED
        assert "below threshold" not in result.reason
        assert hasattr(result, 'confidence')