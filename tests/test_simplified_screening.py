"""Pytest tests for the simplified screening module."""

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


def test_unified_screener_initialization():
    """Test the unified LLMScreener initialization."""
    # Create screening config with inclusion/exclusion criteria
    config = ScreeningConfig()
    
    # Add some test criteria
    config.inclusion_criteria = ["fMRI neuroimaging", "Human participants"]
    config.exclusion_criteria = ["Animal studies", "Review articles"]

    # Create unified screener
    screener = LLMScreener(
        config,
        output_dir=str(temp_dir),
        inclusion_criteria=config.inclusion_criteria,
        exclusion_criteria=config.exclusion_criteria
    )
    
    assert screener.config == config


def test_unified_screener_abstract_screening(temp_dir):
    """Test the unified LLMScreener abstract screening functionality."""
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

    # Create screening config with inclusion/exclusion criteria
    config = ScreeningConfig()
    config.inclusion_criteria = ["fMRI neuroimaging", "Human participants"]
    config.exclusion_criteria = ["Animal studies", "Review articles"]

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
        screener = LLMScreener(
            config,
            output_dir=str(temp_dir),
            inclusion_criteria=config.inclusion_criteria,
            exclusion_criteria=config.exclusion_criteria
        )

        # Test abstract screening
        studies_list = [study]
        abstract_results = asyncio.run(screener.screen_abstracts(studies_list))
        
        assert isinstance(abstract_results, list)
        # We should get exactly one result
        assert len(abstract_results) == 1
        
        result = abstract_results[0]
        assert result.study_id == "TEST001"
        assert hasattr(result, 'decision')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reason')


def test_unified_screener_get_info(temp_dir):
    """Test getting screening engine information."""
    # Create screening config
    config = ScreeningConfig()
    config.inclusion_criteria = ["fMRI neuroimaging", "Human participants"]
    config.exclusion_criteria = ["Animal studies", "Review articles"]

    # Create unified screener
    screener = LLMScreener(
        config,
        output_dir=str(temp_dir),
        inclusion_criteria=config.inclusion_criteria,
        exclusion_criteria=config.exclusion_criteria
    )

    # Test screening info
    info = screener.get_screening_info()
    
    assert isinstance(info, dict)
    assert 'engine' in info
    assert 'abstract_model' in info
    assert 'fulltext_model' in info