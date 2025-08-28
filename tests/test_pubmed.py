"""Pytest tests for PubMed search functionality."""

import pytest
import asyncio
from autonima.models.types import SearchConfig
from autonima.search.pubmed import PubMedSearch


@pytest.fixture
def search_config():
    """Create a search configuration for testing."""
    return SearchConfig(
        database="pubmed",
        query="fMRI AND schizophrenia",
        max_results=3,  # Limit for faster tests
        email="test@example.com"
    )


@pytest.fixture
def pubmed_search(search_config):
    """Create a PubMedSearch instance for testing."""
    return PubMedSearch(search_config)


def test_pubmed_search_initialization(search_config):
    """Test PubMedSearch initialization."""
    search_engine = PubMedSearch(search_config)
    
    assert search_engine.config == search_config
    expected_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    assert search_engine.base_url == expected_url
    assert search_engine.max_retries == 3
    assert search_engine.retry_delay == 1.0


def test_pubmed_search_execute(search_config, pubmed_search):
    """Test executing a PubMed search."""
    # Execute the search
    studies = asyncio.run(pubmed_search.search(search_config.query))
    
    # Verify results
    assert isinstance(studies, list)
    assert len(studies) <= search_config.max_results
    
    # If we have results, verify their structure
    if studies:
        study = studies[0]
        assert hasattr(study, 'pmid')
        assert hasattr(study, 'title')
        assert hasattr(study, 'abstract')
        assert hasattr(study, 'authors')
        assert hasattr(study, 'journal')
        assert hasattr(study, 'publication_date')
        
        # Verify types
        assert isinstance(study.pmid, str)
        assert isinstance(study.title, str)
        assert isinstance(study.abstract, str)
        assert isinstance(study.authors, list)
        assert isinstance(study.journal, str)
        assert isinstance(study.publication_date, str)


def test_pubmed_search_get_info(pubmed_search):
    """Test getting search engine information."""
    info = pubmed_search.get_search_info()
    
    assert isinstance(info, dict)
    assert info['engine'] == 'pubmed'
    expected_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    assert info['api_url'] == expected_url
    assert 'max_results' in info
    assert 'email' in info



def test_pubmed_search_empty_query(search_config):
    """Test PubMed search with empty query."""
    search_config.query = ""
    search_engine = PubMedSearch(search_config)
    
    # Execute the search - should raise an exception for empty query
    with pytest.raises(Exception):
        asyncio.run(search_engine.search(search_config.query))


# Integration test with the actual pipeline
def test_pubmed_search_integration(search_config):
    """Integration test with PubMed search."""
    search_engine = PubMedSearch(search_config)
    
    # Execute the search
    studies = asyncio.run(search_engine.search(search_config.query))
    
    # Verify we can access all expected fields
    for study in studies[:2]:  # Check first 2 studies
        assert study.pmid and isinstance(study.pmid, str)
        assert study.title and isinstance(study.title, str)
        assert isinstance(study.abstract, str)  # Can be empty
        assert isinstance(study.authors, list)
        assert study.journal and isinstance(study.journal, str)
        pub_date = study.publication_date
        assert pub_date and isinstance(pub_date, str)
        assert isinstance(study.keywords, list)
        assert study.doi is None or isinstance(study.doi, str)