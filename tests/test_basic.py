"""Pytest tests for basic Autonima functionality."""

import asyncio
from autonima.config import ConfigManager
from autonima.models.types import PipelineConfig
from autonima.pipeline import AutonimaPipeline


def test_configuration_loading():
    """Test configuration loading."""
    config_manager = ConfigManager()
    sample_config = config_manager.create_sample_config()
    
    assert isinstance(sample_config, PipelineConfig)
    assert sample_config.search.query is not None
    
    # Check that at least one screening stage has an objective
    abstract_objective = sample_config.screening.abstract.get('objective')
    fulltext_objective = sample_config.screening.fulltext.get('objective')
    assert abstract_objective is not None or fulltext_objective is not None
    
    # Check that at least one screening stage has inclusion criteria
    abstract_inclusion = sample_config.screening.abstract.get('inclusion_criteria')
    fulltext_inclusion = sample_config.screening.fulltext.get('inclusion_criteria')
    assert abstract_inclusion is not None or fulltext_inclusion is not None


def test_pipeline_initialization():
    """Test pipeline initialization."""
    config_manager = ConfigManager()
    sample_config = config_manager.create_sample_config()
    
    # This should not raise an exception
    pipeline = AutonimaPipeline(sample_config)
    
    assert pipeline.config == sample_config
    assert pipeline._search_engine is not None
    assert pipeline._screener is not None


def test_pipeline_search_functionality():
    """Test search functionality."""
    config_manager = ConfigManager()
    sample_config = config_manager.create_sample_config()
    
    # Limit the number of results for faster testing
    sample_config.search.max_results = 2
    
    pipeline = AutonimaPipeline(sample_config)
    
    # Execute search
    search_method = pipeline._search_engine.search
    query = sample_config.search.query
    studies = asyncio.run(search_method(query))
    
    # Verify results
    assert isinstance(studies, list)
    assert len(studies) <= sample_config.search.max_results
    
    # If we have results, verify their structure
    if studies:
        study = studies[0]
        assert hasattr(study, 'pmid')
        assert hasattr(study, 'title')
        assert hasattr(study, 'abstract')
        assert hasattr(study, 'authors')
        assert hasattr(study, 'journal')
        assert hasattr(study, 'publication_date')


def test_pipeline_statistics():
    """Test pipeline statistics generation."""
    config_manager = ConfigManager()
    sample_config = config_manager.create_sample_config()
    
    # Limit the number of results for faster testing
    sample_config.search.max_results = 2
    
    pipeline = AutonimaPipeline(sample_config)
    
    # Execute search
    asyncio.run(pipeline._execute_search_phase())
    
    # Get statistics
    stats = pipeline.get_statistics()
    
    assert isinstance(stats, dict)
    assert 'config' in stats
    assert 'results' in stats
    assert 'execution' in stats
    
    # Verify config stats
    # Check that at least one screening stage has an objective
    abstract_objective = sample_config.screening.abstract.get('objective')
    fulltext_objective = sample_config.screening.fulltext.get('objective')
    assert stats['config']['objective'] == (abstract_objective or fulltext_objective)
    assert stats['config']['database'] == sample_config.search.database
    assert stats['config']['query'] == sample_config.search.query
    
    # Verify results stats
    assert 'total_studies' in stats['results']
    assert 'included_studies' in stats['results']
    assert 'excluded_studies' in stats['results']