"""Tests for the pipeline retrieval functionality."""

import unittest
from pathlib import Path
import tempfile

from autonima.pipeline import AutonimaPipeline
from autonima.config import ConfigManager


class TestPipelineRetrieval(unittest.TestCase):
    """Test cases for pipeline retrieval functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a config manager and sample config
        self.config_manager = ConfigManager()
        self.sample_config = self.config_manager.create_sample_config()
        
        # Modify config for testing
        self.sample_config.search.max_results = 2  # Limit for faster testing
        self.sample_config.output.directory = str(self.temp_dir / "output")

    def test_pipeline_retriever_initialization(self):
        """Test that the pipeline initializes with the retriever."""
        # This should not raise an exception
        pipeline = AutonimaPipeline(self.sample_config)
        
        # Check that the retriever was initialized
        self.assertIsNotNone(pipeline._retriever)
        
    def test_pipeline_retrieval_config(self):
        """Test that the retrieval config is properly set."""
        pipeline = AutonimaPipeline(self.sample_config)
        _ = pipeline  # Use the variable to avoid unused variable warning
        
        # Check that the retrieval config has the expected attributes
        self.assertTrue(hasattr(self.sample_config.retrieval, 'n_jobs'))
        self.assertTrue(hasattr(self.sample_config.retrieval, 'sources'))
        self.assertTrue(hasattr(self.sample_config.retrieval, 'fallback'))

    def test_retrieval_phase_method_exists(self):
        """Test that the retrieval phase method exists."""
        pipeline = AutonimaPipeline(self.sample_config)
        
        # Check that the retrieval phase method exists
        self.assertTrue(hasattr(pipeline, '_execute_retrieval_phase'))
        self.assertTrue(
            callable(getattr(pipeline, '_execute_retrieval_phase'))
        )


if __name__ == '__main__':
    unittest.main()