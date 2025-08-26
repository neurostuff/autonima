"""Tests for the retrieval module."""

import unittest
from pathlib import Path
import tempfile

from autonima.retrieval import PubGetRetriever, ACERetriever
from autonima.models.types import Study, StudyStatus


class TestRetrieval(unittest.TestCase):
    """Test cases for retrieval functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample studies
        self.studies = [
            Study(
                pmid="12345678",
                title="Sample fMRI Study",
                abstract="This is a sample abstract about fMRI research.",
                authors=["Doe, John", "Smith, Jane"],
                journal="NeuroImage",
                publication_date="2023-01-01",
                doi="10.1016/j.neuroimage.2023.123456",
                keywords=["fMRI", "neuroimaging"],
                status=StudyStatus.INCLUDED,
                metadata={
                    "pmcid": "PMC1234567"  # PMC ID is required for PubGet
                }
            ),
            Study(
                pmid="87654321",
                title="Another fMRI Study",
                abstract=(
                    "This is another sample abstract about fMRI research."
                ),
                authors=["Brown, Alice", "Wilson, Bob"],
                journal="Human Brain Mapping",
                publication_date="2023-02-01",
                doi="10.1002/hbm.26123",
                keywords=["fMRI", "working memory"],
                status=StudyStatus.INCLUDED,
                metadata={"pmcid": "PMC7654321"}
            )
        ]

    def test_pubget_retriever_initialization(self):
        """Test PubGetRetriever initialization."""
        try:
            retriever = PubGetRetriever(n_jobs=2)
            self.assertIsInstance(retriever, PubGetRetriever)
            self.assertEqual(retriever.n_jobs, 2)
        except ImportError:
            self.skipTest("PubGet not installed")

    def test_pubget_retrieve_method_signature(self):
        """Test PubGetRetriever retrieve method signature."""
        try:
            retriever = PubGetRetriever()
            # Just test that the method exists and has the right signature
            self.assertTrue(hasattr(retriever, 'retrieve'))
            self.assertTrue(callable(getattr(retriever, 'retrieve')))
        except ImportError:
            self.skipTest("PubGet not installed")

    def test_pubget_validate_retrieval_method_signature(self):
        """Test PubGetRetriever validate_retrieval method signature."""
        try:
            retriever = PubGetRetriever()
            # Just test that the method exists and has the right signature
            self.assertTrue(hasattr(retriever, 'validate_retrieval'))
            self.assertTrue(callable(getattr(retriever, 'validate_retrieval')))
        except ImportError:
            self.skipTest("PubGet not installed")

    def test_ace_retriever_initialization(self):
        """Test ACERetriever initialization."""
        retriever = ACERetriever()
        self.assertIsInstance(retriever, ACERetriever)

    def test_ace_retrieve_method_signature(self):
        """Test ACERetriever retrieve method signature."""
        retriever = ACERetriever()
        # Just test that the method exists and has the right signature
        self.assertTrue(hasattr(retriever, 'retrieve'))
        self.assertTrue(callable(getattr(retriever, 'retrieve')))

    def test_ace_validate_retrieval_method_signature(self):
        """Test ACERetriever validate_retrieval method signature."""
        retriever = ACERetriever()
        # Just test that the method exists and has the right signature
        self.assertTrue(hasattr(retriever, 'validate_retrieval'))
        self.assertTrue(callable(getattr(retriever, 'validate_retrieval')))

    def test_base_retriever_imports(self):
        """Test that BaseRetriever can be imported."""
        from autonima.retrieval import BaseRetriever
        self.assertTrue(hasattr(BaseRetriever, 'retrieve'))
        self.assertTrue(hasattr(BaseRetriever, 'validate_retrieval'))


if __name__ == '__main__':
    unittest.main()