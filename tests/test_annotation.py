"""Tests for the annotation module."""

import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from autonima.annotation.schema import AnnotationConfig, AnnotationCriteriaConfig
from autonima.annotation.processor import AnnotationProcessor
from autonima.models.types import Study, StudyStatus
from autonima.coordinates.schema import Analysis, CoordinatePoint


class TestAnnotationProcessor(unittest.TestCase):
    """Test cases for the AnnotationProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample annotation configuration
        self.criteria = AnnotationCriteriaConfig(
            name="test_annotation",
            description="Test annotation for testing",
            inclusion_criteria=["Include analyses with working memory tasks"],
            exclusion_criteria=["Exclude resting state analyses"]
        )
        
        self.config = AnnotationConfig(
            model="gpt-4o-mini",
            include_all_analyses=True,
            annotations=[self.criteria],
            metadata_fields=["analysis_name", "analysis_description"],
            inclusion_criteria=["Global inclusion criterion"],
            exclusion_criteria=["Global exclusion criterion"]
        )
        
        # Create a sample study with analyses
        self.point = CoordinatePoint(
            coordinates=[10.0, 20.0, 30.0],
            space="MNI"
        )
        
        self.analysis = Analysis(
            name="Working Memory Task",
            description="N-back working memory task with 2-back vs 0-back contrast",
            points=[self.point]
        )
        
        self.study = Study(
            pmid="12345678",
            title="Test Study on Working Memory",
            abstract="This is a test study about working memory tasks.",
            authors=["Doe, John", "Smith, Jane"],
            journal="Journal of Cognitive Neuroscience",
            publication_date="2023-01-01",
            status=StudyStatus.INCLUDED,
            analyses=[self.analysis]
        )
        
        self.processor = AnnotationProcessor(self.config)

    def test_init(self):
        """Test AnnotationProcessor initialization."""
        self.assertIsInstance(self.processor, AnnotationProcessor)
        self.assertEqual(self.processor.config, self.config)
        self.assertIsNotNone(self.processor.client)

    def test_create_all_analyses_annotations(self):
        """Test creating all analyses annotations."""
        studies = [self.study]
        decisions = self.processor._create_all_analyses_annotations(studies)
        
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].annotation_name, "all_analyses")
        self.assertEqual(decisions[0].analysis_id, "12345678_analysis_0")
        self.assertEqual(decisions[0].study_id, "12345678")
        self.assertTrue(decisions[0].include)
        self.assertEqual(decisions[0].reasoning, "All analyses included by default")
        self.assertEqual(decisions[0].model_used, "none")

    def test_extract_analysis_metadata(self):
        """Test extracting analysis metadata."""
        analysis_id = "12345678_analysis_0"
        metadata = self.processor._extract_analysis_metadata(self.study, self.analysis, analysis_id)
        
        self.assertEqual(metadata.analysis_id, analysis_id)
        self.assertEqual(metadata.study_id, "12345678")
        self.assertEqual(metadata.analysis_name, "Working Memory Task")
        self.assertEqual(metadata.analysis_description, "N-back working memory task with 2-back vs 0-back contrast")
        self.assertEqual(metadata.study_title, "Test Study on Working Memory")
        self.assertEqual(metadata.study_abstract, "This is a test study about working memory tasks.")

    @patch('autonima.annotation.processor.AnnotationClient.make_decision')
    def test_process_single_decision(self, mock_make_decision):
        """Test processing a single annotation decision."""
        # Mock the client response
        mock_decision = MagicMock()
        mock_decision.include = True
        mock_decision.reasoning = "Analysis meets inclusion criteria"
        mock_make_decision.return_value = mock_decision
        
        # Extract metadata
        analysis_id = "12345678_analysis_0"
        metadata = self.processor._extract_analysis_metadata(self.study, self.analysis, analysis_id)
        
        # Process decision
        decision = self.processor._process_single_decision(metadata, self.criteria, "gpt-4o-mini")
        
        # Verify the mock was called
        mock_make_decision.assert_called_once_with(metadata, self.criteria, "gpt-4o-mini")
        self.assertEqual(decision, mock_decision)

    def test_are_cached_results_valid(self):
        """Test validating cached results."""
        # Create some mock cached results
        from autonima.annotation.schema import AnnotationDecision
        cached_results = [
            AnnotationDecision(
                annotation_name="all_analyses",
                analysis_id="12345678_analysis_0",
                study_id="12345678",
                include=True,
                reasoning="All analyses included",
                model_used="none"
            ),
            AnnotationDecision(
                annotation_name="test_annotation",
                analysis_id="12345678_analysis_0",
                study_id="12345678",
                include=True,
                reasoning="Meets criteria",
                model_used="gpt-4o-mini"
            )
        ]
        
        # Test with valid results
        self.assertTrue(self.processor._are_cached_results_valid(cached_results))
        
        # Test with missing annotation
        self.config.include_all_analyses = True
        self.config.annotations = []
        self.assertFalse(self.processor._are_cached_results_valid(cached_results[:1]))

    def test_save_and_load_results(self):
        """Test saving and loading annotation results."""
        # Create some mock results
        from autonima.annotation.schema import AnnotationDecision
        decisions = [
            AnnotationDecision(
                annotation_name="test_annotation",
                analysis_id="12345678_analysis_0",
                study_id="12345678",
                include=True,
                reasoning="Meets criteria",
                model_used="gpt-4o-mini"
            )
        ]
        
        # Create a temporary directory for testing
        test_dir = Path("test_outputs")
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Test saving
            self.processor._save_results(decisions, str(test_dir))
            
            # Check that file was created
            cache_file = test_dir / "outputs" / "annotation_results.json"
            self.assertTrue(cache_file.exists())
            
            # Test loading
            loaded_results = self.processor._load_cached_results(str(test_dir))
            self.assertEqual(len(loaded_results), 1)
            self.assertEqual(loaded_results[0].annotation_name, "test_annotation")
            self.assertEqual(loaded_results[0].analysis_id, "12345678_analysis_0")
            self.assertTrue(loaded_results[0].include)
            
        finally:
            # Clean up
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()