"""Pytest tests for objective inclusion in prompts."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener
from autonima.screening.prompts import PromptLibrary


def test_objective_in_abstract_prompt():
    """Test that objective is included in abstract screening prompt."""
    # Create a test study
    study = Study(
        pmid="TEST001",
        title="Test Study",
        abstract="This is a test abstract.",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2023",
        status=StudyStatus.PENDING
    )
    
    
    # Test prompt generation with objective
    objective = "Test objective for systematic review"
    prompt = PromptLibrary.get_abstract_screening_prompt(
        study=study,
        inclusion_criteria=["Test inclusion criterion"],
        exclusion_criteria=["Test exclusion criterion"],
        objective=objective
    )
    
    # Verify objective is in the prompt
    assert objective in prompt
    assert "MEtA-ANALYSIS OBJECTIVE:" in prompt


def test_objective_in_fulltext_prompt():
    """Test that objective is included in fulltext screening prompt."""
    # Create a test study
    study = Study(
        pmid="TEST001",
        title="Test Study",
        abstract="This is a test abstract.",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2023",
        status=StudyStatus.FULLTEXT_RETRIEVED,
        pmcid="123456"
    )
    
    
    # Create temporary directory and file for fulltext
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        retrieval_dir = temp_path / "retrieval" / "pubget_data"
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        text_file = retrieval_dir / "text.csv"
        
        # Create test CSV with fulltext content
        import pandas as pd
        test_data = {
            'pmcid': [123456],
            'body': ['This is the full text content for testing.']
        }
        df = pd.DataFrame(test_data)
        df.to_csv(text_file, index=False)
        
        # Test prompt generation with objective
        objective = "Test objective for systematic review"
        prompt = PromptLibrary.get_fulltext_screening_prompt(
            study=study,
            inclusion_criteria=["Test inclusion criterion"],
            exclusion_criteria=["Test exclusion criterion"],
            output_dir=str(temp_dir),
            objective=objective
        )
        
        # Verify objective is in the prompt
        assert objective in prompt
        assert "MEtA-ANALYSIS OBJECTIVE:" in prompt


def test_screener_with_objective():
    """Test that LLMScreener passes objective to prompts."""
    # Create a test study
    study = Study(
        pmid="TEST001",
        title="Test Study",
        abstract="This is a test abstract.",
        authors=["Test Author"],
        journal="Test Journal",
        publication_date="2023",
        status=StudyStatus.PENDING
    )
    
    # Create screening config
    config = ScreeningConfig()
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create unified screener with objective
        objective = "Test objective for systematic review"
        screener = LLMScreener(
            config, 
            output_dir=str(temp_dir),
            objective=objective
        )
        
        # Mock the LLM client
        with patch('autonima.screening.screener.GenericLLMClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the screen_abstract method to return a predefined response
            mock_response = MagicMock()
            mock_response.decision = "INCLUDED"
            mock_response.confidence = 0.95
            mock_response.reason = "Meets all inclusion criteria"
            mock_client.screen_abstract.return_value = mock_response
            
            # Test screening - this will call the prompt generation
            studies_list = [study]
            results = asyncio.run(screener.screen_abstracts(studies_list))
            
            # Verify that the screen_abstract method was called
            assert mock_client.screen_abstract.call_count == 1
            
            # Verify we got results
            assert len(results) == 1
            
            # Get the prompt that was passed to the LLM
            call_args = mock_client.screen_abstract.call_args
            prompt = call_args[0][0]  # First positional argument
            
            # Verify objective is in the prompt
            assert objective in prompt
            assert "MEtA-ANALYSIS OBJECTIVE:" in prompt


if __name__ == "__main__":
    test_objective_in_abstract_prompt()
    test_objective_in_fulltext_prompt()
    test_screener_with_objective()
    print("All objective in prompt tests passed!")