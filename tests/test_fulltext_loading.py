"""Test cases for full text loading functionality."""

import sys
from pathlib import Path

import pandas as pd

from autonima.models.types import Study, StudyStatus
from autonima.retrieval.utils import _load_full_text

# Add the autonima directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_load_full_text():
    """Test the _load_full_text function."""
    # Create a test CSV file
    test_data = {
        'pmcid': ['PMC123456', 'PMC789012'],
        'text': ['This is the full text for study 1.',
                 'This is the full text for study 2.']
    }
    
    df = pd.DataFrame(test_data)
    test_file = Path('test_text.csv')
    df.to_csv(test_file, index=False)
    
    try:
        # Create a test study
        study = Study(
            pmid="123456",
            title="Test Study",
            abstract="Test abstract",
            authors=["Author1", "Author2"],
            journal="Test Journal",
            publication_date="2023-01-01",
            pmcid="PMC123456"
        )
        
        # Test loading full text
        result = _load_full_text(study, str(test_file))
        
        # Verify the result
        expected = 'This is the full text for study 1.'
        assert result == expected, f"Expected '{expected}', got {result}"
        
        # Test with a study that doesn't exist in the CSV
        study2 = Study(
            pmid="789012",
            title="Test Study 2",
            abstract="Test abstract 2",
            authors=["Author3", "Author4"],
            journal="Test Journal 2",
            publication_date="2023-01-02",
            pmcid="PMC999999"  # This PMCID doesn't exist in the CSV
        )
        
        result2 = _load_full_text(study2, str(test_file))
        assert result2 is None, f"Expected None, got {result2}"
        
        print("All tests passed!")
        
    finally:
        # Clean up the test file
        if test_file.exists():
            test_file.unlink()


def test_study_load_full_text():
    """Test the Study.load_full_text method."""
    # Create a test CSV file in the standard location
    test_data = {
        'pmcid': ['PMC123456', 'PMC789012'],
        'text': ['This is the full text for study 1.',
                 'This is the full text for study 2.']
    }
    
    df = pd.DataFrame(test_data)
    # Use the standard path for pubget data
    test_dir = (Path(__file__).parent.parent / 'test_output' / 'retrieval' /
                'pubget_data')
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / 'text.csv'
    df.to_csv(test_file, index=False)
    
    try:
        # Create a test study
        study = Study(
            pmid="123456",
            title="Test Study",
            abstract="Test abstract",
            authors=["Author1", "Author2"],
            journal="Test Journal",
            publication_date="2023-01-01",
            pmcid="PMC123456",
            status=StudyStatus.FULLTEXT_RETRIEVED
        )
        
        # Test loading full text through the Study method
        result = study.load_full_text()
        
        # Verify the result
        expected = 'This is the full text for study 1.'
        assert result == expected, f"Expected '{expected}', got {result}"
        
        print("Study load_full_text test passed!")
        
    finally:
        # Clean up the test file
        if test_file.exists():
            test_file.unlink()
        # Clean up the test directory if it's empty
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()


if __name__ == "__main__":
    test_load_full_text()
    test_study_load_full_text()