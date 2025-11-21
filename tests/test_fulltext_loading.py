"""Test cases for full text loading functionality."""

import sys
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import pytest

from autonima.models.types import Study, StudyStatus
from autonima.retrieval.utils import _load_full_text

# Add the autonima directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_load_full_text(temp_dir):
    """Test the _load_full_text function."""
    # Create a test CSV file in the standard location
    test_data = {
        'pmcid': ['123456', '789012'],
        'body': ['This is the full text for study 1.',
                 'This is the full text for study 2.']
    }
    
    df = pd.DataFrame(test_data)
    # Create standard directory structure
    test_dir = temp_dir / 'retrieval' / 'pubget_data'
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / 'text.csv'
    df.to_csv(test_file, index=False)
    
    # Create a test study
    study = Study(
        pmid="123456",
        title="Test Study",
        abstract="Test abstract",
        authors=["Author1", "Author2"],
        journal="Test Journal",
        publication_date="2023-01-01",
        pmcid="123456"
    )
    
    # Test loading full text with output_dir parameter
    result = _load_full_text(study, output_dir=str(temp_dir))
    
    # Verify the result
    expected = 'This is the full text for study 1.'
    assert result == expected, f"Expected '{expected}', got {result}"
    
    # Test with a study that doesn't exist in the CSV
    # study2 = Study(
    #     pmid="789012",
    #     title="Test Study 2",
    #     abstract="Test abstract 2",
    #     authors=["Author3", "Author4"],
    #     journal="Test Journal 2",
    #     publication_date="2023-01-02",
    #     pmcid="9999996"  # This PMCID doesn't exist in the CSV
    # )
    
    # Modify this to to expect a ValueError
    # result2 = _load_full_text(study2, text_path=str(test_file))
    # assert result2 is None, f"Expected None, got {result2}"
    
    print("All tests passed!")


def test_study_load_full_text(temp_dir):
    """Test the Study.load_full_text method."""
    # Create a test CSV file in the standard location
    test_data = {
        'pmcid': ['123456', '789012'],
        'body': ['This is the full text for study 1.',
                 'This is the full text for study 2.']
    }
    
    df = pd.DataFrame(test_data)
    # Use the temporary directory for pubget data
    test_dir = temp_dir / 'retrieval' / 'pubget_data'
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / 'text.csv'
    df.to_csv(test_file, index=False)
    
    # Create a test study
    study = Study(
        pmid="123456",
        title="Test Study",
        abstract="Test abstract",
        authors=["Author1", "Author2"],
        journal="Test Journal",
        publication_date="2023-01-01",
        pmcid="123456",
        status=StudyStatus.FULLTEXT_RETRIEVED
    )
    
    # Test loading full text through the Study method
    result = study.load_full_text(output_dir=str(temp_dir))
    
    # Verify the result
    expected = 'This is the full text for study 1.'
    assert result == expected, f"Expected '{expected}', got {result}"
    
    print("Study load_full_text test passed!")

