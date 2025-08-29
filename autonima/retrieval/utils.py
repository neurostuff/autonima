"""Utility functions for retrieval modules."""

import pandas as pd
from pathlib import Path
from typing import Optional
from ..models.types import Study


def _load_full_text(study: Study, text_path: str) -> Optional[str]:
    """
    Load the full text content for a study from a CSV file.
    
    Args:
        study: The study object containing the pmcid
        text_path: Path to the text.csv file
        
    Returns:
        The full text content as a string, or None if not found
    """
    try:
        # Check if the text file exists
        text_file = Path(text_path)
        if not text_file.exists():
            return None
            
        # Read the CSV file
        df = pd.read_csv(text_file)
        
        # Look for the row matching the study's pmcid
        if study.pmcid:
            row = df[df['pmcid'] == study.pmcid]
            if not row.empty:
                return row.iloc[0]['text']
        
        # If no pmcid or not found by pmcid, return None
        return None
        
    except Exception:
        # Handle any errors during file reading or processing
        return None