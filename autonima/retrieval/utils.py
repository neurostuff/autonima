"""Utility functions for retrieval modules."""

import pandas as pd
from pathlib import Path
from typing import Optional
from ..models.types import Study


def _load_full_text(study: Study, text_path: str = None, output_dir: str = None) -> Optional[str]:
    """
    Load the full text content for a study from a CSV file.
    
    Args:
        study: The study object containing the pmcid
        text_path: Path to the text.csv file (deprecated, use output_dir instead)
        output_dir: Output directory where pubget data is stored
        
    Returns:
        The full text content as a string, or None if not found
        
    Raises:
        ValueError: If neither text_path nor output_dir is provided
        FileNotFoundError: If the text file doesn't exist at the expected location
    """
    try:
        # Determine the text file path
        if text_path:
            text_file = Path(text_path)
        elif output_dir:
            # Construct the standard path: {output_dir}/retrieval/pubget_data/text.csv
            text_file = Path(output_dir) / "retrieval" / "pubget_data" / "text.csv"
        else:
            raise ValueError("Either text_path or output_dir must be provided")
        
        # Check if the text file exists
        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found at {text_file}")
            
        # Read the CSV file
        df = pd.read_csv(text_file)
        
        # Look for the row matching the study's pmcid
        if study.pmcid:
            row = df[df['pmcid'] == int(study.pmcid)]
            if not row.empty:
                return row.iloc[0]['body']
        
    except Exception:
        # Handle any errors during file reading or processing
        raise

    # If no matching pmcid found or no pmcid provided
    raise ValueError(f"No full text found for study with pmcid {study.pmcid}")