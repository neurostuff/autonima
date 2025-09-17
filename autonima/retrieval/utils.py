"""Utility functions for retrieval modules."""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union, List, Set, Dict
from ..models.types import Study


def _load_full_text(study: Study, text_path: str = None, output_dir: str = None) -> Optional[str]:
    """
    Load the full text content for a study from a CSV file or a direct text file.
    
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
        # If study has a direct full_text_path, load from that file
        if study.full_text_path:
            full_text_file = Path(study.full_text_path)
            if full_text_file.exists():
                # If it's a text file, read it directly
                if full_text_file.suffix.lower() == '.txt':
                    with open(full_text_file, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    raise ValueError(f"Unsupported file format: {full_text_file.suffix}")
        
        # Determine the text file path for CSV-based loading
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


def _map_pmids_to_text(
    root_path: Union[str, Path],
    pmid_source: str,
    text_path_templates: Optional[List[str]] = None,
    pmids_to_include: Optional[Set[int]] = None,
    json_filename: str = 'identifiers.json',
    json_pmid_key: str = 'pmid',
    allowed_extensions: Optional[List[str]] = None
) -> Dict[int, Path]:
    """
    Generically maps PubMed IDs (PMIDs) to their full-text file paths.

    Args:
        root_path (Union[str, Path]): The path to the root folder containing publications.
        pmid_source (str): Method to find the PMID. Must be one of:
                           'json': Look for a JSON file in each sub-directory.
                           'folder_name': Use the sub-directory's name as the PMID.
                           'file_name': Use the name of the file (without extension) as the PMID.
        text_path_templates (Optional[List[str]]): A list of relative path templates to search for the
                                                   text file, in order of preference. Required for 'json'
                                                   and 'folder_name' sources.
                                                   Example: ['processed/pubget/text.txt', 'text.txt']
        pmids_to_include (Optional[Set[int]]): An optional set of PMIDs to filter for.
                                               If provided, only these PMIDs will be included.
        json_filename (str): The name of the JSON file to read when pmid_source is 'json'.
                             Defaults to 'identifiers.json'.
        json_pmid_key (str): The key in the JSON file that holds the PMID.
                             Defaults to 'pmid'.
        allowed_extensions (Optional[List[str]]): A list of file extensions (e.g., ['.txt', '.xml'])
                                                  to consider when pmid_source is 'file_name'.
                                                  Defaults to ['.txt'].

    Returns:
        Dict[int, Path]: A dictionary mapping integer PMIDs to the Path object of their text file.
    """
    root = Path(root_path)
    index = {}
    
    # Set default for file_name mode
    if pmid_source == 'file_name' and allowed_extensions is None:
        allowed_extensions = ['.txt']

    # Validate parameters
    if pmid_source in ['json', 'folder_name'] and not text_path_templates:
        raise ValueError("`text_path_templates` must be provided for 'json' and 'folder_name' pmid_source.")

    iterator = root.iterdir()

    for item in iterator:
        pmid = None
        text_file_path = None

        if pmid_source == 'json' and item.is_dir():
            id_file = item / json_filename
            if id_file.exists():
                try:
                    with open(id_file) as f:
                        data = json.load(f)
                    raw_pmid = data.get(json_pmid_key)
                    if raw_pmid is not None:
                        pmid = int(raw_pmid)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue  # Skip if JSON is invalid or PMID is not an integer

            if pmid:
                # Find the best full text file using the templates
                for template in text_path_templates:
                    candidate = item / template
                    if candidate.exists():
                        text_file_path = candidate
                        break # Found the preferred version

        elif pmid_source == 'folder_name' and item.is_dir():
            try:
                pmid = int(item.name)
            except ValueError:
                continue # Folder name is not a valid integer PMID
            
            # Find the best full text file using the templates
            for template in text_path_templates:
                candidate = item / template
                if candidate.exists():
                    text_file_path = candidate
                    break

        elif pmid_source == 'file_name' and item.is_file():
            if item.suffix in allowed_extensions:
                try:
                    pmid = int(item.stem)
                    text_file_path = item
                except ValueError:
                    continue # File stem is not a valid integer PMID

        # If we have a valid PMID and its text file, add it to the index
        if pmid and text_file_path:
            if pmids_to_include is None or pmid in pmids_to_include:
                index[pmid] = text_file_path

    return index