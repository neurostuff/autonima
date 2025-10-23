"""Utility functions for retrieval modules."""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Set, Dict, Any
from ..models.types import Study, ActivationTable
from bs4 import BeautifulSoup, Comment

# Try to import readabilipy for enhanced HTML cleaning
try:
    from readabilipy import simple_json_from_html_string
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    logging.warning("readabilipy not installed. Install with 'pip install readabilipy' for enhanced HTML cleaning. "
                     "Note: Node.js is also required for readabilipy to work.")


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
                elif full_text_file.suffix.lower() == '.html':
                    # Load HTML body text
                    return _clean_html_with_readability(full_text_file.read_text(encoding='utf-8'))
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

    # For file_name option, recursively search all files
    if pmid_source == 'file_name':
        iterator = root.rglob('*')
    else:
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


def _safe_clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # 1. Remove non-text tags
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas"]):
        tag.decompose()

    # 2. Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 3. Strip heavy attributes but keep the tags/text
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr in ["style", "onclick", "class", "id", "aria-hidden", "aria-label"]:
                del tag[attr]

    return str(soup)


def _clean_html_with_readability(html: str) -> str:
    """
    Clean HTML content using Mozilla's readability algorithm via readabilipy.
    
    Falls back to _safe_clean_html if readabilipy is not available or fails.
    
    Args:
        html: The HTML content to clean
        
    Returns:
        The cleaned text content
    """
    global READABILITY_AVAILABLE
    
    # If readabilipy is not available, fall back to safe cleaning
    if not READABILITY_AVAILABLE:
        logging.warning("Falling back to basic HTML cleaning as readabilipy is not available")
        return _safe_clean_html(html)
    
    try:
        # Use readabilipy with Mozilla's readability algorithm
        article = simple_json_from_html_string(html, use_readability=True)
        if article and 'content' in article and article['content']:
            # Extract text content from the HTML
            soup = BeautifulSoup(article['content'], "lxml")
            # Get text content, preserving some structure
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text(strip=False)
                if text.strip():
                    text_parts.append(text.strip())
            return '\n\n'.join(text_parts) if text_parts else soup.get_text()
        else:
            # If readability failed to extract content, fall back to safe cleaning
            logging.warning("Readability failed to extract content, falling back to basic HTML cleaning")
            return _safe_clean_html(html)
    except Exception as e:
        # If any error occurs, fall back to safe cleaning
        logging.warning(f"Error using readabilipy, falling back to basic HTML cleaning: {e}")
        return _safe_clean_html(html)


def _load_activation_table_metadata(
    df: pd.DataFrame,
    root_path: Path,
    pmcids_to_include: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Turn a tables dataframe into a mapping of pmcid -> list of table metadata.
    """

    # Ensure at least one of the "file path" columns exists
    has_raw_file = 'table_raw_file' in df.columns
    has_data_file = 'table_data_file' in df.columns
    if not (has_raw_file or has_data_file):
        raise ValueError(
            "Missing required columns: must have either "
            "'table_raw_file' or 'table_data_file'"
        )

    # Required metadata columns
    required_columns = ['pmcid', 'table_id', 'table_label', 'table_caption', 'table_foot']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pmcid_to_tables: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in df.iterrows():
        pmcid = str(row['pmcid'])

        if pmcids_to_include is not None and pmcid not in pmcids_to_include:
            continue

        # Resolve file paths if present
        table_path = (
            str(root_path / row['table_raw_file'])
            if has_raw_file and pd.notna(row['table_raw_file'])
            and row['table_raw_file']
            else None
        )

        table_data_path = (
            str(root_path / row['table_data_file'])
            if has_data_file and pd.notna(row['table_data_file'])
            and row['table_data_file']
            else None
        )

        table_metadata = {
            'table_id': str(row['table_id']),
            'table_label': str(row['table_label']),
            'table_path': table_path,
            'table_data_path': table_data_path,
            'table_caption': (
                row['table_caption'] if pd.notna(row['table_caption']) else None
            ),
            'table_foot': (
                row['table_foot'] if pd.notna(row['table_foot']) else None
            ),
        }

        pmcid_to_tables.setdefault(pmcid, []).append(table_metadata)

    return pmcid_to_tables


def load_activation_table_map(
    data_dir: Path,
    pmcids_to_include: Optional[Set[str]] = None,
    filter_by_coordinates: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Core function: Load and (optionally) filter activation tables.
    Returns pmcid -> list of table metadata dicts.
    """
    tables_file = data_dir / "tables.csv"
    if not tables_file.exists():
        logging.info(f"No tables.csv in {data_dir}, skipping...")
        return {}

    try:
        df = pd.read_csv(tables_file)

        # Optional coordinate filtering
        coords_file = data_dir / "coordinates.csv"
        if filter_by_coordinates and coords_file.exists():
            coords_df = pd.read_csv(coords_file)
            df = df[
                df.set_index(["pmcid", "table_id"]).index
                .isin(coords_df.set_index(["pmcid", "table_id"]).index)
            ]

        return _load_activation_table_metadata(
            df=df,
            root_path=data_dir,
            pmcids_to_include=pmcids_to_include,
        )

    except Exception as e:
        logging.warning(f"Failed to load activation tables: {e}")
        return {}



def _map_pmcids_to_activation_tables(
    full_text_config: Dict[str, Any],
    pmcids_to_include: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    processed_data = full_text_config.get("processed_data")
    if not processed_data:
        return {}

    data_dir = Path(processed_data)
    return load_activation_table_map(
        data_dir=data_dir,
        pmcids_to_include=pmcids_to_include,
        filter_by_coordinates=False,  # 🚫 No coords in this flow
    )


def _apply_activation_tables_to_studies(
    studies: List["Study"],
    pmcid_to_tables: Dict[str, List[Dict[str, Any]]],
    clear_existing: bool = True,
) -> None:
    """
    Attach activation tables to studies based on PMCID mappings.
    """
    for study in studies:
        if not study.pmcid:
            continue

        if study.pmcid not in pmcid_to_tables:
            continue

        if clear_existing:
            study.activation_tables.clear()

        for t in pmcid_to_tables[study.pmcid]:
            study.activation_tables.append(
                ActivationTable(
                    table_id=t['table_id'],
                    table_label=t['table_label'],
                    table_path=t['table_path'],
                    table_data_path=t.get('table_data_path'),
                    table_caption=t['table_caption'],
                    table_foot=t['table_foot'],
                )
            )
