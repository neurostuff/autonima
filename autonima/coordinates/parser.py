"""Coordinate table parser for neuroimaging results."""

import os
import glob
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
from tqdm import tqdm

from .openai_client import CoordinateParsingClient
from .schema import ParseAnalysesOutput

logger = logging.getLogger(__name__)


def parse_single_table(
    file_name: str,
    table_caption: str,
    table_foot: str,
    table_text: str,
    client: CoordinateParsingClient,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Parse a single table using the LLM client.
    
    Args:
        file_name: Name of the file being parsed
        table_caption: Caption of the table
        table_foot: Footer of the table
        table_text: Text content of the table
        client: CoordinateParsingClient instance
        model: Model to use for parsing
        
    Returns:
        Dictionary containing the parsed results
    """
    logger.info(f"Processing: {file_name}")
    
    # Create detailed prompt
    detailed_prompt = f"""
You are a neuroimaging data curation assistant.

You will receive a CSV table extracted from a published fMRI/neuroimaging article.
The table reports statistical activation results, typically with contrasts, regions, MNI coordinates, voxel counts, and test statistics.

Table Caption: {table_caption}
Table Foot: {table_foot}

Your task:
1. Parse the table into JSON grouped by *analysis* (e.g., each distinct contrast such as "PTSD < HC").
2. Keep all table metadata in a generic structured format (do NOT drop any rows).
3. Each row should be placed inside a `"points"` array under its corresponding analysis.
4. Missing values must be explicitly represented as `null`.
5. Coordinates must be grouped into an array with fields `"coordinates"` containing [x, y, z] values and `"space"` indicating the template space (e.g., MNI or TAL).
6. IMPORTANT: Only include rows that have valid coordinate data ([x, y, z] values). Skip rows that do not contain coordinate information.
7. Each point MUST have a "coordinates" field with exactly 3 numeric values [x, y, z].
8. Rows without valid coordinates should be completely excluded from the points array.
9. The top-level JSON must match the provided schema for the `parse_analyses` function, which includes fields: "name", "description", and "points".

Input table:
{table_text}
"""
    
    # Send to API
    response = client.parse_analyses(detailed_prompt, model=model)
    
    # Convert to dictionary for serialization
    parsed_json = response.model_dump()
    
    return {
        "file_name": file_name,
        "parsed_json": parsed_json
    }


def load_tables_info(tables_csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load tables_with_coordinates.csv if it exists.
    
    Args:
        tables_csv_path: Path to the tables CSV file
        
    Returns:
        Dictionary mapping table data files to their metadata
    """
    tables_info = {}
    if os.path.exists(tables_csv_path):
        with open(tables_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use the table_data_file as the key
                table_data_file = row["table_data_file"]
                tables_info[table_data_file] = row
    return tables_info


def parse_tables(
    input_folder: str = "./tables",
    output_folder: str = "./parsed_json",
    model: str = "gpt-4o-mini",
    num_workers: int = 1
) -> Dict[str, Any]:
    """
    Parse tables from CSV files using OpenAI API with parallel processing.
    
    Args:
        input_folder: Path to folder with CSV files
        output_folder: Path to output folder for JSON files
        model: Model to use for parsing
        num_workers: Number of parallel workers (default: 1 for serial)
        
    Returns:
        Dictionary containing all parsed results
    """
    # Initialize client
    client = CoordinateParsingClient()
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    
    # Load tables_with_coordinates.csv if it exists in the parent directory
    tables_csv_path = os.path.join(os.path.dirname(input_folder), "tables_with_coordinates.csv")
    tables_info = load_tables_info(tables_csv_path)
    
    # Get all CSV files to process
    csv_files = list(glob.glob(os.path.join(input_folder, "pmcid*.csv")))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_folder}")
        return results
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process files with or without parallelization
    if num_workers <= 1 or len(csv_files) <= 1:
        # Serial processing
        logger.info("Using serial processing")
        parsed_results = []
        for file_path in tqdm(csv_files):
            # Read raw CSV as text
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                table_text = "\n".join([",".join(r) for r in rows])
            
            # Get table caption and foot if available
            relative_file_path = os.path.relpath(file_path, input_folder)
            table_caption = ""
            table_foot = ""
            if relative_file_path in tables_info:
                table_info = tables_info[relative_file_path]
                table_caption = table_info.get("table_caption", "")
                table_foot = table_info.get("table_foot", "")
            
            # Get file name
            file_name = os.path.basename(file_path)
            
            # Parse the table
            result = parse_single_table(
                file_name, table_caption, table_foot, table_text, client, model
            )
            parsed_results.append(result)
    else:
        # Parallel processing
        logger.info(f"Using {num_workers} workers for parallel processing")
        
        def process_file(file_path):
            # Read raw CSV as text
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                table_text = "\n".join([",".join(r) for r in rows])
            
            # Get table caption and foot if available
            relative_file_path = os.path.relpath(file_path, input_folder)
            table_caption = ""
            table_foot = ""
            if relative_file_path in tables_info:
                table_info = tables_info[relative_file_path]
                table_caption = table_info.get("table_caption", "")
                table_foot = table_info.get("table_foot", "")
            
            # Get file name
            file_name = os.path.basename(file_path)
            
            # Parse the table
            return parse_single_table(
                file_name, table_caption, table_foot, table_text, client, model
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_file, file_path)
                for file_path in csv_files
            ]
            parsed_results = [
                future.result()
                for future in tqdm(futures, total=len(futures))
            ]
    
    # Process results
    for result in parsed_results:
        file_name = result["file_name"]
        parsed_json = result["parsed_json"]
        
        # Store results
        results[file_name] = parsed_json
        
        # Save one JSON file per CSV
        output_path = os.path.join(output_folder, file_name.replace(".csv", ".json"))
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(parsed_json, out_f, indent=2)
    
    # Save all results into one combined file
    with open(os.path.join(output_folder, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Parsing complete. Results saved in: {output_folder}")
    return results