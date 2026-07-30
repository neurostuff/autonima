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
from .prompts import create_coordinate_parsing_prompt
from .schema import ParseAnalysesOutput

logger = logging.getLogger(__name__)


def parse_single_table(
    table_id: str,
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
    detailed_prompt = create_coordinate_parsing_prompt(
        table_text,
        table_caption=table_caption,
        table_foot=table_foot,
    )

    # Send to API
    response = client.parse_analyses(detailed_prompt, model=model)
    
    # Convert to dictionary for serialization
    parsed_json = response.model_dump()
    
    return {
        "table_id": table_id,
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
    
    # Create table metadata mapping
    table_meta_map = {}
    for file_path in csv_files:
        rel_path = os.path.relpath(file_path, input_folder)
        if rel_path in tables_info:
            info = tables_info[rel_path]
            table_id = os.path.splitext(rel_path)[0]
            table_meta_map[table_id] = {
                "caption": info.get("table_caption", ""),
                "footer": info.get("table_foot", "")
            }
    
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
            
            # Get table_id and metadata
            rel_path = os.path.relpath(file_path, input_folder)
            table_id = os.path.splitext(rel_path)[0]
            meta = table_meta_map.get(table_id, {})
            
            result = parse_single_table(
                table_id,
                meta.get("caption", ""),
                meta.get("footer", ""),
                table_text,
                client,
                model
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
            
            # Get table_id and metadata
            rel_path = os.path.relpath(file_path, input_folder)
            table_id = os.path.splitext(rel_path)[0]
            meta = table_meta_map.get(table_id, {})
            
            return parse_single_table(
                table_id,
                meta.get("caption", ""),
                meta.get("footer", ""),
                table_text,
                client,
                model
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
        table_id = result["table_id"]
        parsed_json = result["parsed_json"]
        
        # Store results
        results[table_id] = parsed_json
        
        # Save one JSON file per table
        output_path = os.path.join(
            output_folder,
            f"{table_id}.json"
        )
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(parsed_json, out_f, indent=2)
    
    # Save all results into one combined file
    with open(os.path.join(output_folder, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Parsing complete. Results saved in: {output_folder}")
    return results
