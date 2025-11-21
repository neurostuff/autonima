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
    # Create detailed prompt
    detailed_prompt = f"""
    You are a neuroimaging data curation assistant.

    You will receive a CSV table extracted from a published fMRI/neuroimaging article.
    The table reports statistical activation results, usually organized by *analysis* or *contrast*
    (e.g., "Athletes: motor imagery", "Non-athletes: motor imagery"). Each analysis may contain multiple rows of
    activation foci, with region names, MNI/TAL coordinates, and statistics.

    Table Caption: {table_caption}
    Table Foot: {table_foot}

    Your task is to output JSON strictly matching the schema of the `parse_analyses` function:

    {{
    "analyses": [
        {{
        "name": <string or null>,
        "description": <string or null>,
        "points": [
            {{
            "coordinates": [x, y, z],
            "space": <"MNI" | "TAL" | null>
            "values": [
                {{
                "value": <float or string or null>,
                "kind": <string or null>
                }},
                ...
            ]  # Omit this field if no statistical values are available
            }},
            ...
        ]
        }}
    ]
    }}

    ⚠️ CRITICAL RULES for coordinates:
    - Coordinates **must come ONLY from the X, Y, Z columns** (or an equivalent labeled "MNI coordinates").
    - Do NOT use any values from other numeric columns (e.g., Cluster, Volume, Brodmann area, ALE, T, Z).
    - If a row does not contain all three values under X, Y, Z → exclude that row.
    - Coordinates must be exactly three numeric values, extracted in order: [X, Y, Z].

    Other rules:
    1. **Analyses/contrasts**  
    - Start a new analysis whenever a distinct label is present (e.g., "Athletes: motor imagery").  
    - If no explicit contrasts, treat the whole table as a single analysis.  
    - Use only names that explicitly appear in the provided table, caption, or footnotes. Never invent.  

    2. **Space**  
    - If the table mentions MNI or Talairach, set `"space"` accordingly.  
    - If unclear, use `"space": null`.  

    3. **Values**
    - If the table has statistical values (e.g., T, Z), include them in `"values"`
    - For the `"kind"` field, you MUST use ONLY these exact values:
      * "z-statistic" for Z-scores
      * "t-statistic" for T-values
      * "f-statistic" for F-values
      * "p-value" for p-values (including FDR-corrected)
      * "beta" for beta coefficients
      * "correlation" for correlation coefficients
      * "other" for any other statistical measures
    - If no statistical columns, omit the `"values"` field entirely
    - Do NOT include values from non-statistical columns (e.g., Cluster, Volume, Brodmann area, ALE).
    - Each value must correspond to the same row as its X, Y, Z coordinates

    4. **Filtering**  
    - Ignore all other columns (cluster size, Brodmann area, ALE, etc.).  
    - Only extract X, Y, Z → nothing else.  

    5. **Null handling**  
    - Missing analysis names → `"name": null`.  
    - No valid coordinates in an analysis → keep `"points": []`.  

    6. **Consistency**  
    - Ensure coordinates are always `[float, float, float]`.  
    - Do not include fields outside the schema.  
    - Do not fabricate analysis names from prompt examples.  

    ---

    Example clarification:

    If a row looks like this:  
    ```

    Cluster,Volume,Brain regions,Hemisphere,Brodmann area,X,Y,Z,ALE
    2,864,Precentral Gyrus,L,6,-24,-12,54,1.78

    ```

    ✅ Correct coordinates = `[-24, -12, 54]`  
    ❌ Do NOT use `6,-24,-12` (the "6" is Brodmann area, not a coordinate).  
    ❌ Do NOT use ALE (1.78).  

    ---

    Now apply these rules to the following table:

    {table_text}
    """

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