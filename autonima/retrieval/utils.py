"""Utility functions for retrieval modules."""

import pandas as pd
import json
import logging
import csv
import hashlib
import os
import shutil
import tempfile
import re
from pathlib import Path
from typing import Optional, Union, List, Set, Dict, Any
from ..models.types import Study, ActivationTable
from bs4 import BeautifulSoup, Comment


class ACEProcessingError(RuntimeError):
    """Raised when a configured local HTML source cannot be processed by ACE."""

# Try to import readabilipy for enhanced HTML cleaning
try:
    from readabilipy import simple_json_from_html_string
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    logging.warning("readabilipy not installed. Install with 'pip install readabilipy' for enhanced HTML cleaning. "
                     "Note: Node.js is also required for readabilipy to work.")


def _load_full_text(study: Study,  output_dir: str = None) -> Optional[str]:
    """
    Load the full text content for a study from a CSV file or a direct text file.
    
    Args:
        study: The study object containing the pmcid
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
        
        # If output_dir is not provided, cannot proceed
        if not output_dir:
            raise ValueError("output_dir must be provided if full_text_path is not set")

        # Prefer the current standard path but keep backward compatibility.
        candidate_paths = [
            Path(output_dir) / "retrieval" / "pubget_data" / "text.csv",
            Path(output_dir) / "pubget_data" / "text.csv",
        ]
        text_file = next((p for p in candidate_paths if p.exists()), None)
        if text_file is None:
            raise FileNotFoundError(
                "Text file not found at "
                f"{candidate_paths[0]} or {candidate_paths[1]}"
            )
            
        # Read the CSV file
        df = pd.read_csv(text_file)
        
        # Look for the row matching the study's pmcid
        if study.pmcid:
            row = df[df['pmcid'].astype(str) == str(study.pmcid)]
            if not row.empty:
                return row.iloc[0]['body']
        
    except Exception:
        # Handle any errors during file reading or processing
        raise

    # If no matching pmcid found or no pmcid provided
    raise ValueError(f"No full text found for study with pmcid {study.pmcid}")


def _load_ace_api():
    try:
        import ace
        from ace.ingest import extract_and_export
        from ace.tableparser import (
            GATING_VERSION,
            classify_coordinate_table,
        )
    except ImportError as exc:
        raise ACEProcessingError(
            "ACE is required to process local HTML coordinate tables. "
            "Install a compatible ACE package or provide a complete "
            "processed_data_path."
        ) from exc
    return ace, extract_and_export, GATING_VERSION, classify_coordinate_table


def _table_rows_from_path(path: Path) -> List[List[str]]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    return list(csv.reader(text.splitlines(), dialect=dialect))


def _is_coordinate_candidate_path(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    _, _, _, classify_coordinate_table = _load_ace_api()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return classify_coordinate_table(
            rows=_table_rows_from_path(path)
        ).candidate
    if suffix in {".html", ".htm", ".xml"}:
        return classify_coordinate_table(
            html=path.read_text(encoding="utf-8", errors="replace")
        ).candidate
    return False


def _normalized_table_id(table_id: Any) -> str:
    value = str(table_id or "").strip().lower()
    return re.sub(r"^\d+[_-]+(?=(?:t|tbl)\d)", "", value)


def _normalized_table_content(
    *,
    path: Optional[Path] = None,
    raw_table: Optional[str] = None,
) -> str:
    rows: List[List[str]] = []
    if path is not None and path.exists():
        if path.suffix.lower() in {".csv", ".tsv"}:
            rows = _table_rows_from_path(path)
        elif path.suffix.lower() in {".html", ".htm", ".xml"}:
            raw_table = path.read_text(encoding="utf-8", errors="replace")
    if raw_table:
        soup = BeautifulSoup(raw_table, "lxml")
        rows = [
            [cell.get_text(" ", strip=True) for cell in row.find_all(
                ["th", "td", "entry"],
                recursive=True,
            )]
            for row in soup.find_all(["tr", "row"])
        ]
    normalized_rows = []
    for row in rows:
        normalized = [
            re.sub(r"\s+", " ", str(cell).replace("−", "-")).strip().lower()
            for cell in row
        ]
        if any(normalized):
            normalized_rows.append("\x1f".join(normalized))
    return "\x1e".join(normalized_rows)


def _ace_export_complete(processed_path: Optional[Path]) -> bool:
    if processed_path is None:
        return False
    coordinates_file = processed_path / "coordinates.csv"
    tables_file = processed_path / "tables.csv"
    if not coordinates_file.is_file() or not tables_file.is_file():
        return False
    try:
        tables_df = pd.read_csv(tables_file)
    except Exception:
        return False
    if "table_raw_file" not in tables_df.columns:
        return False
    for raw_path in tables_df["table_raw_file"].dropna():
        if raw_path and not (processed_path / str(raw_path)).is_file():
            return False
    return True


def _ace_source_fingerprint(
    html_files: List[Path],
    pmids_to_include: Optional[Set[int]],
) -> Dict[str, Any]:
    ace, _, gate_version, _ = _load_ace_api()
    digest = hashlib.sha256()
    for path in html_files:
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "ace_version": getattr(ace, "__version__", "unknown"),
        "gate_version": gate_version,
        "pmids": sorted(int(pmid) for pmid in (pmids_to_include or set())),
        "input_sha256": digest.hexdigest(),
        "input_count": len(html_files),
    }


def _ensure_ace_export(
    root_path: Path,
    configured_processed_path: Optional[Path],
    generated_processed_path: Path,
    pmids_to_include: Optional[Set[int]],
    num_workers: int = 1,
) -> Path:
    """Return a complete ACE export, generating a run-managed one if needed."""
    if _ace_export_complete(configured_processed_path):
        return configured_processed_path

    generated_processed_path = generated_processed_path.resolve()
    excluded_roots = {
        path.resolve()
        for path in (configured_processed_path, generated_processed_path)
        if path is not None
    }
    html_files = []
    for path in root_path.rglob("*.html"):
        try:
            pmid = int(path.stem)
        except ValueError:
            continue
        if pmids_to_include is not None and pmid not in pmids_to_include:
            continue
        resolved = path.resolve()
        if any(
            resolved == excluded or excluded in resolved.parents
            for excluded in excluded_roots
        ):
            continue
        html_files.append(path)
    html_files.sort(key=lambda path: str(path.resolve()))

    if not html_files:
        raise ACEProcessingError(
            f"No PMID-named HTML files were found under {root_path} "
            "for the current retrieval scope."
        )

    fingerprint = _ace_source_fingerprint(html_files, pmids_to_include)
    manifest_path = generated_processed_path / "ace_manifest.json"
    if _ace_export_complete(generated_processed_path) and manifest_path.is_file():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == fingerprint:
                return generated_processed_path
        except (OSError, json.JSONDecodeError):
            pass

    generated_processed_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path = Path(tempfile.mkdtemp(
        prefix=f".{generated_processed_path.name}.",
        dir=str(generated_processed_path.parent),
    ))
    backup_path = generated_processed_path.with_name(
        f".{generated_processed_path.name}.previous"
    )
    try:
        _, extract_and_export, _, _ = _load_ace_api()
        extract_and_export(
            html_files,
            stage_path,
            pmid_filenames=True,
            skip_metadata=True,
            num_workers=max(1, int(num_workers)),
            use_readability=False,
        )
        if not _ace_export_complete(stage_path):
            raise ACEProcessingError(
                f"ACE produced an incomplete export in {stage_path}"
            )
        (stage_path / "ace_manifest.json").write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if backup_path.exists():
            shutil.rmtree(backup_path)
        if generated_processed_path.exists():
            os.replace(generated_processed_path, backup_path)
        os.replace(stage_path, generated_processed_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)
    except Exception:
        if (
            backup_path.exists()
            and not generated_processed_path.exists()
        ):
            os.replace(backup_path, generated_processed_path)
        raise
    finally:
        if stage_path.exists():
            shutil.rmtree(stage_path)

    return generated_processed_path


def _map_pmids_to_text(
    root_path: Union[str, Path],
    pmid_source: str,
    text_path_templates: Optional[List[str]] = None,
    coordinates_path_templates: Optional[List[str]] = None,
    pmids_to_include: Optional[Set[int]] = None,
    json_filename: str = 'identifiers.json',
    json_pmid_key: str = 'pmid',
    allowed_extensions: Optional[List[str]] = None,
    processed_data_path: Optional[str] = None,
    generated_processed_data_path: Optional[Union[str, Path]] = None,
    ace_num_workers: int = 1,
    source_name: Optional[str] = None,
    name: Optional[str] = None,
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
        coordinates_path_templates (Optional[List[str]]): A list of relative path templates to search for
                                                          coordinate files, in order of preference.
                                                          Typically used to locate files like 'coordinates.json', 
                                                          and instead of processed data path.
        pmids_to_include (Optional[Set[int]]): An optional set of PMIDs to filter for.
                                               If provided, only these PMIDs will be included.
        json_filename (str): The name of the JSON file to read when pmid_source is 'json'.
                             Defaults to 'identifiers.json'.
        json_pmid_key (str): The key in the JSON file that holds the PMID.
                             Defaults to 'pmid'.
        allowed_extensions (Optional[List[str]]): A list of file extensions (e.g., ['.txt', '.xml'])
                                                  to consider when pmid_source is 'file_name'.
                                                  Defaults to ['.txt'].
        processed_data_path (Optional[str]): The path to the processed data directory.

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

    processed_coordinate_paths = {}
    for item in iterator:
        pmid = None
        text_file_path = None

        def _find_template_file(item_path: Path, templates: List[str]) -> Optional[Path]:
            """Find the best full text file using the provided templates."""
            for template in templates:
                candidate = item_path / template
                if candidate.exists():
                    return candidate
            return None

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
                text_file_path = _find_template_file(item, text_path_templates) 

        elif pmid_source == 'folder_name' and item.is_dir():
            try:
                pmid = int(item.name)
            except ValueError:
                continue # Folder name is not a valid integer PMID
            
            text_file_path = _find_template_file(item, text_path_templates)

        elif pmid_source == 'file_name' and item.is_file():
            if item.suffix in allowed_extensions:
                try:
                    pmid = int(item.stem)
                    text_file_path = item
                except ValueError:
                    continue # File stem is not a valid integer PMID

        if coordinates_path_templates:
            coordinates_file_path = _find_template_file(item, coordinates_path_templates)
            if coordinates_file_path and pmid is not None:
                processed_coordinate_paths[pmid] = coordinates_file_path

        # If we have a valid PMID and its text file, add it to the index
        if pmid and text_file_path:
            if pmids_to_include is None or pmid in pmids_to_include:
                index[pmid] = text_file_path

    is_html_source = (
        pmid_source == "file_name"
        and any(
            str(extension).lower() in {".html", ".htm"}
            for extension in (allowed_extensions or [])
        )
    )
    if is_html_source and not index:
        return index, {}, {}
    if is_html_source:
        configured_path = (
            Path(processed_data_path)
            if processed_data_path
            else None
        )
        if _ace_export_complete(configured_path):
            processed_data_path = configured_path
        elif generated_processed_data_path is None:
            raise ValueError(
                "generated_processed_data_path is required when processing "
                "a local HTML source without a complete ACE export"
            )
        else:
            try:
                processed_data_path = _ensure_ace_export(
                    root_path=root,
                    configured_processed_path=configured_path,
                    generated_processed_path=Path(generated_processed_data_path),
                    pmids_to_include=pmids_to_include,
                    num_workers=ace_num_workers,
                )
            except ACEProcessingError:
                raise
            except Exception as exc:
                raise ACEProcessingError(
                    f"ACE failed while processing HTML source {root}: {exc}"
                ) from exc
    elif processed_data_path:
        processed_data_path = Path(processed_data_path)

    analyses, tables = load_activation_table_map(
        processed_data_path=processed_data_path,
        processed_coordinate_paths=processed_coordinate_paths,
        ids_to_include=pmids_to_include,
        filter_by_coordinates=not is_html_source,
        identifier_key='pmid',
    )

    if coordinates_path_templates:
        _append_sibling_candidate_tables(
            tables=tables,
            text_paths=index,
            ids_to_include=pmids_to_include,
        )

    return index, analyses, tables


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
    ids_to_include: Optional[Set[str]] = None,
    identifier_key: str = "pmcid",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Turn a tables dataframe into a mapping of identifier -> list of table metadata.
    
    Args:
        df: DataFrame containing table metadata
        root_path: Root path for resolving file paths
        ids_to_include: Optional set of identifiers to include
        identifier_key: Column name to use as identifier (default: "pmcid")
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
    required_columns = [identifier_key, 'table_id', 'table_label', 'table_caption', 'table_foot']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    id_to_tables: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in df.iterrows():
        identifier = row[identifier_key]

        if ids_to_include is not None and identifier not in ids_to_include:
            continue

        # Resolve file paths if present
        table_raw_path = (
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
            'table_raw_path': table_raw_path,
            'table_data_path': table_data_path,
            'table_caption': (
                row['table_caption'] if pd.notna(row['table_caption']) else None
            ),
            'table_foot': (
                row['table_foot'] if pd.notna(row['table_foot']) else None
            ),
        }

        id_to_tables.setdefault(identifier, []).append(table_metadata)

    return id_to_tables


def _append_sibling_candidate_tables(
    tables: Dict[Any, List[Dict[str, Any]]],
    text_paths: Dict[int, Path],
    ids_to_include: Optional[Set[int]] = None,
) -> None:
    """Add coordinate-like sibling table files omitted by coordinates.json."""
    supported = {".csv", ".tsv", ".html", ".htm", ".xml"}
    for pmid, text_path in text_paths.items():
        if ids_to_include is not None and pmid not in ids_to_include:
            continue
        article_dir = text_path.parent
        table_dir = article_dir / "tables"
        if not table_dir.is_dir():
            continue

        existing = tables.setdefault(pmid, [])
        existing_ids = {
            _normalized_table_id(item.get("table_id"))
            for item in existing
        }
        existing_paths = {
            str(Path(path).resolve())
            for item in existing
            for path in (
                item.get("table_raw_path"),
                item.get("table_data_path"),
            )
            if path
        }
        existing_content = {
            signature
            for item in existing
            for signature in [_normalized_table_content(
                path=Path(
                    item.get("table_raw_path")
                    or item.get("table_data_path")
                )
                if (
                    item.get("table_raw_path")
                    or item.get("table_data_path")
                )
                else None,
                raw_table=item.get("raw_table"),
            )]
            if signature
        }
        for candidate_path in sorted(table_dir.rglob("*")):
            if (
                not candidate_path.is_file()
                or candidate_path.suffix.lower() not in supported
            ):
                continue
            table_id = candidate_path.stem
            normalized_id = _normalized_table_id(table_id)
            resolved = str(candidate_path.resolve())
            normalized_content = _normalized_table_content(
                path=candidate_path,
            )
            if (
                normalized_id in existing_ids
                or resolved in existing_paths
                or (
                    normalized_content
                    and normalized_content in existing_content
                )
            ):
                continue
            if not _is_coordinate_candidate_path(candidate_path):
                continue
            existing.append({
                "table_id": table_id,
                "table_label": table_id,
                "table_raw_path": (
                    resolved
                    if candidate_path.suffix.lower() in {".html", ".htm", ".xml"}
                    else None
                ),
                "table_data_path": (
                    resolved
                    if candidate_path.suffix.lower() in {".csv", ".tsv"}
                    else None
                ),
                "table_caption": None,
                "table_foot": None,
            })
            existing_ids.add(normalized_id)
            existing_paths.add(resolved)
            if normalized_content:
                existing_content.add(normalized_content)


def _load_analyses_from_coordinates_df(
    coords_df: pd.DataFrame,
    ids_to_include: Optional[Set[str]] = None,
    identifier_key: str = "pmcid",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load analyses from a coordinates dataframe.
    
    Args:
        coords_df: DataFrame containing coordinate data
        ids_to_include: Optional set of identifiers to include
        identifier_key: Column name to use as identifier (default: "pmcid")
        
    Returns:
        Mapping of identifier -> list of analysis metadata dicts
    """
    required_columns = [
        identifier_key,
        'table_id',
        'table_label',
        'x',
        'y',
        'z'
    ]
    missing = [c for c in required_columns if c not in coords_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in coordinates dataframe: {missing}")

    id_to_analyses: Dict[str, List[Dict[str, Any]]] = {}

    # Group by table_id
    grouped = coords_df.groupby([identifier_key, 'table_id'])
    for (identifier, table_id), group in grouped:
        first_row = group.iloc[0]

        if ids_to_include is not None and identifier not in ids_to_include:
            continue

        # Build list of coordinate points
        points = []
        for _, row in group.iterrows():
            # Build coordinate dict with x, y, z
            coordinates = [float(row['x']), float(row['y']), float(row['z'])]

            # Create point dict matching CoordinatePoint schema
            point = {
                'coordinates': coordinates,
                'space': None,  # Space not provided in coordinates.csv
            }
            points.append(point)

        # Create analysis metadata matching Analysis schema
        analysis_metadata = {
            'name': str(table_id),
            'description': str(first_row['table_label']) if pd.notna(first_row['table_label']) else None,
            'points': points,
            'parsed': False,  # IMPORTANT: Set to False as requested
            'table_id': str(table_id)
        }

        id_to_analyses.setdefault(identifier, []).append(analysis_metadata)

    return id_to_analyses


def load_activation_table_map(
    processed_data_path: Path,
    processed_coordinate_paths: Optional[Dict[int, Path]] = None,
    ids_to_include: Optional[Set[str]] = None,
    filter_by_coordinates: bool = True,
    identifier_key: str = "pmcid",
    fallback_candidate_gate: bool = False,
) -> tuple[Optional[Dict[str, List[Dict[str, Any]]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Core function: Load and (optionally) filter activation tables.
    Returns (analyses, tables) tuple where:
    - analyses: identifier -> list of analysis metadata dicts from coordinates
    - tables: identifier -> list of table metadata dicts from tables.csv

    Can provide either a processed data directory or a dict of pmids to coordinate file paths.
    
    Args:
        processed_data_dir: Directory containing tables.csv and coordinates.csv
        processed_coordinate_paths: Optional dict mapping PMIDs to coordinates.json file paths
        ids_to_include: Optional set of identifiers to include
        filter_by_coordinates: Whether to filter by coordinates
        identifier_key: Column name to use as identifier (default: "pmcid")
        
    Returns:
        Tuple of (analyses_dict, tables_dict)
    """
    if processed_data_path is not None:
        coords_file = processed_data_path / "coordinates.csv"
        coords_df = pd.read_csv(coords_file) if coords_file.exists() else None

        # Load Analyses from coordinates
        if coords_df is not None:
            analyses = _load_analyses_from_coordinates_df(
                coords_df=coords_df,
                ids_to_include=ids_to_include,
                identifier_key=identifier_key,
            )
        else:
            analyses = None

        tables_file = processed_data_path / "tables.csv"

        if not tables_file.exists():
            logging.info(f"No tables.csv in {processed_data_path}, skipping...")
            return analyses, {}

        try:
            df = pd.read_csv(tables_file)

            # Optional coordinate filtering, with a recall-oriented ACE fallback
            # for source tables rejected by the source's first-pass gate.
            if filter_by_coordinates and coords_df is not None:
                coordinate_index = coords_df.set_index(
                    [identifier_key, "table_id"]
                ).index
                selected_mask = df.set_index(
                    [identifier_key, "table_id"]
                ).index.isin(coordinate_index)
                if fallback_candidate_gate:
                    fallback_mask = []
                    for selected, (_, row) in zip(
                        selected_mask,
                        df.iterrows(),
                    ):
                        if selected:
                            fallback_mask.append(True)
                            continue
                        raw_file = row.get("table_raw_file")
                        data_file = row.get("table_data_file")
                        relative_path = (
                            raw_file if pd.notna(raw_file) and raw_file
                            else data_file
                        )
                        fallback_mask.append(bool(
                            relative_path
                            and _is_coordinate_candidate_path(
                                processed_data_path / str(relative_path)
                            )
                        ))
                    df = df[fallback_mask]
                else:
                    df = df[selected_mask]

            tables = _load_activation_table_metadata(
                df=df,
                root_path=processed_data_path,
                ids_to_include=ids_to_include,
                identifier_key=identifier_key,
            )
            return analyses, tables

        except Exception as e:
            logging.warning(f"Failed to load activation tables: {e}")
            return analyses, {}
        
    elif processed_coordinate_paths is not None:
        # Handle loading from coordinates.json files
        analyses = {}
        tables = {}
        
        for pmid, coord_file_path in processed_coordinate_paths.items():
            if ids_to_include is not None and pmid not in ids_to_include:
                continue
                
            try:
                with open(coord_file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract studies from the coordinates.json structure
                studies = data.get('studyset', {}).get('studies', [])
                
                for study in studies:
                    # Extract analyses from each study
                    study_analyses = study.get('analyses', [])
                    
                    for analysis in study_analyses:
                        # Convert points to the expected format
                        points = []
                        for point in analysis.get('points', []):
                            points.append({
                                'coordinates': point['coordinates'],
                                'space': point.get('space')
                            })
                        
                        # Create analysis metadata
                        metadata = analysis.get('metadata', {})
                        analysis_metadata = {
                            'name': analysis.get('name', ''),
                            'description': metadata.get('table_label'),
                            'points': points,
                            'parsed': False,
                            'table_id': metadata.get('table_id', '')
                        }
                        
                        # Add to analyses dict
                        if pmid not in analyses:
                            analyses[pmid] = []
                        analyses[pmid].append(analysis_metadata)
                        
                        # Create activation table metadata
                        metadata = analysis.get('metadata', {})
                        table_metadata = {
                            'table_id': metadata.get('table_id', ''),
                            'table_label': metadata.get('table_label', ''),
                            'raw_table': metadata.get('raw_table_xml', ''),
                            'table_caption': None,
                            'table_foot': None,
                            'table_data_path': None,
                            'table_raw_path': None
                        }
                        
                        # Add to tables dict
                        if pmid not in tables:
                            tables[pmid] = []
                        tables[pmid].append(table_metadata)
                        
            except Exception as e:
                logging.warning(f"Failed to load coordinates from {coord_file_path}: {e}")
                continue
        return analyses, tables

    return None, {}


def _apply_activation_tables_to_studies(
    studies: List["Study"],
    id_to_tables: Dict[str, List[Dict[str, Any]]],
    identifier_key: str,
    clear_existing: bool = True,
    identifier_type: str = None
) -> None:
    """
    Attach activation tables to studies based on identifier mappings.
    
    Args:
        studies: List of studies to attach tables to
        id_to_tables: Mapping of identifiers to table metadata
        clear_existing: Whether to clear existing activation tables
        identifier_key: Which study attribute to use as identifier
    """
    # If identifier_type is provided, convert keys of id_to_tables accordingly
    if identifier_type == "int":
        id_to_tables = {
            int(k): v for k, v in id_to_tables.items()
        }
    elif identifier_type == "str":
        id_to_tables = {
            str(k): v for k, v in id_to_tables.items()
        }
        
    for study in studies:
        # Get the identifier value from the study based on the identifier_key
        identifier_value = getattr(study, identifier_key, None)
        if not identifier_value:
            continue

        if identifier_value not in id_to_tables:
            continue

        if clear_existing:
            study.activation_tables.clear()

        for t in id_to_tables[identifier_value]:
            study.activation_tables.append(
                ActivationTable(
                    table_id=t['table_id'],
                    table_label=t['table_label'],
                    table_data_path=t.get('table_data_path', None),
                    table_raw_path=t.get('table_raw_path', None),
                    table_caption=t['table_caption'],
                    table_foot=t['table_foot'],
                    raw_table=t.get('raw_table', None)
                )
            )


def _apply_analyses_to_studies(
    studies: List["Study"],
    id_to_analyses: Dict[str, List[Dict[str, Any]]],
    identifier_key: str,
    clear_existing: bool = False,
    identifier_type: str = None
) -> None:
    """
    Attach analyses (from coordinates) to studies based on identifier mappings.
    
    Args:
        studies: List of studies to attach analyses to
        id_to_analyses: Mapping of identifiers to analysis metadata
        identifier_key: Which study attribute to use as identifier
        clear_existing: Whether to clear existing analyses
        identifier_type: Type conversion for identifier ('int' or 'str')
    """
    from ..coordinates.schema import Analysis, CoordinatePoint, PointsValue
    
    # If identifier_type is provided, convert keys accordingly
    if identifier_type == "int":
        id_to_analyses = {
            int(k): v for k, v in id_to_analyses.items()
        }
    elif identifier_type == "str":
        id_to_analyses = {
            str(k): v for k, v in id_to_analyses.items()
        }
        
    for study in studies:
        # Get the identifier value from the study
        identifier_value = getattr(study, identifier_key, None)
        if not identifier_value:
            continue

        if identifier_value not in id_to_analyses:
            continue

        if clear_existing:
            study.analyses.clear()

        # Convert analysis metadata dicts to Analysis objects
        for analysis_data in id_to_analyses[identifier_value]:
            # Convert points
            points = []
            for point_data in analysis_data.get('points', []):
                # Convert values if present
                values = None
                if point_data.get('values'):
                    values = [
                        PointsValue(
                            value=v.get('value'),
                            kind=v.get('kind')
                        )
                        for v in point_data['values']
                    ]
                
                points.append(CoordinatePoint(
                    coordinates=point_data['coordinates'],
                    space=point_data.get('space'),
                    values=values
                ))
            
            # Create Analysis object
            study.analyses.append(Analysis(
                name=analysis_data.get('name'),
                description=analysis_data.get('description'),
                points=points,
                parsed=analysis_data.get('parsed', False),
                table_id=analysis_data.get('table_id')
            ))
