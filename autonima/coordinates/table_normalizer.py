"""Canonical table normalization across Pubget XML and ACE HTML sources."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from .boundary_rules import build_boundary_markers, normalize_text

logger = logging.getLogger(__name__)


def _clean_cell_text(text: str) -> str:
    cleaned = (
        text.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
        .replace("\u2009", " ")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = _clean_cell_text(value)
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_axis_header(value: str, axis: str) -> bool:
    text = normalize_text(value)
    return bool(re.search(rf"(^|[^a-z]){axis}([^a-z]|$)", text))


def _looks_like_section_label(value: str) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    section_markers = (
        "main effect",
        "interaction",
        "correlation",
        " vs ",
        " versus ",
        " > ",
        " < ",
    )
    return any(marker in text for marker in section_markers)


def _extract_pubget_table_html_and_source(
    table_xml_path: str,
    table_id: str,
) -> Tuple[str, str]:
    xml_path = Path(table_xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Pubget tables.xml not found: {table_xml_path}")

    xml_content = xml_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(xml_content, "xml")

    target_table = None
    for extracted in soup.find_all("extracted-table"):
        table_id_tag = extracted.find("table-id")
        if table_id_tag and table_id_tag.get_text(strip=True) == table_id:
            target_table = extracted
            break

    if target_table is None:
        raise ValueError(
            f"Table id '{table_id}' not found in Pubget XML: {table_xml_path}"
        )

    original = target_table.find("original-table")
    if original is not None:
        table_tag = original.find("table")
        if table_tag is not None:
            return str(table_tag), "pubget_xml_original"

    transformed = target_table.find("transformed-table")
    if transformed is not None:
        table_tag = transformed.find("table")
        if table_tag is not None:
            return str(table_tag), "pubget_xml_transformed"

    raise ValueError(
        f"No parseable table found for table id '{table_id}' in {table_xml_path}"
    )


def parse_pubget_tables_xml(table_xml_path: str, table_id: str) -> str:
    """Extract one table fragment from a Pubget tables.xml file."""
    table_html, _ = _extract_pubget_table_html_and_source(table_xml_path, table_id)
    return table_html


def parse_ace_table_html(table_html_path: str) -> str:
    """Load an ACE table HTML file and return the first <table> fragment."""
    html_path = Path(table_html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"ACE table HTML not found: {table_html_path}")

    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    table_tag = soup.find("table")
    if table_tag is None:
        raise ValueError(f"No <table> tag found in ACE HTML: {table_html_path}")
    return str(table_tag)


def _collect_rows(table_tag: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    thead_sections = table_tag.find_all("thead", recursive=False)
    for section in thead_sections:
        trs = section.find_all("tr", recursive=False) or section.find_all("tr")
        for tr in trs:
            rows.append({"row_type": "header", "tr": tr})

    body_sections = (
        table_tag.find_all("tbody", recursive=False)
        + table_tag.find_all("tfoot", recursive=False)
    )
    for section in body_sections:
        trs = section.find_all("tr", recursive=False) or section.find_all("tr")
        for tr in trs:
            rows.append({"row_type": "body", "tr": tr})

    if rows:
        return rows

    # Fallback when table has no explicit thead/tbody.
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        row_type = "header" if any(cell.name == "th" for cell in cells) else "body"
        rows.append({"row_type": row_type, "tr": tr})
    return rows


def expand_table_grid(html_fragment: str) -> Dict[str, Any]:
    """Expand rowspan/colspan into a rectangular cell grid."""
    soup = BeautifulSoup(html_fragment, "lxml")
    table_tag = soup.find("table")
    if table_tag is None:
        raise ValueError("No <table> element found in table fragment")

    row_specs = _collect_rows(table_tag)
    span_carry: Dict[int, Dict[str, Any]] = {}
    expanded_rows: List[Dict[str, Any]] = []

    for row_index, row_spec in enumerate(row_specs):
        tr = row_spec["tr"]
        row_type = row_spec["row_type"]
        row_cells: List[Dict[str, Any]] = []
        col_idx = 0

        def consume_carry() -> None:
            nonlocal col_idx
            while col_idx in span_carry:
                carried = span_carry[col_idx]
                row_cells.append(
                    {
                        "text": carried["text"],
                        "is_header": carried["is_header"],
                        "from_span": True,
                    }
                )
                carried["rows_left"] -= 1
                if carried["rows_left"] <= 0:
                    del span_carry[col_idx]
                col_idx += 1

        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        for cell in cells:
            consume_carry()
            text = _clean_cell_text(cell.get_text(" ", strip=True))
            colspan = max(int(cell.get("colspan", 1) or 1), 1)
            rowspan = max(int(cell.get("rowspan", 1) or 1), 1)
            is_header = cell.name == "th" or row_type == "header"

            for _ in range(colspan):
                row_cells.append(
                    {
                        "text": text,
                        "is_header": is_header,
                        "from_span": False,
                    }
                )
                if rowspan > 1:
                    span_carry[col_idx] = {
                        "text": text,
                        "is_header": is_header,
                        "rows_left": rowspan - 1,
                    }
                col_idx += 1

        consume_carry()
        expanded_rows.append(
            {
                "row_index": row_index,
                "row_type": row_type,
                "cells": row_cells,
            }
        )

    max_cols = max((len(row["cells"]) for row in expanded_rows), default=0)
    for row in expanded_rows:
        if len(row["cells"]) < max_cols:
            row["cells"].extend(
                [{"text": "", "is_header": False, "from_span": False}]
                * (max_cols - len(row["cells"]))
            )

    grid = [[cell["text"] for cell in row["cells"]] for row in expanded_rows]
    header_row_indices = [
        row["row_index"] for row in expanded_rows if row["row_type"] == "header"
    ]

    return {
        "grid": grid,
        "rows_expanded": expanded_rows,
        "header_row_indices": header_row_indices,
        "column_count": max_cols,
    }


def _build_header_hierarchy(grid_data: Dict[str, Any]) -> List[List[str]]:
    grid = grid_data["grid"]
    header_rows = grid_data["header_row_indices"]
    col_count = grid_data["column_count"]

    hierarchy: List[List[str]] = []
    for col_idx in range(col_count):
        path: List[str] = []
        for row_idx in header_rows:
            value = _clean_cell_text(grid[row_idx][col_idx])
            if value and (not path or path[-1] != value):
                path.append(value)
        hierarchy.append(path)
    return hierarchy


def _detect_coordinate_columns(header_hierarchy: List[List[str]]) -> Dict[str, Any]:
    def header_text(col_idx: int) -> str:
        return " ".join(header_hierarchy[col_idx])

    x_col = y_col = z_col = None
    for col_idx in range(len(header_hierarchy)):
        text = header_text(col_idx)
        if x_col is None and _is_axis_header(text, "x"):
            x_col = col_idx
            continue
        if y_col is None and _is_axis_header(text, "y"):
            y_col = col_idx
            continue
        if z_col is None and _is_axis_header(text, "z"):
            z_col = col_idx

    stat_cols: List[Dict[str, Any]] = []
    stat_keywords = (
        ("t-statistic", (" t", "t-value", "t statistic", "statistic")),
        ("z-statistic", (" z", "z-value", "z statistic")),
        ("f-statistic", (" f", "f-value", "f statistic")),
        ("p-value", (" p", "p-value", "p value")),
        ("beta", ("beta",)),
        ("correlation", ("correlation", "corr")),
    )
    for col_idx in range(len(header_hierarchy)):
        text = normalize_text(header_text(col_idx))
        for kind, keywords in stat_keywords:
            if any(keyword in text for keyword in keywords):
                stat_cols.append(
                    {"column_index": col_idx, "kind": kind, "header": header_text(col_idx)}
                )
                break

    return {
        "x_col": x_col,
        "y_col": y_col,
        "z_col": z_col,
        "has_xyz": x_col is not None and y_col is not None and z_col is not None,
        "stat_columns": stat_cols,
    }


def _detect_space_hints(
    table_caption: Optional[str],
    table_foot: Optional[str],
    header_hierarchy: List[List[str]],
) -> List[str]:
    text_blob = " ".join(
        [
            table_caption or "",
            table_foot or "",
            " ".join(" ".join(path) for path in header_hierarchy),
        ]
    )
    text = normalize_text(text_blob)
    spaces: List[str] = []
    if "mni" in text:
        spaces.append("MNI")
    if "talairach" in text or re.search(r"\btal\b", text):
        spaces.append("TAL")
    return spaces


def normalize_table(activation_table: Any) -> Dict[str, Any]:
    """Normalize table input to a canonical source-agnostic JSON structure."""
    source_type = activation_table.table_source_type or ""
    source_parser = "csv_fallback"
    table_html: Optional[str] = None

    if source_type == "pubget_xml" and activation_table.table_xml_path:
        table_html, source_parser = _extract_pubget_table_html_and_source(
            activation_table.table_xml_path,
            activation_table.table_id,
        )
    elif source_type == "ace_html":
        html_path = activation_table.table_html_path or activation_table.table_raw_path
        if html_path:
            table_html = parse_ace_table_html(html_path)
            source_parser = "ace_html"
    elif activation_table.raw_table:
        table_html = activation_table.raw_table

    if table_html is None:
        activation_table.load_raw_table()
        if activation_table.raw_table is None:
            raise ValueError(
                f"No readable table content for table_id={activation_table.table_id}"
            )
        table_html = activation_table.raw_table

    grid_data = expand_table_grid(table_html)
    header_hierarchy = _build_header_hierarchy(grid_data)
    coord_detection = _detect_coordinate_columns(header_hierarchy)
    space_hints = _detect_space_hints(
        activation_table.table_caption,
        activation_table.table_foot,
        header_hierarchy,
    )

    rows: List[Dict[str, Any]] = []
    current_section: Optional[str] = None

    for row in grid_data["rows_expanded"]:
        row_index = row["row_index"]
        row_type = row["row_type"]
        cell_texts = [_clean_cell_text(cell["text"]) for cell in row["cells"]]
        non_empty = [(idx, value) for idx, value in enumerate(cell_texts) if value]
        primary_label = non_empty[0][1] if non_empty else None

        x_col = coord_detection["x_col"]
        y_col = coord_detection["y_col"]
        z_col = coord_detection["z_col"]
        x_val = _parse_float(cell_texts[x_col]) if x_col is not None else None
        y_val = _parse_float(cell_texts[y_col]) if y_col is not None else None
        z_val = _parse_float(cell_texts[z_col]) if z_col is not None else None
        has_coordinates = (
            x_col is not None
            and y_col is not None
            and z_col is not None
            and x_val is not None
            and y_val is not None
            and z_val is not None
        )

        section_label = None
        if row_type != "header" and not has_coordinates:
            if len(non_empty) == 1:
                section_label = non_empty[0][1]
            elif primary_label and _looks_like_section_label(primary_label):
                section_label = primary_label
        if section_label:
            current_section = section_label

        stat_values: List[Dict[str, Any]] = []
        if has_coordinates:
            for stat_col in coord_detection["stat_columns"]:
                col_idx = stat_col["column_index"]
                if col_idx < len(cell_texts):
                    raw = cell_texts[col_idx]
                    if raw:
                        parsed = _parse_float(raw)
                        stat_values.append(
                            {
                                "column_index": col_idx,
                                "kind": stat_col["kind"],
                                "raw": raw,
                                "value": parsed,
                            }
                        )

        rows.append(
            {
                "row_index": row_index,
                "row_type": row_type,
                "section_label": current_section,
                "primary_label": primary_label,
                "cell_text_by_col": cell_texts,
                "header_path_by_col": header_hierarchy,
                "coordinate_values": {"x": x_val, "y": y_val, "z": z_val},
                "stat_values": stat_values,
                "has_coordinates": has_coordinates,
            }
        )

    boundary_markers = build_boundary_markers(rows)

    canonical = {
        "table_identity": {
            "table_id": activation_table.table_id,
            "table_label": activation_table.table_label,
            "table_caption": activation_table.table_caption,
            "table_foot": activation_table.table_foot,
            "table_source_type": source_type or "csv_fallback",
        },
        "source_parser": source_parser,
        "grid": grid_data["grid"],
        "header_hierarchy": header_hierarchy,
        "rows": rows,
        "coordinate_column_detection": coord_detection,
        "space_hints": space_hints,
        "boundary_markers": boundary_markers,
    }
    return canonical

