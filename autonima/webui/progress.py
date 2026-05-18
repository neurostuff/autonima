"""Run progress aggregation utilities."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

STAGES = [
    "search",
    "abstract",
    "retrieval",
    "fulltext",
    "parsing",
    "annotation",
    "output",
]


def _safe_read_json(path: Path) -> Dict[str, Any] | List[Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_stage_from_logs(log_lines: List[str]) -> str | None:
    """Best-effort current stage from logs."""
    current_stage = None
    for line in log_lines:
        text = line.lower()
        if "starting autonima pipeline" in text:
            current_stage = "search"
        elif "starting abstract screening" in text:
            current_stage = "abstract"
        elif "retrieval: starting" in text:
            current_stage = "retrieval"
        elif "starting full-text screening" in text:
            current_stage = "fulltext"
        elif "starting coordinate parsing" in text:
            current_stage = "parsing"
        elif "processing" in text and "annotation" in text:
            current_stage = "annotation"
        elif "pipeline completed" in text:
            current_stage = "output"
    return current_stage


def extract_log_issues(log_lines: List[str], max_items: int = 25) -> Dict[str, Any]:
    """Extract warning/error log lines for dedicated UI display."""
    errors: List[str] = []
    warnings: List[str] = []
    seen_errors = set()
    seen_warnings = set()

    for raw_line in log_lines:
        line = str(raw_line or "").strip()
        if not line:
            continue
        text = line.lower()

        is_error = " - error - " in text or text.startswith("error:")
        is_warning = " - warning - " in text or text.startswith("warning:")

        if is_error and line not in seen_errors:
            seen_errors.add(line)
            errors.append(line)
            continue
        if is_warning and line not in seen_warnings:
            seen_warnings.add(line)
            warnings.append(line)

    if max_items > 0:
        errors = errors[-max_items:]
        warnings = warnings[-max_items:]

    return {
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def extract_live_progress(log_lines: List[str], current_stage: str | None) -> Dict[str, Any] | None:
    """Parse the latest tqdm-style progress sample from captured subprocess output."""
    stage = current_stage
    latest: Dict[str, Any] | None = None

    for raw_line in log_lines:
        line = str(raw_line or "").strip()
        if not line:
            continue

        inferred_stage = infer_stage_from_logs([line])
        if inferred_stage:
            stage = inferred_stage

        if "|" not in line:
            continue
        match = re.search(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)", line)
        if not match:
            continue

        total = int(match.group("total"))
        if total <= 0:
            continue
        current = min(int(match.group("current")), total)
        percent = round((current / total) * 100, 1)
        label = line.split("|", 1)[0].strip() or (stage or "running")
        latest = {
            "stage": stage,
            "label": label,
            "current": current,
            "total": total,
            "percent": percent,
        }

    return latest


def _is_parsing_enabled_from_config(config: Dict[str, Any]) -> bool:
    """Infer whether coordinate parsing is enabled from saved config."""
    if not isinstance(config, dict):
        return True

    parsing = config.get("parsing")
    if isinstance(parsing, dict) and isinstance(parsing.get("parse_coordinates"), bool):
        return bool(parsing.get("parse_coordinates"))

    # Backward-compat fallback for configs where this lived under retrieval.
    retrieval = config.get("retrieval")
    if isinstance(retrieval, dict) and isinstance(retrieval.get("parse_coordinates"), bool):
        return bool(retrieval.get("parse_coordinates"))

    return True


def _read_missing_fulltexts(outputs_dir: Path | None) -> Dict[str, Any]:
    missing_payload: Dict[str, Any] = {
        "available": False,
        "count": 0,
        "txt_path": None,
        "csv_path": None,
        "preview_pmids": [],
        "preview_rows": [],
    }
    if not outputs_dir or not outputs_dir.exists():
        return missing_payload

    txt_path = outputs_dir / "missing_fulltexts.txt"
    csv_path = outputs_dir / "missing_fulltexts.csv"

    pmids: List[str] = []
    if txt_path.exists() and txt_path.is_file():
        try:
            with txt_path.open("r", encoding="utf-8") as f:
                pmids = [line.strip() for line in f if line.strip()]
            missing_payload["txt_path"] = str(txt_path)
        except Exception:
            pmids = []

    preview_rows: List[Dict[str, Any]] = []
    csv_count = 0
    if csv_path.exists() and csv_path.is_file():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            csv_count = len(rows)
            preview_rows = rows[:5]
            missing_payload["csv_path"] = str(csv_path)
        except Exception:
            csv_count = 0
            preview_rows = []

    count = len(pmids) if pmids else csv_count
    if count > 0:
        missing_payload["available"] = True
        missing_payload["count"] = count
        missing_payload["preview_pmids"] = pmids[:10]
        missing_payload["preview_rows"] = preview_rows

    return missing_payload


def build_stage_status(
    run_status: str,
    output_folder: str | None,
    log_lines: List[str],
) -> Dict[str, Any]:
    """Build stage timeline and counters from outputs and logs."""
    stages = {stage: {"status": "pending"} for stage in STAGES}
    counters: Dict[str, Any] = {}
    parsing_enabled = True

    output_dir = Path(output_folder).expanduser().resolve() if output_folder else None
    outputs_dir = output_dir / "outputs" if output_dir else None
    nimads_studyset_path = outputs_dir / "nimads_studyset.json" if outputs_dir else None
    nimads_available = bool(nimads_studyset_path and nimads_studyset_path.exists())
    missing_fulltexts = _read_missing_fulltexts(outputs_dir)
    nimads_export_logged = any(
        "nimads export completed" in str(line or "").lower()
        for line in log_lines
    )

    if outputs_dir and outputs_dir.exists():
        search_data = _safe_read_json(outputs_dir / "search_results.json")
        if isinstance(search_data, dict):
            stages["search"]["status"] = "completed"
            studies = search_data.get("studies", [])
            counters["search"] = {
                "studies_found": len(studies) if isinstance(studies, list) else 0
            }

        abstract_data = _safe_read_json(outputs_dir / "abstract_screening_results.json")
        if isinstance(abstract_data, dict):
            stages["abstract"]["status"] = "completed"
            results = abstract_data.get("screening_results", [])
            included = 0
            excluded = 0
            incomplete = 0
            for item in results if isinstance(results, list) else []:
                decision = str(item.get("decision", "")).lower()
                if "included" in decision:
                    included += 1
                elif "incomplete" in decision:
                    incomplete += 1
                else:
                    excluded += 1
            counters["abstract"] = {
                "screened": len(results) if isinstance(results, list) else 0,
                "included": included,
                "excluded": excluded,
                "incomplete": incomplete,
            }

        retrieval_data = _safe_read_json(outputs_dir / "fulltext_retrieval_results.json")
        if isinstance(retrieval_data, dict):
            stages["retrieval"]["status"] = "completed"
            rows = retrieval_data.get("studies_with_fulltext", [])
            counters["retrieval"] = {
                "fulltext_candidates": len(rows) if isinstance(rows, list) else 0
            }

        fulltext_data = _safe_read_json(outputs_dir / "fulltext_screening_results.json")
        if isinstance(fulltext_data, dict):
            stages["fulltext"]["status"] = "completed"
            results = fulltext_data.get("screening_results", [])
            included = 0
            excluded = 0
            incomplete = 0
            for item in results if isinstance(results, list) else []:
                decision = str(item.get("decision", "")).lower()
                if "included" in decision:
                    included += 1
                elif "incomplete" in decision:
                    incomplete += 1
                else:
                    excluded += 1
            counters["fulltext"] = {
                "screened": len(results) if isinstance(results, list) else 0,
                "included": included,
                "excluded": excluded,
                "incomplete": incomplete,
            }

        parsing_data = _safe_read_json(outputs_dir / "coordinate_parsing_results.json")
        if parsing_data is not None:
            stages["parsing"]["status"] = "completed"

        annotation_data = _safe_read_json(outputs_dir / "annotation_results.json")
        if isinstance(annotation_data, list):
            stages["annotation"]["status"] = "completed"
            counters["annotation"] = {"decisions": len(annotation_data)}

        final_data = _safe_read_json(outputs_dir / "final_results.json")
        if isinstance(final_data, dict):
            parsing_enabled = _is_parsing_enabled_from_config(final_data.get("config", {}))
            stages["output"]["status"] = "completed"
            execution_stats = final_data.get("execution_stats", {})
            prisma = execution_stats.get("prisma_stats", {})
            if prisma:
                counters["output"] = prisma

            if stages["parsing"]["status"] != "completed":
                if not parsing_enabled:
                    stages["parsing"]["status"] = "completed"
                elif run_status == "completed":
                    # The run reached final outputs; parsing did not produce a results file
                    # (e.g., no parseable tables). Treat this stage as complete rather than
                    # leaving a misleading pending state.
                    stages["parsing"]["status"] = "completed"

    current_stage = infer_stage_from_logs(log_lines)
    live_progress = (
        extract_live_progress(log_lines, current_stage)
        if run_status in {"running", "canceling"}
        else None
    )

    if run_status in {"running", "canceling"}:
        for stage in STAGES:
            if stages[stage]["status"] == "completed":
                continue
            if current_stage == stage:
                stages[stage]["status"] = "running"
                break
            if current_stage is None:
                stages[stage]["status"] = "running"
                break

    if run_status == "failed":
        for stage in STAGES:
            if stages[stage]["status"] == "running":
                stages[stage]["status"] = "failed"

    if run_status == "canceled":
        for stage in STAGES:
            if stages[stage]["status"] == "running":
                stages[stage]["status"] = "canceled"

    return {
        "timeline": [{"stage": stage, **stages[stage]} for stage in STAGES],
        "counters": counters,
        "current_stage": current_stage,
        "live_progress": live_progress,
        "log_issues": extract_log_issues(log_lines),
        "nimads_available": nimads_available,
        "nimads_export_logged": nimads_export_logged,
        "nimads_studyset_path": str(nimads_studyset_path) if nimads_available else None,
        "missing_fulltexts": missing_fulltexts,
    }
