"""Execution provenance and cache-signature helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import yaml

from .cache_versions import (
    ANNOTATION_PROMPT_VERSION,
    ABSTRACT_SCREENING_PROMPT_VERSION,
    CACHE_SCHEMA_VERSION,
    FULLTEXT_SCREENING_PROMPT_VERSION,
)
from .coordinates.prompts import COORDINATE_PARSING_PROMPT_VERSION
from .llm import usage as llm_usage

logger = logging.getLogger(__name__)

CACHE_POLICIES = {"auto", "ignore"}
CLEAR_CACHE_STAGES = {
    "search",
    "abstract",
    "retrieval",
    "fulltext",
    "parsing",
    "annotation",
    "output",
    "all",
}

STAGE_ARTIFACTS: Dict[str, List[str]] = {
    "search": ["search_results.json"],
    "abstract": ["abstract_screening_results.json"],
    "retrieval": ["fulltext_retrieval_results.json"],
    "fulltext": ["fulltext_screening_results.json"],
    "parsing": ["coordinate_parsing_results.json"],
    "annotation": ["annotation_results.json"],
    "output": [
        "final_results.json",
        "nimads_studyset.json",
        "nimads_annotation.json",
        "missing_fulltexts.csv",
        "missing_fulltexts.txt",
    ],
}

RUN_STAGE_LIMITS: Dict[str, str] = {
    "search": "search",
    "abstract": "abstract",
    "full": "output",
}

EXECUTION_PROGRESS_STATUSES = {
    "pending",
    "running",
    "completed",
    "skipped",
    "failed",
}
EXECUTION_PROGRESS_SOURCES = {
    "fresh",
    "cache",
    "mixed",
    "not_applicable",
    "unknown",
}


class UnsupportedCacheError(ValueError):
    """Raised when cache artifacts cannot be verified safely."""


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _to_plain(value: Any) -> Any:
    """Convert dataclasses/Pydantic models to JSON-compatible primitives."""
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _strip_noise(value: Any) -> Any:
    """Remove volatile fields that should not affect semantic hashes."""
    if isinstance(value, dict):
        return {
            k: _strip_noise(v)
            for k, v in sorted(value.items())
            if k
            not in {
                "timestamp",
                "started_at",
                "completed_at",
                "retrieved_at",
                "screened_at",
                "criteria_mapping",
            }
        }
    if isinstance(value, list):
        return [_strip_noise(v) for v in value]
    return value


def stable_hash(value: Any) -> str:
    """Hash canonical JSON for deterministic config/input signatures."""
    plain = _strip_noise(_to_plain(value))
    encoded = json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=4096)
def _file_content_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Hash a file; size/mtime make the process-local cache self-invalidating."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _pubget_text_hashes(
    path: str,
    size: int,
    mtime_ns: int,
) -> Dict[str, str]:
    """Read one PubGet text table once and index content hashes by PMCID."""
    hashes: Dict[str, str] = {}
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(sys.maxsize)
        with Path(path).open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as stream:
            for row in csv.DictReader(stream):
                pmcid = str(row.get("pmcid") or "").strip()
                if not pmcid:
                    continue
                body = str(row.get("body") or "")
                digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                hashes[pmcid] = digest
                hashes[pmcid.removeprefix("PMC")] = digest
    finally:
        csv.field_size_limit(previous_limit)
    return hashes


def study_full_text_content_hash(study: Any, output_dir: Any = None) -> Optional[str]:
    """Fingerprint the full-text input without running HTML readability tools.

    Raw-file changes conservatively invalidate the cache even when cleaned text
    would be unchanged. That is preferable to performing expensive prompt
    rendering merely to decide whether a cached result can be reused.
    """
    direct_path = getattr(study, "full_text_path", None)
    if direct_path:
        path = Path(str(direct_path)).expanduser()
        if path.is_file():
            stat = path.stat()
            return _file_content_sha256(
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
            )

    base = output_dir or getattr(study, "full_text_output_dir", None)
    if not base:
        return None
    base_path = Path(str(base)).expanduser()
    candidates = [
        base_path / "retrieval" / "pubget_data" / "text.csv",
        base_path / "pubget_data" / "text.csv",
    ]
    text_path = next((path for path in candidates if path.is_file()), None)
    if text_path is None:
        return None
    stat = text_path.stat()
    indexed = _pubget_text_hashes(
        str(text_path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
    )
    pmcid = str(getattr(study, "pmcid", None) or "").strip()
    return indexed.get(pmcid) or indexed.get(pmcid.removeprefix("PMC"))


def _activation_table_fingerprint(table: Any) -> Dict[str, Any]:
    """Build a path-independent fingerprint for one coordinate source table."""
    content_hash = None
    raw_table = getattr(table, "raw_table", None)
    if raw_table is not None:
        content_hash = hashlib.sha256(str(raw_table).encode("utf-8")).hexdigest()
    else:
        for attribute in ("table_raw_path", "table_data_path"):
            raw_path = getattr(table, attribute, None)
            if not raw_path:
                continue
            path = Path(str(raw_path)).expanduser()
            if not path.is_file():
                continue
            stat = path.stat()
            content_hash = _file_content_sha256(
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
            )
            break
    return {
        "table_id": getattr(table, "table_id", None),
        "table_label": getattr(table, "table_label", None),
        "table_caption": getattr(table, "table_caption", None),
        "table_foot": getattr(table, "table_foot", None),
        "content_hash": content_hash,
    }


def coordinate_study_input_hash(study: Any) -> str:
    """Hash coordinate-parsing inputs without coupling reuse to file paths."""
    return stable_hash(
        {
            "pmid": getattr(study, "pmid", None),
            "activation_tables": [
                _activation_table_fingerprint(table)
                for table in (getattr(study, "activation_tables", None) or [])
            ],
            "source_analyses": [
                analysis
                for analysis in (getattr(study, "analyses", None) or [])
                if not getattr(analysis, "parsed", False)
            ],
        }
    )


def pipeline_config_to_dict(config: Any) -> Dict[str, Any]:
    """Serialize a PipelineConfig-like object without requiring a hard type."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return _to_plain(config) or {}


def _pick(mapping: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: mapping.get(key) for key in keys if key in mapping}


def stage_signature_payloads(config_or_dict: Any) -> Dict[str, Any]:
    """Build per-stage semantic payloads for cache validation."""
    config = pipeline_config_to_dict(config_or_dict)
    search = config.get("search") or {}
    screening = config.get("screening") or {}
    retrieval = config.get("retrieval") or {}
    parsing = config.get("parsing") or {}
    annotation = config.get("annotation") or {}
    output = config.get("output") or {}

    return {
        "search": _pick(
            search,
            [
                "database",
                "query",
                "max_results",
                "date_from",
                "date_to",
                "pmids_file",
                "pmids_list",
            ],
        ),
        "abstract": {
            **(screening.get("abstract") or {}),
            "prompt_version": ABSTRACT_SCREENING_PROMPT_VERSION,
        },
        "retrieval": _pick(
            retrieval,
            [
                "sources",
                "timeout",
                "max_retries",
                "download_directory",
                "n_jobs",
                "load_excluded",
                "full_text_sources",
            ],
        ),
        "fulltext": {
            **(screening.get("fulltext") or {}),
            "prompt_version": FULLTEXT_SCREENING_PROMPT_VERSION,
        },
        "parsing": _pick(
            parsing or retrieval,
            ["parse_coordinates", "coordinate_model"],
        )
        | {"prompt_version": COORDINATE_PARSING_PROMPT_VERSION},
        "annotation": _pick(
            annotation,
            [
                "model",
                "create_all_included_annotations",
                "metadata_fields",
                "annotations",
                "enabled",
                "prompt_type",
                "inclusion_criteria",
                "exclusion_criteria",
            ],
        )
        | {"prompt_version": ANNOTATION_PROMPT_VERSION},
        "output": _pick(
            output,
            ["prisma_diagram", "formats", "nimads", "export_excluded_studies"],
        ),
    }


def stage_hashes(config_or_dict: Any) -> Dict[str, str]:
    """Return semantic hashes for all pipeline stages."""
    return {
        stage: stable_hash(payload)
        for stage, payload in stage_signature_payloads(config_or_dict).items()
    }


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "outputs" / "execution_manifest.json"


def execution_progress_path(output_dir: Path) -> Path:
    return output_dir / "outputs" / "execution_progress.json"


def load_execution_progress(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the execution progress file if present."""
    path = execution_progress_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load execution progress %s: %s", path, exc)
        return None


def load_execution_manifest(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the execution manifest if present."""
    path = manifest_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load execution manifest %s: %s", path, exc)
        return None


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _signature_is_modern(signature: Any) -> bool:
    return (
        isinstance(signature, dict)
        and signature.get("schema_version") == CACHE_SCHEMA_VERSION
        and bool(signature.get("stage"))
        and bool(signature.get("stage_hash"))
    )


def _artifact_is_modern(output_dir: Path, stage: str) -> bool:
    """Return whether a stage artifact uses the complete v2 cache contract."""
    outputs_dir = output_dir / "outputs"
    filenames = STAGE_ARTIFACTS.get(stage, [])
    existing = [
        outputs_dir / filename
        for filename in filenames
        if (outputs_dir / filename).exists()
    ]
    if not existing:
        return False

    if stage == "annotation":
        data = _safe_read_json(outputs_dir / "annotation_results.json")
        return isinstance(data, list) and all(
            isinstance(item, dict)
            and _signature_is_modern(item.get("cache_signature"))
            for item in data
        )

    primary = _safe_read_json(existing[0])
    if not isinstance(primary, dict):
        return False
    if _signature_is_modern(primary.get("cache_signature")):
        return True

    # Incremental screening writes can precede the final stage-level wrapper.
    if stage in {"abstract", "fulltext"}:
        rows = primary.get("screening_results")
        return bool(rows) and all(
            isinstance(item, dict)
            and _signature_is_modern(item.get("cache_signature"))
            for item in rows
        )
    return False


def _has_cache_artifacts(output_dir: Path) -> bool:
    outputs_dir = output_dir / "outputs"
    return outputs_dir.exists() and any(
        (outputs_dir / filename).exists()
        for filenames in STAGE_ARTIFACTS.values()
        for filename in filenames
    )


def _stage_action(
    stage: str,
    *,
    has_artifact: bool,
    artifact_is_modern: bool,
    stage_changed: bool,
    cache_policy: str,
    cleared: set[str],
) -> tuple[str, str, bool]:
    """Return (action, reason, copy_artifact) for one stage."""
    if cache_policy == "ignore" or stage in cleared:
        return "recompute", "explicit refresh requested", False
    if stage == "output":
        return "recompute", "derived outputs are regenerated", False
    if not has_artifact:
        return "recompute", "no cache artifact is available", False
    if not artifact_is_modern:
        return "unverified", "cache artifact is unsigned or uses an older schema", False

    if stage == "search":
        reason = (
            "search changed; cached metadata will be matched by PMID"
            if stage_changed
            else "metadata cache is valid and new PMIDs will be fetched"
        )
        return "incremental", reason, True
    if stage == "retrieval":
        if stage_changed:
            return "recompute", "retrieval configuration changed", False
        return "incremental", "retrieval cache is valid and missing texts will be fetched", True
    if stage_changed:
        return "recompute", f"{stage} configuration changed", False
    return "incremental", "signed entries will be validated against current inputs", True


def build_cache_plan(
    config: Any,
    output_dir: Path,
    *,
    cache_policy: str = "auto",
    clear_cache: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build the authoritative cache-reuse plan without modifying files."""
    cache_policy = (cache_policy or "auto").strip().lower()
    if cache_policy not in CACHE_POLICIES:
        raise ValueError(
            f"Invalid cache policy '{cache_policy}'. Expected one of: "
            f"{', '.join(sorted(CACHE_POLICIES))}"
        )

    output_dir = output_dir.expanduser().resolve()
    current_hashes = stage_hashes(config)
    previous_manifest = load_execution_manifest(output_dir)
    has_artifacts = _has_cache_artifacts(output_dir)
    manifest_is_modern = bool(
        previous_manifest
        and previous_manifest.get("schema_version") == CACHE_SCHEMA_VERSION
    )
    unsupported_cache = bool(has_artifacts and not manifest_is_modern)
    previous_hashes = (
        previous_manifest.get("stage_hashes") or {}
        if manifest_is_modern
        else {}
    )
    changed_stages = [
        stage
        for stage, value in current_hashes.items()
        if previous_hashes and previous_hashes.get(stage) != value
    ]
    cleared = set(_expand_clear_stages(clear_cache or []))

    stages: Dict[str, Dict[str, Any]] = {}
    for stage, filenames in STAGE_ARTIFACTS.items():
        outputs_dir = output_dir / "outputs"
        has_artifact = any((outputs_dir / filename).exists() for filename in filenames)
        modern = manifest_is_modern and _artifact_is_modern(output_dir, stage)
        action, reason, copy_artifact = _stage_action(
            stage,
            has_artifact=has_artifact,
            artifact_is_modern=modern,
            stage_changed=stage in changed_stages,
            cache_policy=cache_policy,
            cleared=cleared,
        )
        stages[stage] = {
            "action": action,
            "reason": reason,
            "has_artifact": has_artifact,
            "artifact_is_modern": modern,
            "config_changed": stage in changed_stages,
            "copy_artifact": copy_artifact,
        }

    unverified_stages = [
        stage for stage, item in stages.items() if item["action"] == "unverified"
    ]
    unsupported_cache = unsupported_cache or bool(unverified_stages)

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_output": str(output_dir),
        "has_previous_execution": bool(previous_manifest),
        "has_cache_artifacts": has_artifacts,
        "unsupported_cache": unsupported_cache,
        "changed_stages": changed_stages,
        "stage_hashes": current_hashes,
        "stages": stages,
        "copyable_stages": [
            stage for stage, item in stages.items() if item["copy_artifact"]
        ],
        "recompute_stages": [
            stage for stage, item in stages.items() if item["action"] == "recompute"
        ],
        "unverified_stages": unverified_stages,
    }


def _count_screening_decisions(results: Any) -> Dict[str, int]:
    screened = len(results) if isinstance(results, list) else 0
    included = 0
    excluded = 0
    incomplete = 0
    for item in results if isinstance(results, list) else []:
        decision = str(item.get("decision", "")).lower() if isinstance(item, dict) else ""
        if "included" in decision:
            included += 1
        elif "incomplete" in decision:
            incomplete += 1
        else:
            excluded += 1
    payload = {
        "screened": screened,
        "included": included,
        "excluded": excluded,
    }
    if incomplete:
        payload["incomplete"] = incomplete
    return payload


def _read_stage_counters(output_dir: Path, stage: str) -> Dict[str, Any]:
    """Read lightweight summary counters from existing stage artifacts."""
    outputs_dir = output_dir / "outputs"
    if stage == "search":
        data = _safe_read_json(outputs_dir / "search_results.json")
        if isinstance(data, dict):
            studies = data.get("studies", [])
            return {"studies_found": len(studies) if isinstance(studies, list) else 0}
    if stage == "abstract":
        data = _safe_read_json(outputs_dir / "abstract_screening_results.json")
        if isinstance(data, dict):
            return _count_screening_decisions(data.get("screening_results", []))
    if stage == "retrieval":
        data = _safe_read_json(outputs_dir / "fulltext_retrieval_results.json")
        if isinstance(data, dict):
            rows = data.get("studies_with_fulltext", [])
            available = [
                row for row in rows
                if isinstance(row, dict) and row.get("fulltext_available")
            ] if isinstance(rows, list) else []
            return {
                "fulltext_candidates": len(rows) if isinstance(rows, list) else 0,
                "available": len(available),
            }
    if stage == "fulltext":
        data = _safe_read_json(outputs_dir / "fulltext_screening_results.json")
        if isinstance(data, dict):
            counters = _count_screening_decisions(data.get("screening_results", []))
            counters.setdefault("incomplete", 0)
            return counters
    if stage == "parsing":
        data = _safe_read_json(outputs_dir / "coordinate_parsing_results.json")
        if isinstance(data, dict):
            studies = data.get("studies", [])
            analyses_count = 0
            coordinates_count = 0
            for study in studies if isinstance(studies, list) else []:
                analyses = study.get("analyses", []) if isinstance(study, dict) else []
                if not isinstance(analyses, list):
                    continue
                analyses_count += len(analyses)
                for analysis in analyses:
                    points = analysis.get("points", []) if isinstance(analysis, dict) else []
                    if isinstance(points, list):
                        coordinates_count += len(points)
            return {
                "studies": len(studies) if isinstance(studies, list) else 0,
                "analyses": analyses_count,
                "coordinates": coordinates_count,
            }
    if stage == "annotation":
        data = _safe_read_json(outputs_dir / "annotation_results.json")
        if isinstance(data, list):
            annotation_names = {
                str(item.get("annotation_name", "")).strip()
                for item in data
                if isinstance(item, dict) and str(item.get("annotation_name", "")).strip()
            }
            return {"decisions": len(data), "annotations": len(annotation_names)}
    if stage == "output":
        data = _safe_read_json(outputs_dir / "final_results.json")
        if isinstance(data, dict):
            stats = data.get("execution_stats", {})
            counters = stats.get("prisma_stats", {})
            if isinstance(counters, dict):
                payload = dict(counters)
                payload["nimads_available"] = (outputs_dir / "nimads_studyset.json").exists()
                return payload
    return {}


def _expand_clear_stages(stages: Iterable[str]) -> List[str]:
    normalized = [str(stage).strip().lower() for stage in stages if str(stage).strip()]
    invalid = [stage for stage in normalized if stage not in CLEAR_CACHE_STAGES]
    if invalid:
        raise ValueError(f"Invalid clear-cache stage(s): {', '.join(invalid)}")
    if "all" in normalized:
        return [stage for stage in STAGE_ARTIFACTS]
    return sorted(set(normalized), key=list(STAGE_ARTIFACTS).index)


def _delete_stage_artifacts(output_dir: Path, stages: Iterable[str]) -> List[Dict[str, Any]]:
    outputs_dir = output_dir / "outputs"
    removed: List[Dict[str, Any]] = []
    for stage in _expand_clear_stages(stages):
        for filename in STAGE_ARTIFACTS.get(stage, []):
            path = outputs_dir / filename
            if path.exists():
                path.unlink()
                removed.append({"stage": stage, "path": str(path)})
    return removed


def _copy_stage_artifacts(
    source_output_dir: Path,
    target_output_dir: Path,
    stages: Iterable[str],
    *,
    copy_retrieval_data: bool = True,
) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    source_outputs = source_output_dir / "outputs"
    target_outputs = target_output_dir / "outputs"
    target_outputs.mkdir(parents=True, exist_ok=True)
    for stage in _expand_clear_stages(stages):
        for filename in STAGE_ARTIFACTS.get(stage, []):
            src = source_outputs / filename
            dst = target_outputs / filename
            if src.exists():
                shutil.copy2(src, dst)
                copied.append({"stage": stage, "from": str(src), "to": str(dst)})
    retrieval_src = source_output_dir / "retrieval"
    retrieval_dst = target_output_dir / "retrieval"
    if retrieval_src.exists() and copy_retrieval_data:
        if retrieval_dst.exists():
            shutil.rmtree(retrieval_dst)
        shutil.copytree(retrieval_src, retrieval_dst)
        copied.append({
            "stage": "retrieval_data",
            "from": str(retrieval_src),
            "to": str(retrieval_dst),
        })
    return copied


def write_executed_config(config: Any, output_dir: Path) -> Path:
    """Write the exact runtime config snapshot used by this execution."""
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / "config.executed.yaml"
    path.write_text(
        yaml.safe_dump(pipeline_config_to_dict(config), sort_keys=False),
        encoding="utf-8",
    )
    return path


def prepare_execution(
    config: Any,
    output_dir: Path,
    *,
    cache_policy: str = "auto",
    clear_cache: Optional[Iterable[str]] = None,
    copy_valid_cache_from: Optional[str] = None,
    stop_after_stage: str = "full",
) -> Dict[str, Any]:
    """Prepare output provenance from the shared cache-validity plan."""
    cache_policy = (cache_policy or "auto").strip().lower()
    if cache_policy not in CACHE_POLICIES:
        raise ValueError(
            f"Invalid cache policy '{cache_policy}'. Expected one of: "
            f"{', '.join(sorted(CACHE_POLICIES))}"
        )

    output_dir = output_dir.expanduser().resolve()
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    current_hashes = stage_hashes(config)
    current_config_hash = stable_hash(
        {
            key: value
            for key, value in pipeline_config_to_dict(config).items()
            if key != "output"
        }
    )

    local_plan = build_cache_plan(
        config,
        output_dir,
        cache_policy=cache_policy,
        clear_cache=clear_cache,
    )
    if local_plan["unsupported_cache"] and cache_policy == "auto":
        raise UnsupportedCacheError(
            f"The cache in {output_dir} cannot be verified by this version. "
            "Use --cache-policy ignore to recompute generated results."
        )

    previous_manifest = load_execution_manifest(output_dir)
    parent_execution_id = (
        previous_manifest.get("execution_id") if previous_manifest else None
    )
    copied: List[Dict[str, Any]] = []
    active_plan = local_plan

    # In-place recomputation removes only artifacts the planner cannot safely
    # validate item by item. Valid incremental artifacts remain available.
    invalidated: List[Dict[str, Any]] = []
    if cache_policy == "ignore":
        invalidated.extend(_delete_stage_artifacts(output_dir, ["all"]))
    elif previous_manifest:
        invalidated.extend(
            _delete_stage_artifacts(output_dir, local_plan["recompute_stages"])
        )

    if copy_valid_cache_from and cache_policy != "ignore":
        source_output_dir = Path(copy_valid_cache_from).expanduser().resolve()
        source_plan = build_cache_plan(
            config,
            source_output_dir,
            cache_policy=cache_policy,
            clear_cache=clear_cache,
        )
        if source_plan["unsupported_cache"]:
            raise UnsupportedCacheError(
                f"The cache in {source_output_dir} cannot be verified by this "
                "version and cannot be used as a cache source."
            )
        source_manifest = load_execution_manifest(source_output_dir)
        if source_manifest:
            parent_execution_id = source_manifest.get("execution_id")
        copied = _copy_stage_artifacts(
            source_output_dir,
            output_dir,
            source_plan["copyable_stages"],
            copy_retrieval_data=True,
        )
        active_plan = source_plan

    explicit_clear = _expand_clear_stages(clear_cache or [])
    if explicit_clear:
        invalidated.extend(_delete_stage_artifacts(output_dir, explicit_clear))

    executed_config_path = write_executed_config(config, output_dir)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "execution_id": str(uuid4()),
        "parent_execution_id": parent_execution_id,
        "status": "running",
        "cache_policy": cache_policy,
        "created_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "completed_at": None,
        "config_hash": current_config_hash,
        "stage_hashes": current_hashes,
        "changed_stages": active_plan["changed_stages"],
        "invalidated": invalidated,
        "copied_cache": copied,
        "cache_plan": active_plan,
        "stop_after_stage": stop_after_stage,
        "executed_config_path": str(executed_config_path),
    }
    write_execution_manifest(output_dir, manifest)

    if active_plan["changed_stages"]:
        logger.info(
            "Execution signatures changed for stages: %s",
            ", ".join(active_plan["changed_stages"]),
        )
    if invalidated:
        logger.info("Invalidated %s stale cache artifact(s)", len(invalidated))
    if copied:
        logger.info("Copied %s valid cache artifact(s)", len(copied))

    return manifest


def preview_execution_changes(
    config: Any,
    output_dir: Path,
    *,
    cache_policy: str = "auto",
    clear_cache: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Preview the same cache plan execution will apply."""
    plan = build_cache_plan(
        config,
        output_dir,
        cache_policy=cache_policy,
        clear_cache=clear_cache,
    )
    return {
        **plan,
        # Retained for older UI clients; this now means true recomputation,
        # not a separate dependency approximation.
        "invalidates": plan["recompute_stages"],
    }


def write_execution_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Persist manifest atomically."""
    path = manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def write_execution_progress(output_dir: Path, progress: Dict[str, Any]) -> Path:
    """Persist execution progress atomically."""
    path = execution_progress_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def _stage_status_template(stage: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "pending",
        "source": "unknown",
        "started_at": None,
        "completed_at": None,
        "error": None,
        "counters": {},
        "planned": True,
    }


def _planned_stages(stop_after_stage: str) -> set[str]:
    """Return the pipeline stages included in a stage-limited execution."""
    stop_stage = RUN_STAGE_LIMITS.get(str(stop_after_stage).strip().lower(), "output")
    ordered = list(STAGE_ARTIFACTS)
    return set(ordered[: ordered.index(stop_stage) + 1])


def _invalidated_stage_names(manifest: Dict[str, Any]) -> List[str]:
    invalidated = {
        str(item.get("stage", "")).strip()
        for item in manifest.get("invalidated", [])
        if isinstance(item, dict) and item.get("stage")
    }
    cache_plan = manifest.get("cache_plan") or {}
    invalidated.update(cache_plan.get("recompute_stages", []))
    return sorted(invalidated, key=list(STAGE_ARTIFACTS).index)


def initialize_execution_progress(output_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Create the authoritative progress file for a new execution."""
    output_dir = output_dir.expanduser().resolve()
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    invalidated = set(_invalidated_stage_names(manifest))
    copied = {
        str(item.get("stage", "")).strip()
        for item in manifest.get("copied_cache", [])
        if isinstance(item, dict) and item.get("stage")
    }
    planned = _planned_stages(manifest.get("stop_after_stage", "full"))

    stages: List[Dict[str, Any]] = []
    for stage in STAGE_ARTIFACTS:
        item = _stage_status_template(stage)
        item["planned"] = stage in planned
        has_artifact = any(
            (outputs_dir / filename).exists()
            for filename in STAGE_ARTIFACTS[stage]
        )
        if item["planned"] and stage not in invalidated and (stage in copied or has_artifact):
            item.update(
                {
                    "status": "completed",
                    "source": "cache",
                    "completed_at": utc_now_iso(),
                    "counters": _read_stage_counters(output_dir, stage),
                }
            )
        stages.append(item)

    progress = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "execution_id": manifest.get("execution_id"),
        "status": "running",
        "current_stage": None,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "completed_at": None,
        "cache": {
            "changed_stages": manifest.get("changed_stages", []),
            "invalidated_stages": sorted(invalidated, key=list(STAGE_ARTIFACTS).index),
        },
        "stop_after_stage": manifest.get("stop_after_stage", "full"),
        "stages": stages,
    }
    write_execution_progress(output_dir, progress)
    return progress


def update_execution_progress_stage(
    output_dir: Path,
    stage: str,
    *,
    status: str,
    source: str = "fresh",
    counters: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Update one stage in the authoritative execution progress file."""
    stage = str(stage).strip().lower()
    if stage not in STAGE_ARTIFACTS:
        raise ValueError(f"Unknown execution stage: {stage}")
    if status not in EXECUTION_PROGRESS_STATUSES:
        raise ValueError(f"Unknown execution progress status: {status}")
    if source not in EXECUTION_PROGRESS_SOURCES:
        raise ValueError(f"Unknown execution progress source: {source}")

    output_dir = output_dir.expanduser().resolve()
    progress = load_execution_progress(output_dir)
    if not progress:
        manifest = load_execution_manifest(output_dir) or {}
        progress = initialize_execution_progress(output_dir, manifest)

    now = utc_now_iso()
    stage_items = progress.setdefault(
        "stages",
        [_stage_status_template(item) for item in STAGE_ARTIFACTS],
    )
    by_stage = {
        str(item.get("stage")): item
        for item in stage_items
        if isinstance(item, dict) and item.get("stage")
    }
    item = by_stage.get(stage)
    if item is None:
        item = _stage_status_template(stage)
        stage_items.append(item)

    item["status"] = status
    item["source"] = source
    item["error"] = error
    if status == "running" and not item.get("started_at"):
        item["started_at"] = now
    if status in {"completed", "skipped", "failed"}:
        item["completed_at"] = now
    if counters is not None:
        item["counters"] = counters
    elif status in {"completed", "skipped"} and not item.get("counters"):
        item["counters"] = _read_stage_counters(output_dir, stage)

    # Token/cost accounting for whatever this execution actually computed. Incremental stages
    # only call the API for uncached items, so this is the cost of *this run* rather than of
    # building the artifact from scratch -- `counters` above says how much was reused. Absent
    # when a stage made no calls (fully cached, disabled, or non-LLM).
    if status in {"completed", "skipped", "failed"}:
        stage_usage = llm_usage.snapshot(stage)
        if stage_usage:
            item["usage"] = stage_usage

    progress["status"] = "failed" if status == "failed" else "running"
    progress["current_stage"] = stage if status == "running" else None
    progress["updated_at"] = now
    write_execution_progress(output_dir, progress)
    return progress


def complete_execution_progress(
    output_dir: Path,
    *,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Mark execution progress globally complete/failed/canceled."""
    output_dir = output_dir.expanduser().resolve()
    progress = load_execution_progress(output_dir)
    if not progress:
        return
    progress["status"] = status
    progress["current_stage"] = None
    progress["updated_at"] = utc_now_iso()
    progress["completed_at"] = utc_now_iso()
    if error:
        progress["error"] = error
    if status == "completed":
        now = utc_now_iso()
        for item in progress.get("stages", []):
            if (
                isinstance(item, dict)
                and item.get("planned") is False
                and item.get("status") == "pending"
            ):
                item["status"] = "skipped"
                item["source"] = "not_applicable"
                item["completed_at"] = now

    # Run-level roll-up. Summed from the per-stage records already on the progress file rather
    # than from the accumulator, so a resumed run that reused earlier stages still reports the
    # cost of everything this file describes.
    stage_usage = [
        item.get("usage")
        for item in progress.get("stages", [])
        if isinstance(item, dict) and isinstance(item.get("usage"), dict)
    ]
    if stage_usage:
        costs = [u.get("cost_usd") for u in stage_usage]
        progress["usage_total"] = {
            "calls": sum(u.get("calls", 0) for u in stage_usage),
            "input_tokens": sum(u.get("input_tokens", 0) for u in stage_usage),
            "uncached_input_tokens": sum(u.get("uncached_input_tokens", 0) for u in stage_usage),
            "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for u in stage_usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in stage_usage),
            # None if any stage used an unpriced model: a partial total would read as a full one.
            "cost_usd": (
                round(sum(costs), 6) if all(c is not None for c in costs) else None
            ),
        }
    write_execution_progress(output_dir, progress)


def complete_execution_manifest(
    output_dir: Path,
    *,
    status: str,
    completed_stage: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> None:
    """Update manifest status at run completion/failure."""
    manifest = load_execution_manifest(output_dir) or {}
    manifest["status"] = status
    manifest["completed_at"] = utc_now_iso()
    if completed_stage:
        manifest["completed_stage"] = completed_stage
    if errors:
        manifest["errors"] = errors
    write_execution_manifest(output_dir, manifest)
