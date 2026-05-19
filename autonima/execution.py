"""Execution provenance and cache-signature helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import yaml

logger = logging.getLogger(__name__)

CACHE_POLICIES = {"auto", "ignore", "trust-legacy"}
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
        "missing_fulltexts.csv",
        "missing_fulltexts.txt",
    ],
}


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
        "abstract": screening.get("abstract") or {},
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
        "fulltext": screening.get("fulltext") or {},
        "parsing": _pick(
            parsing or retrieval,
            ["parse_coordinates", "coordinate_model"],
        ),
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
        ),
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


def _load_legacy_config_hashes(output_dir: Path) -> Optional[Dict[str, str]]:
    final_results = output_dir / "outputs" / "final_results.json"
    if not final_results.exists():
        return None
    try:
        data = json.loads(final_results.read_text(encoding="utf-8"))
        config = data.get("config")
        if isinstance(config, dict):
            return stage_hashes(config)
    except Exception as exc:
        logger.warning("Failed to inspect legacy final_results config: %s", exc)
    return None


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


def _stages_to_invalidate(changed: Iterable[str]) -> List[str]:
    """Map changed semantic signatures to conservative cache invalidation."""
    stages = set()
    changed_set = set(changed)
    if "search" in changed_set:
        stages.add("search")
        stages.add("output")
    if "abstract" in changed_set:
        stages.add("abstract")
        stages.add("output")
    if "retrieval" in changed_set:
        stages.update({"retrieval", "fulltext", "parsing", "annotation", "output"})
    if "fulltext" in changed_set:
        stages.add("fulltext")
        stages.add("output")
    if "parsing" in changed_set:
        stages.update({"parsing", "annotation", "output"})
    if "annotation" in changed_set:
        stages.update({"annotation", "output"})
    if "output" in changed_set:
        stages.add("output")
    return sorted(stages, key=list(STAGE_ARTIFACTS).index)


def _copy_stage_artifacts(
    source_output_dir: Path,
    target_output_dir: Path,
    stages: Iterable[str],
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
    if retrieval_src.exists() and "retrieval" in set(stages):
        if retrieval_dst.exists():
            shutil.rmtree(retrieval_dst)
        shutil.copytree(retrieval_src, retrieval_dst)
        copied.append({"stage": "retrieval", "from": str(retrieval_src), "to": str(retrieval_dst)})
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
) -> Dict[str, Any]:
    """Prepare output provenance and invalidate stale stage caches."""
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

    previous_manifest = load_execution_manifest(output_dir)
    previous_hashes = None
    legacy_unverified = False
    parent_execution_id = None
    if previous_manifest:
        previous_hashes = previous_manifest.get("stage_hashes") or {}
        parent_execution_id = previous_manifest.get("execution_id")
    else:
        previous_hashes = _load_legacy_config_hashes(output_dir)
        legacy_unverified = previous_hashes is None and any(outputs_dir.glob("*.json"))

    changed_stages = []
    if previous_hashes:
        changed_stages = [
            stage
            for stage, value in current_hashes.items()
            if previous_hashes.get(stage) != value
        ]

    copied = []
    if copy_valid_cache_from:
        source_output_dir = Path(copy_valid_cache_from).expanduser().resolve()
        source_hashes = {}
        source_manifest = load_execution_manifest(source_output_dir)
        if source_manifest:
            source_hashes = source_manifest.get("stage_hashes") or {}
            parent_execution_id = source_manifest.get("execution_id")
        else:
            source_hashes = _load_legacy_config_hashes(source_output_dir) or {}
        copyable = [
            stage
            for stage, value in current_hashes.items()
            if source_hashes.get(stage) == value
        ]
        copied = _copy_stage_artifacts(source_output_dir, output_dir, copyable)

    invalidated = []
    if cache_policy == "ignore":
        invalidated.extend(_delete_stage_artifacts(output_dir, ["all"]))
    elif cache_policy == "auto" and previous_hashes:
        invalidated.extend(_delete_stage_artifacts(output_dir, _stages_to_invalidate(changed_stages)))

    explicit_clear = _expand_clear_stages(clear_cache or [])
    if explicit_clear:
        invalidated.extend(_delete_stage_artifacts(output_dir, explicit_clear))

    executed_config_path = write_executed_config(config, output_dir)
    manifest = {
        "schema_version": 1,
        "execution_id": str(uuid4()),
        "parent_execution_id": parent_execution_id,
        "status": "running",
        "cache_policy": cache_policy,
        "created_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "completed_at": None,
        "config_hash": current_config_hash,
        "stage_hashes": current_hashes,
        "changed_stages": changed_stages,
        "invalidated": invalidated,
        "copied_cache": copied,
        "legacy_unverified": legacy_unverified,
        "executed_config_path": str(executed_config_path),
    }
    write_execution_manifest(output_dir, manifest)

    if changed_stages:
        logger.info(
            "Execution signatures changed for stages: %s",
            ", ".join(changed_stages),
        )
    if invalidated:
        logger.info("Invalidated %s stale cache artifact(s)", len(invalidated))
    if copied:
        logger.info("Copied %s valid cache artifact(s)", len(copied))

    return manifest


def preview_execution_changes(config: Any, output_dir: Path) -> Dict[str, Any]:
    """Preview signature changes without modifying cache artifacts."""
    output_dir = output_dir.expanduser().resolve()
    current_hashes = stage_hashes(config)
    previous_manifest = load_execution_manifest(output_dir)
    previous_hashes = None
    legacy_unverified = False
    if previous_manifest:
        previous_hashes = previous_manifest.get("stage_hashes") or {}
    else:
        previous_hashes = _load_legacy_config_hashes(output_dir)
        legacy_unverified = (
            previous_hashes is None
            and (output_dir / "outputs").exists()
            and any((output_dir / "outputs").glob("*.json"))
        )

    changed_stages = []
    if previous_hashes:
        changed_stages = [
            stage
            for stage, value in current_hashes.items()
            if previous_hashes.get(stage) != value
        ]

    return {
        "has_previous_execution": bool(previous_manifest or previous_hashes),
        "legacy_unverified": legacy_unverified,
        "changed_stages": changed_stages,
        "invalidates": _stages_to_invalidate(changed_stages),
        "stage_hashes": current_hashes,
    }


def write_execution_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Persist manifest atomically."""
    path = manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


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
