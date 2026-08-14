import json
from pathlib import Path

from autonima.config import ConfigManager
from autonima.execution import (
    CACHE_SCHEMA_VERSION,
    UnsupportedCacheError,
    build_cache_plan,
    complete_execution_progress,
    coordinate_study_input_hash,
    initialize_execution_progress,
    load_execution_progress,
    load_execution_manifest,
    prepare_execution,
    preview_execution_changes,
    stage_hashes,
    stable_hash,
    update_execution_progress_stage,
)
import pytest
from autonima.models.types import ActivationTable, ScreeningConfig, Study
from autonima.screening.screener import LLMScreener


def _config_dict(fulltext_objective="full text objective"):
    return {
        "search": {"database": "pubmed", "query": "ptsd vbm"},
        "screening": {
            "abstract": {
                "objective": "abstract objective",
                "inclusion_criteria": ["human neuroimaging"],
                "exclusion_criteria": ["case report"],
            },
            "fulltext": {
                "objective": fulltext_objective,
                "inclusion_criteria": ["whole-brain coordinates"],
                "exclusion_criteria": ["roi only"],
            },
        },
        "retrieval": {"sources": ["pubget"], "load_excluded": False},
        "parsing": {"parse_coordinates": True, "coordinate_model": "gpt-4o-mini"},
        "annotation": {
            "enabled": True,
            "model": "gpt-4o-mini",
            "annotations": [{"name": "custom", "inclusion_criteria": ["ptsd"]}],
        },
        "output": {"directory": "results", "formats": ["csv"], "nimads": True},
    }


def _config(fulltext_objective="full text objective"):
    return ConfigManager().load_from_dict(_config_dict(fulltext_objective))


def _write_signed_stage(path: Path, stage: str, stage_hash: str, payload: dict) -> None:
    payload = dict(payload)
    payload["cache_signature"] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "stage_hash": stage_hash,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stable_hash_is_deterministic_and_key_order_independent():
    assert stable_hash({"b": 2, "a": [1, 2]}) == stable_hash({"a": [1, 2], "b": 2})


def test_stage_hash_change_is_scoped_to_fulltext():
    original = stage_hashes(_config("first"))
    changed = stage_hashes(_config("second"))

    changed_stages = {
        stage for stage, digest in changed.items() if original.get(stage) != digest
    }

    assert changed_stages == {"fulltext"}


def test_prepare_execution_writes_manifest_and_executed_config(tmp_path):
    config = _config()
    config.output.directory = str(tmp_path)

    manifest = prepare_execution(config, tmp_path)

    assert (tmp_path / "outputs" / "config.executed.yaml").exists()
    assert (tmp_path / "outputs" / "execution_manifest.json").exists()
    loaded = load_execution_manifest(tmp_path)
    assert loaded["execution_id"] == manifest["execution_id"]
    assert loaded["stage_hashes"]["search"] == stage_hashes(config)["search"]


def test_prepare_execution_invalidates_changed_fulltext_cache(tmp_path):
    config = _config("first")
    config.output.directory = str(tmp_path)
    prepare_execution(config, tmp_path)

    outputs = tmp_path / "outputs"
    fulltext_cache = outputs / "fulltext_screening_results.json"
    abstract_cache = outputs / "abstract_screening_results.json"
    hashes = stage_hashes(config)
    _write_signed_stage(
        fulltext_cache,
        "fulltext",
        hashes["fulltext"],
        {"screening_results": []},
    )
    _write_signed_stage(
        abstract_cache,
        "abstract",
        hashes["abstract"],
        {"screening_results": []},
    )

    changed_config = _config("second")
    changed_config.output.directory = str(tmp_path)
    preview = preview_execution_changes(changed_config, tmp_path)
    manifest = prepare_execution(changed_config, tmp_path)

    assert preview["changed_stages"] == ["fulltext"]
    assert fulltext_cache.exists() is False
    assert abstract_cache.exists() is True
    assert any(item["stage"] == "fulltext" for item in manifest["invalidated"])


def test_execution_progress_initializes_cache_and_invalidated_stages(tmp_path):
    config = _config("first")
    config.output.directory = str(tmp_path)
    prepare_execution(config, tmp_path)

    outputs = tmp_path / "outputs"
    hashes = stage_hashes(config)
    _write_signed_stage(
        outputs / "search_results.json",
        "search",
        hashes["search"],
        {"studies": [{"pmid": "1"}, {"pmid": "2"}]},
    )
    _write_signed_stage(
        outputs / "fulltext_screening_results.json",
        "fulltext",
        hashes["fulltext"],
        {"screening_results": []},
    )

    changed_config = _config("second")
    changed_config.output.directory = str(tmp_path)
    manifest = prepare_execution(changed_config, tmp_path)
    progress = initialize_execution_progress(tmp_path, manifest)
    statuses = {item["stage"]: item for item in progress["stages"]}

    assert statuses["search"]["status"] == "completed"
    assert statuses["search"]["source"] == "cache"
    assert statuses["search"]["counters"]["studies_found"] == 2
    assert statuses["fulltext"]["status"] == "pending"
    assert "fulltext" in progress["cache"]["invalidated_stages"]


def test_retrieval_change_preserves_modern_fulltext_cache(tmp_path):
    config = _config()
    config.output.directory = str(tmp_path)
    prepare_execution(config, tmp_path)
    hashes = stage_hashes(config)
    fulltext_cache = tmp_path / "outputs" / "fulltext_screening_results.json"
    _write_signed_stage(
        fulltext_cache,
        "fulltext",
        hashes["fulltext"],
        {"screening_results": []},
    )

    changed = _config()
    changed.output.directory = str(tmp_path)
    changed.retrieval.load_excluded = True
    preview = build_cache_plan(changed, tmp_path)
    manifest = prepare_execution(changed, tmp_path)

    assert preview["changed_stages"] == ["retrieval"]
    assert preview["stages"]["fulltext"]["action"] == "incremental"
    assert fulltext_cache.exists()
    assert not any(item["stage"] == "fulltext" for item in manifest["invalidated"])


def test_auto_rejects_unsigned_cache(tmp_path):
    config = _config()
    config.output.directory = str(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "abstract_screening_results.json").write_text(
        '{"screening_results": []}',
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedCacheError):
        prepare_execution(config, tmp_path)


def test_execution_progress_stage_updates_are_atomic_and_reloadable(tmp_path):
    config = _config()
    config.output.directory = str(tmp_path)
    manifest = prepare_execution(config, tmp_path)
    initialize_execution_progress(tmp_path, manifest)

    update_execution_progress_stage(
        tmp_path,
        "abstract",
        status="running",
        source="fresh",
    )
    update_execution_progress_stage(
        tmp_path,
        "abstract",
        status="completed",
        source="fresh",
        counters={"screened": 3, "included": 2, "excluded": 1},
    )

    progress = load_execution_progress(tmp_path)
    abstract = {
        item["stage"]: item
        for item in progress["stages"]
    }["abstract"]
    assert abstract["status"] == "completed"
    assert abstract["source"] == "fresh"
    assert abstract["counters"]["included"] == 2


def test_stage_limited_progress_does_not_claim_downstream_cache_ran(tmp_path):
    config = _config()
    config.output.directory = str(tmp_path)
    manifest = prepare_execution(config, tmp_path, stop_after_stage="search")
    hashes = stage_hashes(config)
    _write_signed_stage(
        tmp_path / "outputs" / "fulltext_screening_results.json",
        "fulltext",
        hashes["fulltext"],
        {"screening_results": []},
    )

    progress = initialize_execution_progress(tmp_path, manifest)
    stages = {item["stage"]: item for item in progress["stages"]}
    assert stages["fulltext"]["planned"] is False
    assert stages["fulltext"]["status"] == "pending"

    complete_execution_progress(tmp_path, status="completed")
    completed = load_execution_progress(tmp_path)
    stages = {item["stage"]: item for item in completed["stages"]}
    assert stages["fulltext"]["status"] == "skipped"
    assert stages["fulltext"]["source"] == "not_applicable"


def test_fulltext_study_hash_ignores_path_when_content_matches(tmp_path):
    first_path = tmp_path / "first.html"
    second_path = tmp_path / "nested" / "second.html"
    second_path.parent.mkdir()
    full_text = "<html><body>same article body</body></html>"
    first_path.write_text(full_text, encoding="utf-8")
    second_path.write_text(full_text, encoding="utf-8")

    screener = LLMScreener(ScreeningConfig(), output_dir=str(tmp_path))
    first = Study(
        pmid="12345",
        title="Same study",
        abstract="",
        authors=[],
        journal="",
        publication_date="",
        full_text_path=str(first_path),
        fulltext_available=True,
    )
    second = Study(
        pmid="12345",
        title="Same study",
        abstract="",
        authors=[],
        journal="",
        publication_date="",
        full_text_path=str(second_path),
        fulltext_available=True,
    )

    assert screener._study_input_hash(
        first, "fulltext"
    ) == screener._study_input_hash(second, "fulltext")


def test_coordinate_study_hash_ignores_table_path_when_content_matches(tmp_path):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "nested" / "second.csv"
    second_path.parent.mkdir()
    table_content = "x,y,z\n1,2,3\n"
    first_path.write_text(table_content, encoding="utf-8")
    second_path.write_text(table_content, encoding="utf-8")

    def make_study(path):
        study = Study(
            pmid="123",
            title="Example",
            abstract="Abstract",
            authors=[],
            journal="Journal",
            publication_date="2024",
        )
        study.activation_tables = [
            ActivationTable(
                table_id="table-1",
                table_label="Table 1",
                table_data_path=str(path),
            )
        ]
        return study

    assert coordinate_study_input_hash(
        make_study(first_path)
    ) == coordinate_study_input_hash(make_study(second_path))
