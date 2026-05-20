import json
from pathlib import Path

from autonima.config import ConfigManager
from autonima.execution import (
    initialize_execution_progress,
    load_execution_progress,
    load_execution_manifest,
    prepare_execution,
    preview_execution_changes,
    stage_hashes,
    stable_hash,
    update_execution_progress_stage,
)
from autonima.models.types import ScreeningConfig, Study
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
    fulltext_cache.write_text('{"screening_results": []}', encoding="utf-8")
    abstract_cache.write_text('{"screening_results": []}', encoding="utf-8")

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
    (outputs / "search_results.json").write_text(
        '{"studies": [{"pmid": "1"}, {"pmid": "2"}]}',
        encoding="utf-8",
    )
    (outputs / "fulltext_screening_results.json").write_text(
        '{"screening_results": []}',
        encoding="utf-8",
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

    assert screener._study_input_hash(first, "fulltext") == screener._study_input_hash(second, "fulltext")
