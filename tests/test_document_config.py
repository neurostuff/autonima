"""The documents config section: validation, and cache safety for runs that never use it."""

from copy import deepcopy

import pytest
import yaml

from autonima.config import ConfigManager, ConfigurationError, get_sample_config_text
from autonima.execution import stage_hashes


# Stage hashes of the canonical sample config, recorded before documents existed. An article
# run must keep every one of them, or upgrading would silently invalidate its caches.
SAMPLE_STAGE_HASHES = {
    "abstract": "9378a7188766253d2cc7bd52310f40bb8397c920b36cd3bf4c478f6974914fe7",
    "annotation": "969ad529da8a2d03f07829b5ceb71c4baa7d05bcc04f2c52f5a81f13d4b1e0e5",
    "fulltext": "9378a7188766253d2cc7bd52310f40bb8397c920b36cd3bf4c478f6974914fe7",
    "output": "9f6ecf3a2d930011b5cacdb818da0823debe8ef067afb0e082b93d5d96c343da",
    "parsing": "78b4e4fc1e2cbe16023f5ba23b8e4655d28bf8e4122fa562d92afe98aa08045a",
    "retrieval": "b4a43bde2aa38484959a2135ad77596d34cb629dfd93023f78d8f87112bf0112",
    "search": "a163c17df894e270f4b9d0864c2a7d77166691403a9f26f23e50d3f50c616d9f",
}


def _sample() -> dict:
    return yaml.safe_load(get_sample_config_text())


def _with_documents(tmp_path, **documents) -> dict:
    config = _sample()
    config["retrieval"]["records"] = {"enabled": True, "kind": "text", "root": str(tmp_path), **documents}
    config["annotation"]["metadata_fields"] = ["analysis_name", "study_title", "study_fulltext"]
    config["annotation"]["annotations"] = [
        {"name": "wm", "inclusion_criteria": ["Working memory task"]}
    ]
    return config


def _load(config: dict):
    return ConfigManager().load_from_dict(deepcopy(config))


def test_article_config_stage_hashes_are_unchanged():
    assert stage_hashes(_load(_sample())) == SAMPLE_STAGE_HASHES


def test_disabled_documents_serialize_to_nothing():
    config = _load(_sample())
    assert config.documents.enabled is False
    assert "records" not in config.to_dict()["retrieval"]


def test_enabling_documents_changes_only_the_stages_that_read_them(tmp_path):
    config = _with_documents(tmp_path)
    config["parsing"]["parse_coordinates"] = True  # allowed for a text source
    baseline = deepcopy(config)
    del baseline["retrieval"]["records"]

    before = stage_hashes(_load(baseline))
    after = stage_hashes(_load(config))

    changed = sorted(stage for stage in before if before[stage] != after[stage])
    assert changed == ["annotation", "fulltext", "retrieval"]


def test_description_is_part_of_the_retrieval_signature(tmp_path):
    first = stage_hashes(_load(_with_documents(tmp_path, description="a summary")))
    second = stage_hashes(_load(_with_documents(tmp_path, description="a record")))
    assert first["retrieval"] != second["retrieval"]


def test_valid_documents_config_loads(tmp_path):
    config = _load(_with_documents(tmp_path, description="a summary"))
    assert config.documents.kind == "text"
    assert config.documents.description == "a summary"
    assert config.to_dict()["retrieval"]["records"]["root"] == str(tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c, p: c["retrieval"]["records"].update(dir=str(p)), "Unknown retrieval.records key"),
        (lambda c, p: c["retrieval"]["records"].update(kind="pondie"), "retrieval.records.kind must be one of"),
        (lambda c, p: c["retrieval"]["records"].pop("root"), "retrieval.records.root is required"),
        (lambda c, p: c["retrieval"]["records"].update(root=str(p / "nope")), "not a directory"),
        (
            lambda c, p: c["retrieval"]["records"].update(kind="analyses"),
            "parsing.parse_coordinates must be false",
        ),
        (
            lambda c, p: c["annotation"].update(metadata_fields=["analysis_name"]),
            "study_fulltext",
        ),
    ],
)
def test_invalid_documents_config_is_refused_at_load(tmp_path, change, message):
    config = _with_documents(tmp_path)
    config["parsing"]["parse_coordinates"] = True  # the sample config's value
    change(config, tmp_path)
    with pytest.raises(ConfigurationError, match=message):
        _load(config)


def test_analyses_config_loads_without_coordinate_parsing(tmp_path):
    config = _with_documents(tmp_path, kind="analyses")
    config["parsing"]["parse_coordinates"] = False
    assert _load(config).documents.kind == "analyses"


def test_study_fulltext_is_not_required_without_llm_annotations(tmp_path):
    config = _with_documents(tmp_path)
    config["annotation"]["metadata_fields"] = ["analysis_name"]
    config["annotation"].pop("annotations", None)
    assert _load(config).documents.enabled is True


def test_top_level_documents_section_is_refused(tmp_path):
    config = _sample()
    config["documents"] = {"enabled": True, "kind": "text", "root": str(tmp_path)}
    with pytest.raises(ConfigurationError, match="now 'retrieval.records'"):
        _load(config)
