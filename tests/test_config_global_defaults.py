from autonima.config import ConfigManager, ConfigurationError


def _minimal_valid_config():
    return {
        "search": {
            "database": "pubmed",
            "query": "working memory fMRI",
            "max_results": 10,
        },
        "retrieval": {
            "sources": ["pubget"],
            "load_excluded": False,
        },
        "screening": {
            "abstract": {
                "objective": "Identify task fMRI studies",
                "inclusion_criteria": ["Human participants"],
            },
            "fulltext": {
                "objective": "Identify task fMRI studies",
                "inclusion_criteria": ["Human participants"],
            },
        },
        "parsing": {
            "parse_coordinates": False,
        },
        "output": {
            "directory": "results",
        },
        "annotation": {
            "enabled": True,
        },
    }


def test_global_model_default_applies_to_missing_model_fields():
    config_dict = _minimal_valid_config()
    config_dict["defaults"] = {"model": "gpt-5-mini-2025-08-07"}

    config = ConfigManager().load_from_dict(config_dict)

    assert config.screening.abstract["model"] == "gpt-5-mini-2025-08-07"
    assert config.screening.fulltext["model"] == "gpt-5-mini-2025-08-07"
    assert config.annotation.model == "gpt-5-mini-2025-08-07"
    assert config.parsing.coordinate_model == "gpt-5-mini-2025-08-07"
    assert config.retrieval.coordinate_model == "gpt-5-mini-2025-08-07"


def test_global_model_default_does_not_override_explicit_values():
    config_dict = _minimal_valid_config()
    config_dict["defaults"] = {"model": "gpt-5-mini-2025-08-07"}
    config_dict["screening"]["abstract"]["model"] = "gpt-4"
    config_dict["screening"]["fulltext"]["model"] = "gpt-4o"
    config_dict["parsing"]["coordinate_model"] = "gpt-4o-mini"
    config_dict["annotation"]["model"] = "gpt-4.1-mini"

    config = ConfigManager().load_from_dict(config_dict)

    assert config.screening.abstract["model"] == "gpt-4"
    assert config.screening.fulltext["model"] == "gpt-4o"
    assert config.parsing.coordinate_model == "gpt-4o-mini"
    assert config.annotation.model == "gpt-4.1-mini"


def test_global_model_default_requires_non_empty_string():
    config_dict = _minimal_valid_config()
    config_dict["defaults"] = {"model": ""}

    try:
        ConfigManager().load_from_dict(config_dict)
    except ConfigurationError as exc:
        assert "defaults.model" in str(exc)
    else:
        raise AssertionError("Expected ConfigurationError for empty defaults.model")
