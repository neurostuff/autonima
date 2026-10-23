from importlib import resources
import logging
import os
import subprocess
import sys
import types

from click.testing import CliRunner

from autonima.cli import (
    create_sample_config,
    meta,
    run,
    run_abstract,
    run_search,
    validate,
)
from autonima.config import ConfigManager


def test_run_uses_config_stem_as_default_output_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("search: {}\n", encoding="utf-8")

    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    result = CliRunner().invoke(run, [str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert sample_config.output.directory == str(config_path.with_suffix(""))


def test_validate_uses_config_stem_as_default_output_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")

    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    result = CliRunner().invoke(validate, [str(config_path)])

    assert result.exit_code == 0
    assert sample_config.output.directory == str(config_path.with_suffix(""))
    assert f"✓ Output directory: {config_path.with_suffix('')}" in result.output


def test_validate_keeps_explicit_output_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "default.yaml"
    config_path.write_text("search: {}\n", encoding="utf-8")
    explicit_output = tmp_path / "custom-output"

    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    result = CliRunner().invoke(
        validate, [str(config_path), str(explicit_output)]
    )

    assert result.exit_code == 0
    assert sample_config.output.directory == str(explicit_output)
    assert f"✓ Output directory: {explicit_output}" in result.output


def test_run_search_uses_config_stem_as_default_output_dir(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "search_only.yaml"
    config_path.write_text("search: {}\n", encoding="utf-8")

    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    result = CliRunner().invoke(run_search, [str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert sample_config.output.directory == str(config_path.with_suffix(""))


def test_run_abstract_uses_config_stem_as_default_output_dir(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "abstract_only.yaml"
    config_path.write_text("search: {}\n", encoding="utf-8")

    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    result = CliRunner().invoke(run_abstract, [str(config_path), "--dry-run"])

    assert result.exit_code == 0
    assert sample_config.output.directory == str(config_path.with_suffix(""))


def test_importing_cli_does_not_load_heavy_pipeline_modules():
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(__file__))
    env["PYTHONPATH"] = repo_root
    command = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import autonima.cli; "
            "print(json.dumps({"
            "'pipeline': 'autonima.pipeline' in sys.modules, "
            "'meta': 'autonima.meta' in sys.modules, "
            "'pubget': 'pubget' in sys.modules, "
            "'nimare': 'nimare' in sys.modules}))"
        ),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == (
        '{"pipeline": false, "meta": false, "pubget": false, "nimare": false}'
    )


def test_create_sample_config_outputs_canonical_template():
    result = CliRunner().invoke(create_sample_config)

    assert result.exit_code == 0
    assert result.output == (
        resources.files("autonima.templates")
        .joinpath("sample_config.yml")
        .read_text(encoding="utf-8")
    )


def test_run_passes_force_reextract_incomplete_fulltext_flag(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")

    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    captured = {}

    class DummyResults:
        execution_stats = {"completed_stage": "full"}
        studies = []
        errors = []

    async def fake_run_pipeline_from_config(
        config,
        stop_after_stage="full",
        num_workers=1,
        force_reextract_incomplete_fulltext=False,
    ):
        captured["stop_after_stage"] = stop_after_stage
        captured["num_workers"] = num_workers
        captured["force_reextract_incomplete_fulltext"] = (
            force_reextract_incomplete_fulltext
        )
        return DummyResults()

    monkeypatch.setattr(
        "autonima.cli._get_run_pipeline_from_config",
        lambda: fake_run_pipeline_from_config,
    )

    result = CliRunner().invoke(
        run,
        [str(config_path), "--force-reextract-incomplete-fulltext"],
    )

    assert result.exit_code == 0
    assert captured["stop_after_stage"] == "full"
    assert captured["force_reextract_incomplete_fulltext"] is True


def test_run_search_passes_search_stop_after_stage(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")
    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    captured = {}

    class DummyResults:
        execution_stats = {"completed_stage": "search"}
        studies = []
        errors = []

    async def fake_run_pipeline_from_config(
        config,
        stop_after_stage="full",
        num_workers=1,
        force_reextract_incomplete_fulltext=False,
    ):
        captured["stop_after_stage"] = stop_after_stage
        captured["num_workers"] = num_workers
        captured["force_reextract_incomplete_fulltext"] = (
            force_reextract_incomplete_fulltext
        )
        return DummyResults()

    monkeypatch.setattr(
        "autonima.cli._get_run_pipeline_from_config",
        lambda: fake_run_pipeline_from_config,
    )

    result = CliRunner().invoke(run_search, [str(config_path)])

    assert result.exit_code == 0
    assert captured["stop_after_stage"] == "search"
    assert captured["force_reextract_incomplete_fulltext"] is False


def test_run_abstract_passes_abstract_stop_after_stage(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yml"
    config_path.write_text("search: {}\n", encoding="utf-8")
    sample_config = ConfigManager().create_sample_config()

    def fake_load_from_file(self, path):
        assert path == str(config_path)
        return sample_config

    monkeypatch.setattr(ConfigManager, "load_from_file", fake_load_from_file)

    captured = {}

    class DummyResults:
        execution_stats = {"completed_stage": "abstract"}
        studies = []
        errors = []

    async def fake_run_pipeline_from_config(
        config,
        stop_after_stage="full",
        num_workers=1,
        force_reextract_incomplete_fulltext=False,
    ):
        captured["stop_after_stage"] = stop_after_stage
        captured["num_workers"] = num_workers
        captured["force_reextract_incomplete_fulltext"] = (
            force_reextract_incomplete_fulltext
        )
        return DummyResults()

    monkeypatch.setattr(
        "autonima.cli._get_run_pipeline_from_config",
        lambda: fake_run_pipeline_from_config,
    )

    result = CliRunner().invoke(run_abstract, [str(config_path)])

    assert result.exit_code == 0
    assert captured["stop_after_stage"] == "abstract"
    assert captured["force_reextract_incomplete_fulltext"] is False


def test_meta_reports_an_incompatible_nimare_instead_of_calling_it_missing(
    tmp_path, monkeypatch, caplog
):
    # An installed NiMARE that lacks an API autonima imports -- as 0.21 did when it dropped
    # nimare.nimads.Annotation -- used to be reported as "NiMARE is not installed", which sent
    # users to reinstall the same broken version. The stand-ins model one without Studyset.
    stand_ins = {
        "nimare": {},
        "nimare.correct": {"FDRCorrector": object, "FWECorrector": object},
        "nimare.workflows": {"CBMAWorkflow": object},
        "nimare.meta": {},
        "nimare.meta.cbma": {"MKDADensity": object, "ALE": object, "KDA": object},
        "nimare.nimads": {},
        "nimare.reports": {},
        "nimare.reports.base": {"run_reports": object},
    }
    for name, attrs in stand_ins.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "autonima.meta", raising=False)

    with caplog.at_level(logging.ERROR):
        result = CliRunner().invoke(meta, [str(tmp_path)])

    assert result.exit_code == 1
    assert "cannot import name 'Studyset'" in caplog.text
    assert "nimare>=0.11" in caplog.text
    assert "not installed" not in caplog.text
