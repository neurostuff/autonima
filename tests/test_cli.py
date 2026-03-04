from importlib import resources
import os
import subprocess
import sys

from click.testing import CliRunner

from autonima.cli import create_sample_config, run, validate
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
