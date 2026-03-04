from importlib import resources
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from autonima.config import ConfigManager


def test_canonical_sample_config_is_valid():
    sample_path = (
        Path(__file__).resolve().parents[1]
        / "autonima"
        / "templates"
        / "sample_config.yml"
    )

    config = ConfigManager().load_from_file(sample_path)

    assert config.search.query
    assert config.screening.abstract.get("objective")
    assert config.screening.fulltext.get("objective")
    assert config.output.directory


def test_examples_sample_config_matches_canonical_template():
    canonical = (
        resources.files("autonima.templates")
        .joinpath("sample_config.yml")
        .read_text(encoding="utf-8")
    )
    example = (
        Path(__file__).resolve().parents[1] / "examples" / "sample_config.yml"
    ).read_text(encoding="utf-8")

    assert example == canonical


def test_mkdocs_build_strict():
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocs_click")

    repo_root = Path(__file__).resolve().parents[1]
    mkdocs_executable = shutil.which("mkdocs")
    if mkdocs_executable is None:
        pytest.skip("mkdocs executable is not available")

    subprocess.run(
        [mkdocs_executable, "build", "--strict"],
        cwd=repo_root,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
