from importlib import resources
import os
from pathlib import Path
import shutil
import subprocess
import sys

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


# The docs build imports autonima.cli (via mkdocs-click) to generate the CLI reference page, and
# it installs only the `docs` extra. So the CLI's module-scope import graph must stay inside the
# core dependencies. test_mkdocs_build_strict above cannot catch a violation: it runs in a
# developer environment where the optional packages happen to be installed, which is exactly how
# an undeclared `tqdm` and an llm-extra-only `openai` reached master and broke the docs workflow.
# One entry per extras_require group that ships an importable module.
OPTIONAL_AT_IMPORT_TIME = ["openai", "readabilipy", "nimare", "fastapi", "uvicorn"]

_IMPORT_UNDER_BLOCK = """
import sys

BLOCKED = set({blocked!r})


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named %r" % name.split(".")[0])
        return None


sys.meta_path.insert(0, _Blocker())
import autonima.cli  # noqa: F401
"""


@pytest.mark.parametrize("blocked", OPTIONAL_AT_IMPORT_TIME)
def test_cli_imports_without_optional_dependency(blocked):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_UNDER_BLOCK.format(blocked=[blocked])],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"`import autonima.cli` requires {blocked!r}, which the docs build does not install:\n"
        f"{result.stderr}"
    )
