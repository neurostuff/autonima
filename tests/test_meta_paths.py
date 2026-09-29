"""`autonima meta` takes the run folder or its outputs/ folder."""

import pytest

pytest.importorskip("nimare")

from autonima import meta


@pytest.mark.parametrize("pass_outputs", [True, False], ids=["outputs-folder", "run-folder"])
def test_run_meta_analyses_accepts_run_folder_or_outputs(tmp_path, monkeypatch, pass_outputs):
    outputs = tmp_path / "run" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "nimads_studyset.json").write_text("{}")
    (outputs / "nimads_annotation.json").write_text("{}")

    captured = {}
    monkeypatch.setattr(
        meta,
        "run_meta_analyses_from_files",
        lambda **kwargs: captured.update(kwargs) or {},
    )

    # The docs pass <run>/outputs; the web UI passes <run>. Both must land in the same place.
    meta.run_meta_analyses(outputs if pass_outputs else tmp_path / "run")

    assert captured["studyset_file"] == str(outputs / "nimads_studyset.json")
    assert captured["annotation_file"] == str(outputs / "nimads_annotation.json")
    assert captured["output_dir"] == outputs / "meta_analysis_results"
    assert not (outputs / "outputs").exists()


def test_run_meta_analyses_reports_missing_nimads_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="StudySet file not found"):
        meta.run_meta_analyses(tmp_path)
