import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from autonima.webui.app import create_app
from autonima.webui.state import utc_now_iso


def test_webui_api_project_and_spec_flow(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    workspace = client.get("/api/workspace")
    assert workspace.status_code == 200

    created = client.post("/api/projects", json={"name": "api-test"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    spec = client.get(f"/api/projects/{project_id}/spec")
    assert spec.status_code == 200
    assert "yaml_text" in spec.json()

    updated_spec = "search:\n  database: pubmed\n  query: test query\nretrieval:\n  sources:\n    - pubget\nscreening:\n  abstract:\n    objective: test\n    inclusion_criteria:\n      - one\n  fulltext:\n    objective: test\n    inclusion_criteria:\n      - one\noutput:\n  directory: results\n"
    saved = client.put(
        f"/api/projects/{project_id}/spec",
        json={"yaml_text": updated_spec},
    )
    assert saved.status_code == 200

    validation = client.post(f"/api/projects/{project_id}/validate")
    assert validation.status_code == 200
    assert validation.json().get("ok") in {True, False}


def test_webui_api_project_description_roundtrip(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post(
        "/api/projects",
        json={
            "name": "described-project",
            "description": "A short description for display on cards",
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["description"] == "A short description for display on cards"

    listed = client.get("/api/projects")
    assert listed.status_code == 200
    projects = listed.json().get("projects", [])
    assert any(
        item.get("name") == "described-project"
        and item.get("description") == "A short description for display on cards"
        for item in projects
    )


def test_webui_api_update_project_name_and_description(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post("/api/projects", json={"name": "to-edit"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    updated = client.put(
        f"/api/projects/{project_id}",
        json={"name": "edited-name", "description": "edited description"},
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["name"] == "edited-name"
    assert payload["description"] == "edited description"


def test_webui_api_new_project_spec_starts_blank(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post("/api/projects", json={"name": "blank-project"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    spec = client.get(f"/api/projects/{project_id}/spec")
    assert spec.status_code == 200
    payload = spec.json()
    assert payload["yaml_text"] == ""
    assert payload["form"] == {}


def test_webui_api_secrets_roundtrip(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    write = client.put(
        "/api/settings/secrets",
        json={
            "OPENAI_API_KEY": "sk-test-key",
            "OPENAI_API_GATEWAY": "https://gateway.example/v1",
            "PUBGET_API_KEY": "pubget-xyz",
            "NCBI_EMAIL": "test@example.com",
        },
    )
    assert write.status_code == 200

    read = client.get("/api/settings/secrets")
    assert read.status_code == 200
    values = read.json().get("values", {})
    assert values.get("OPENAI_API_KEY") == "sk-test-key"
    assert values.get("OPENAI_API_GATEWAY") == "https://gateway.example/v1"
    assert values.get("NCBI_EMAIL") == "test@example.com"


def test_webui_api_preferences_roundtrip(tmp_path):
    app = create_app(
        tmp_path,
        env_path=tmp_path / ".autonima.env",
        preferences_path=tmp_path / ".autonima-ui.json",
    )
    client = TestClient(app)

    write = client.put(
        "/api/settings/preferences",
        json={
            "preferred_models": [
                "gpt-5-mini-2025-08-07",
                "gpt-4o-mini",
                "gpt-4o-mini",
                "",
            ],
            "default_model": "gpt-4o-mini",
        },
    )
    assert write.status_code == 200
    assert write.json().get("preferred_models") == [
        "gpt-5-mini-2025-08-07",
        "gpt-4o-mini",
    ]
    assert write.json().get("default_model") == "gpt-4o-mini"

    read = client.get("/api/settings/preferences")
    assert read.status_code == 200
    assert read.json().get("preferred_models") == [
        "gpt-5-mini-2025-08-07",
        "gpt-4o-mini",
    ]
    assert read.json().get("default_model") == "gpt-4o-mini"

    update_default_only = client.put(
        "/api/settings/preferences",
        json={"default_model": "gpt-5-mini-2025-08-07"},
    )
    assert update_default_only.status_code == 200
    assert update_default_only.json().get("default_model") == "gpt-5-mini-2025-08-07"


def test_webui_api_delete_preview_and_delete(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post("/api/projects", json={"name": "delete-api-test"})
    assert created.status_code == 200
    project = created.json()
    project_id = project["id"]

    output_folder = tmp_path / "delete-output"
    output_folder.mkdir(parents=True, exist_ok=True)
    run_path = tmp_path / ".autonima-ui" / "runs" / "run-delete.json"
    run_path.write_text(
        (
            "{\n"
            '  "id": "run-delete",\n'
            f'  "project_id": "{project_id}",\n'
            '  "status": "completed",\n'
            f'  "output_folder": "{str(output_folder)}",\n'
            f'  "created_at": "{utc_now_iso()}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    preview = client.get(f"/api/projects/{project_id}/delete-preview")
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["has_outputs"] is True
    assert payload["run_metadata_count"] == 1
    assert str(output_folder) in payload["output_folders_detected"]

    deleted = client.post(
        f"/api/projects/{project_id}/delete",
        json={"mode": "metadata_config_and_outputs"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["removed"]["project_metadata"] is True
    assert output_folder.exists() is False


def test_webui_api_delete_blocks_on_active_runs(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post("/api/projects", json={"name": "delete-active-test"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    run_path = tmp_path / ".autonima-ui" / "runs" / "run-active.json"
    run_path.write_text(
        (
            "{\n"
            '  "id": "run-active",\n'
            f'  "project_id": "{project_id}",\n'
            '  "status": "running",\n'
            f'  "output_folder": "{str(tmp_path / "active-out")}",\n'
            f'  "created_at": "{utc_now_iso()}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    blocked = client.post(
        f"/api/projects/{project_id}/delete",
        json={"mode": "metadata_only"},
    )
    assert blocked.status_code == 409


def test_webui_api_meta_artifacts_list_and_download(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post("/api/projects", json={"name": "meta-artifacts"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    output_folder = tmp_path / "meta-output"
    meta_dir = output_folder / "outputs" / "meta_analysis_results" / "ale"
    meta_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = meta_dir / "z_desc-association.nii.gz"
    nifti_payload = b"test-nifti-content"
    nifti_path.write_bytes(nifti_payload)
    (meta_dir / "notes.txt").write_text("not an artifact", encoding="utf-8")

    run_id = "run-meta-files"
    run_path = tmp_path / ".autonima-ui" / "runs" / f"{run_id}.json"
    run_path.write_text(
        (
            "{\n"
            f'  "id": "{run_id}",\n'
            f'  "project_id": "{project_id}",\n'
            '  "kind": "meta",\n'
            '  "mode": "meta",\n'
            '  "status": "completed",\n'
            f'  "output_folder": "{str(output_folder)}",\n'
            f'  "created_at": "{utc_now_iso()}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    listed = client.get(f"/api/runs/{run_id}/meta-artifacts")
    assert listed.status_code == 200
    files = listed.json().get("files", [])
    assert [item["relative_path"] for item in files] == ["ale/z_desc-association.nii.gz"]

    downloaded = client.get(
        f"/api/runs/{run_id}/meta-artifact",
        params={"path": "ale/z_desc-association.nii.gz"},
    )
    assert downloaded.status_code == 200
    assert downloaded.content == nifti_payload

    path_route_download = client.get(
        f"/api/runs/{run_id}/meta-artifacts/ale/z_desc-association.nii.gz"
    )
    assert path_route_download.status_code == 200
    assert path_route_download.content == nifti_payload

    unsupported = client.get(f"/api/runs/{run_id}/meta-artifacts/ale/notes.txt")
    assert unsupported.status_code == 404

    invalid_query_path = client.get(
        f"/api/runs/{run_id}/meta-artifact",
        params={"path": "../ale/z_desc-association.nii.gz"},
    )
    assert invalid_query_path.status_code == 400


def test_webui_api_meta_artifacts_missing_output_folder(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    run_id = "run-meta-missing-output"
    run_path = tmp_path / ".autonima-ui" / "runs" / f"{run_id}.json"
    run_path.write_text(
        (
            "{\n"
            f'  "id": "{run_id}",\n'
            '  "project_id": "missing",\n'
            '  "kind": "meta",\n'
            '  "mode": "meta",\n'
            '  "status": "completed",\n'
            f'  "created_at": "{utc_now_iso()}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    listed = client.get(f"/api/runs/{run_id}/meta-artifacts")
    assert listed.status_code == 400


def test_webui_api_missing_fulltexts_download_routes(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post("/api/projects", json={"name": "missing-fulltexts"})
    assert created.status_code == 200
    project_id = created.json()["id"]

    output_folder = tmp_path / "screening-output"
    outputs_dir = output_folder / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    txt_payload = "PMID_1\nPMID_2\n"
    csv_payload = (
        "pmid,type,source,full_text_path,in_included_set\n"
        "PMID_1,unavailable,pubget,,True\n"
    )
    (outputs_dir / "missing_fulltexts.txt").write_text(txt_payload, encoding="utf-8")
    (outputs_dir / "missing_fulltexts.csv").write_text(csv_payload, encoding="utf-8")

    run_id = "run-missing-fulltexts"
    run_path = tmp_path / ".autonima-ui" / "runs" / f"{run_id}.json"
    run_path.write_text(
        (
            "{\n"
            f'  "id": "{run_id}",\n'
            f'  "project_id": "{project_id}",\n'
            '  "kind": "pipeline",\n'
            '  "mode": "run",\n'
            '  "status": "completed",\n'
            f'  "output_folder": "{str(output_folder)}",\n'
            f'  "created_at": "{utc_now_iso()}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    txt_download = client.get(f"/api/runs/{run_id}/missing-fulltexts.txt")
    assert txt_download.status_code == 200
    assert txt_download.text == txt_payload

    csv_download = client.get(f"/api/runs/{run_id}/missing-fulltexts.csv")
    assert csv_download.status_code == 200
    assert csv_download.text == csv_payload

    (outputs_dir / "missing_fulltexts.csv").unlink()
    csv_missing = client.get(f"/api/runs/{run_id}/missing-fulltexts.csv")
    assert csv_missing.status_code == 404


def test_webui_api_clone_project_schema_and_cached_results(tmp_path):
    app = create_app(tmp_path, env_path=tmp_path / ".autonima.env")
    client = TestClient(app)

    created = client.post(
        "/api/projects",
        json={"name": "source-project", "description": "source description"},
    )
    assert created.status_code == 200
    source_id = created.json()["id"]

    spec_yaml = "search:\n  query: copied schema\n"
    saved = client.put(
        f"/api/projects/{source_id}/spec",
        json={"yaml_text": spec_yaml},
    )
    assert saved.status_code == 200

    run_path = tmp_path / ".autonima-ui" / "runs" / "run-clone-source.json"
    output_folder = tmp_path / "clone-source-output"
    output_folder.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        (
            "{\n"
            '  "id": "run-clone-source",\n'
            f'  "project_id": "{source_id}",\n'
            '  "kind": "pipeline",\n'
            '  "mode": "run",\n'
            '  "status": "completed",\n'
            f'  "output_folder": "{str(output_folder)}",\n'
            f'  "created_at": "{utc_now_iso()}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    cloned = client.post(
        f"/api/projects/{source_id}/clone",
        json={"mode": "schema_and_cached_results", "name": "cloned-project"},
    )
    assert cloned.status_code == 200
    clone_payload = cloned.json()
    clone_id = clone_payload["id"]
    assert clone_payload["name"] == "cloned-project"
    assert clone_payload["clone_report"]["cloned_runs_count"] == 1

    clone_spec = client.get(f"/api/projects/{clone_id}/spec")
    assert clone_spec.status_code == 200
    assert clone_spec.json()["yaml_text"] == spec_yaml

    listed_runs = client.get(f"/api/runs?project_id={clone_id}")
    assert listed_runs.status_code == 200
    runs = listed_runs.json().get("runs", [])
    assert len(runs) == 1
    assert runs[0]["output_folder"] == str(output_folder)
