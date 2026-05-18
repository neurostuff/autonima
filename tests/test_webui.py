from pathlib import Path

from click.testing import CliRunner

from autonima.cli import ui
from autonima.webui.preferences import PreferencesManager
from autonima.webui.progress import build_stage_status
from autonima.webui.secrets import SecretsManager
from autonima.webui.state import WorkspaceState, utc_now_iso


def test_workspace_state_creates_and_persists_project(tmp_path):
    state = WorkspaceState(tmp_path)

    project = state.create_project("demo-project")
    assert project["name"] == "demo-project"

    spec = state.get_project_spec(project["id"])
    assert spec["config_path"].endswith("config.yaml")

    new_yaml = "search:\n  database: pubmed\n  query: test\n"
    state.save_project_spec(project["id"], new_yaml)

    stored = state.get_project_spec(project["id"])
    assert stored["yaml_text"] == new_yaml


def test_secrets_manager_save_and_mask(tmp_path):
    env_path = tmp_path / "autonima.env"
    manager = SecretsManager(env_path=env_path)

    manager.save(
        {
            "OPENAI_API_KEY": "sk-123456789",
            "OPENAI_API_GATEWAY": "https://example.com/v1",
            "PUBGET_API_KEY": "pg-abcdef",
            "NCBI_EMAIL": "user@example.com",
        }
    )

    loaded = manager.load()
    assert loaded["OPENAI_API_KEY"] == "sk-123456789"
    assert loaded["NCBI_EMAIL"] == "user@example.com"

    masked = manager.load_masked()
    assert masked["OPENAI_API_KEY"].startswith("sk-")
    assert "*" in masked["OPENAI_API_KEY"]


def test_preferences_manager_save_and_normalize(tmp_path):
    preferences_path = tmp_path / "autonima-ui.json"
    manager = PreferencesManager(preferences_path=preferences_path)

    manager.save(
        {
            "preferred_models": [
                " gpt-5-mini-2025-08-07 ",
                "gpt-4o-mini",
                "gpt-4o-mini",
                "",
                123,
            ],
            "default_model": "gpt-4o-mini",
        }
    )

    loaded = manager.load()
    assert loaded["preferred_models"] == ["gpt-5-mini-2025-08-07", "gpt-4o-mini"]
    assert loaded["default_model"] == "gpt-4o-mini"


def test_preferences_manager_clears_default_when_not_in_preferred_list(tmp_path):
    preferences_path = tmp_path / "autonima-ui.json"
    manager = PreferencesManager(preferences_path=preferences_path)

    manager.save(
        {
            "preferred_models": ["gpt-5-mini-2025-08-07"],
            "default_model": "gpt-4o-mini",
        }
    )

    loaded = manager.load()
    assert loaded["preferred_models"] == ["gpt-5-mini-2025-08-07"]
    assert loaded["default_model"] == ""


def test_progress_aggregator_reads_output_files(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "search_results.json").write_text(
        '{"studies": [{"pmid": "1"}, {"pmid": "2"}]}',
        encoding="utf-8",
    )
    (outputs_dir / "abstract_screening_results.json").write_text(
        '{"screening_results": [{"decision": "included_abstract"}, {"decision": "excluded_abstract"}]}',
        encoding="utf-8",
    )
    (outputs_dir / "final_results.json").write_text(
        '{"execution_stats": {"prisma_stats": {"final_included": 1}}}',
        encoding="utf-8",
    )
    (outputs_dir / "nimads_studyset.json").write_text(
        '{"studies": []}',
        encoding="utf-8",
    )

    progress = build_stage_status(
        run_status="running",
        output_folder=str(tmp_path),
        log_lines=["Starting Abstract screening"],
    )

    statuses = {item["stage"]: item["status"] for item in progress["timeline"]}
    assert statuses["search"] == "completed"
    assert statuses["abstract"] == "completed"
    assert progress["counters"]["search"]["studies_found"] == 2
    assert progress["nimads_available"] is True
    assert progress["nimads_studyset_path"] == str(outputs_dir / "nimads_studyset.json")


def test_progress_aggregator_marks_parsing_complete_when_disabled(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "final_results.json").write_text(
        (
            "{"
            '"config": {"parsing": {"parse_coordinates": false}},'
            '"execution_stats": {"prisma_stats": {"final_included": 1}}'
            "}"
        ),
        encoding="utf-8",
    )

    progress = build_stage_status(
        run_status="completed",
        output_folder=str(tmp_path),
        log_lines=["Pipeline completed"],
    )

    statuses = {item["stage"]: item["status"] for item in progress["timeline"]}
    assert statuses["output"] == "completed"
    assert statuses["parsing"] == "completed"


def test_progress_aggregator_extracts_log_issues(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "final_results.json").write_text(
        '{"execution_stats": {"prisma_stats": {"final_included": 1}}}',
        encoding="utf-8",
    )

    progress = build_stage_status(
        run_status="completed",
        output_folder=str(tmp_path),
        log_lines=[
            "2026-01-01 10:00:00 - autonima.pipeline - WARNING - Retrieval source missing index",
            "2026-01-01 10:00:01 - autonima.pipeline - ERROR - Failed to load source path",
            "2026-01-01 10:00:01 - autonima.pipeline - ERROR - Failed to load source path",
        ],
    )

    issues = progress.get("log_issues") or {}
    assert issues.get("error_count") == 1
    assert issues.get("warning_count") == 1
    assert len(issues.get("errors") or []) == 1
    assert len(issues.get("warnings") or []) == 1


def test_progress_aggregator_marks_nimads_export_logged(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "final_results.json").write_text(
        '{"execution_stats": {"prisma_stats": {"final_included": 1}}}',
        encoding="utf-8",
    )

    progress = build_stage_status(
        run_status="completed",
        output_folder=str(tmp_path),
        log_lines=[
            "2026-01-01 10:00:01 - autonima.pipeline - INFO - NiMADS export completed: 8 studies, 12 analyses",
        ],
    )

    assert progress["nimads_export_logged"] is True


def test_progress_aggregator_detects_missing_fulltexts_artifacts(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (outputs_dir / "missing_fulltexts.txt").write_text(
        "PMID_1\nPMID_2\nPMID_3\n",
        encoding="utf-8",
    )
    (outputs_dir / "missing_fulltexts.csv").write_text(
        "pmid,type,source,full_text_path,in_included_set\n"
        "PMID_1,unavailable,pubget,,True\n"
        "PMID_2,incomplete,/tmp/local,/tmp/local/PMID_2.html,False\n",
        encoding="utf-8",
    )

    progress = build_stage_status(
        run_status="running",
        output_folder=str(tmp_path),
        log_lines=[],
    )

    missing = progress["missing_fulltexts"]
    assert missing["available"] is True
    assert missing["count"] == 3
    assert missing["txt_path"] == str(outputs_dir / "missing_fulltexts.txt")
    assert missing["csv_path"] == str(outputs_dir / "missing_fulltexts.csv")
    assert missing["preview_pmids"] == ["PMID_1", "PMID_2", "PMID_3"]
    assert len(missing["preview_rows"]) == 2


def test_progress_aggregator_extracts_tqdm_live_progress(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    progress = build_stage_status(
        run_status="running",
        output_folder=str(tmp_path),
        log_lines=[
            "Starting Abstract screening",
            " 40%|####      | 4/10 [00:03<00:04,  1.25it/s]",
        ],
    )

    live = progress["live_progress"]
    assert live["stage"] == "abstract"
    assert live["current"] == 4
    assert live["total"] == 10
    assert live["percent"] == 40.0

    completed_progress = build_stage_status(
        run_status="completed",
        output_folder=str(tmp_path),
        log_lines=[
            "Starting Abstract screening",
            "100%|##########| 10/10 [00:08<00:00,  1.25it/s]",
        ],
    )
    assert completed_progress["live_progress"] is None


def test_ui_command_invokes_launcher(monkeypatch):
    captured = {}

    def fake_run_ui_server(workspace, host, port, open_browser):
        captured.update(
            {
                "workspace": workspace,
                "host": host,
                "port": port,
                "open_browser": open_browser,
            }
        )

    monkeypatch.setattr("autonima.webui.run_ui_server", fake_run_ui_server)

    result = CliRunner().invoke(
        ui,
        ["--workspace", "/tmp/autonima-ui", "--host", "0.0.0.0", "--port", "8123", "--no-open"],
    )

    assert result.exit_code == 0
    assert captured["workspace"] == "/tmp/autonima-ui"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8123
    assert captured["open_browser"] is False


def test_delete_project_metadata_only_keeps_files(tmp_path):
    state = WorkspaceState(tmp_path)
    project = state.create_project("delete-test")
    project_id = project["id"]

    config_path = Path(project["config_path"])
    config_path.write_text("search:\n  query: test\n", encoding="utf-8")

    output_path = tmp_path / "project-output"
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = "run-1"
    state.save_run_metadata(
        run_id,
        {
            "id": run_id,
            "project_id": project_id,
            "status": "completed",
            "output_folder": str(output_path),
            "created_at": utc_now_iso(),
        },
    )

    report = state.delete_project(project_id, "metadata_only")
    assert report["removed"]["project_metadata"] is True
    assert state.get_project(project_id) is None
    assert config_path.exists()
    assert output_path.exists()
    assert not state.run_metadata_path(run_id).exists()


def test_delete_project_with_outputs_workspace_only(tmp_path):
    state = WorkspaceState(tmp_path)
    outside_config = tmp_path.parent / "external-config.yaml"
    outside_config.write_text("search:\n  query: external\n", encoding="utf-8")

    project = state.create_project(
        name="imported-delete-test",
        config_path=str(outside_config),
        imported=True,
    )
    project_id = project["id"]

    inside_output = tmp_path / "inside-output"
    inside_output.mkdir(parents=True, exist_ok=True)
    outside_output = tmp_path.parent / "outside-output"
    outside_output.mkdir(parents=True, exist_ok=True)

    state.save_run_metadata(
        "run-inside",
        {
            "id": "run-inside",
            "project_id": project_id,
            "status": "completed",
            "output_folder": str(inside_output),
            "created_at": utc_now_iso(),
        },
    )
    state.save_run_metadata(
        "run-outside",
        {
            "id": "run-outside",
            "project_id": project_id,
            "status": "completed",
            "output_folder": str(outside_output),
            "created_at": utc_now_iso(),
        },
    )

    report = state.delete_project(project_id, "metadata_config_and_outputs")
    assert state.get_project(project_id) is None
    assert outside_config.exists()
    assert inside_output.exists() is False
    assert outside_output.exists()
    assert report["skipped"]["config"] == "Config path is outside workspace boundary"
    assert report["skipped"]["output_folders"]


def test_delete_project_blocks_when_active_run_exists(tmp_path):
    state = WorkspaceState(tmp_path)
    project = state.create_project("active-run-project")
    project_id = project["id"]

    state.save_run_metadata(
        "run-active",
        {
            "id": "run-active",
            "project_id": project_id,
            "status": "running",
            "output_folder": str(tmp_path / "active-out"),
            "created_at": utc_now_iso(),
        },
    )

    from pytest import raises

    with raises(RuntimeError):
        state.delete_project(project_id, "metadata_only")


def test_clone_project_schema_only(tmp_path):
    state = WorkspaceState(tmp_path)
    source = state.create_project("source-project", description="source desc")
    source_id = source["id"]
    source_yaml = "search:\n  query: clone me\n"
    state.save_project_spec(source_id, source_yaml)

    cloned = state.clone_project(source_id, mode="schema_only", name="clone-project")
    assert cloned["id"] != source_id
    assert cloned["name"] == "clone-project"
    assert cloned["description"] == "source desc"
    assert state.get_project_spec(cloned["id"])["yaml_text"] == source_yaml
    assert state.list_runs(project_id=cloned["id"]) == []


def test_clone_project_schema_and_cached_results(tmp_path):
    state = WorkspaceState(tmp_path)
    source = state.create_project("source-project")
    source_id = source["id"]
    output_path = tmp_path / "output-cache"
    output_path.mkdir(parents=True, exist_ok=True)

    run_completed = {
        "id": "run-completed",
        "project_id": source_id,
        "kind": "pipeline",
        "mode": "run",
        "status": "completed",
        "created_at": utc_now_iso(),
        "output_folder": str(output_path),
    }
    run_running = {
        "id": "run-running",
        "project_id": source_id,
        "kind": "pipeline",
        "mode": "run",
        "status": "running",
        "created_at": utc_now_iso(),
        "output_folder": str(output_path),
    }
    state.save_run_metadata(run_completed["id"], run_completed)
    state.save_run_metadata(run_running["id"], run_running)
    state.update_project(source_id, {"last_output_folder": str(output_path)})

    cloned = state.clone_project(source_id, mode="schema_and_cached_results")
    cloned_runs = state.list_runs(project_id=cloned["id"])

    assert len(cloned_runs) == 1
    assert cloned_runs[0]["status"] == "completed"
    assert cloned_runs[0]["output_folder"] == str(output_path)
    assert cloned["clone_report"]["cloned_runs_count"] == 1
    assert cloned["clone_report"]["skipped_active_runs_count"] == 1
