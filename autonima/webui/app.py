"""FastAPI app for Autonima local web UI."""

import os
import webbrowser
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import yaml

from autonima.config import ConfigManager, ConfigurationError

from .preferences import PreferencesManager
from .runs import RunManager
from .secrets import SECRETS_KEYS, SecretsManager
from .state import WorkspaceState


def _ensure_fastapi_imports():
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "FastAPI UI dependencies are missing. Install with `pip install -e .[ui]`."
        ) from exc

    return {
        "FastAPI": FastAPI,
        "HTTPException": HTTPException,
        "Query": Query,
        "CORSMiddleware": CORSMiddleware,
        "FileResponse": FileResponse,
        "StaticFiles": StaticFiles,
        "BaseModel": BaseModel,
        "Field": Field,
    }


def _config_to_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Config -> form adapter for wizard UI without synthetic defaults."""
    return dict(config or {})


def _form_to_yaml_text(form: Dict[str, Any]) -> str:
    """Form payload -> YAML text."""
    return yaml.safe_dump(form, sort_keys=False)


def create_app(
    workspace_root: Path,
    env_path: Optional[Path] = None,
    preferences_path: Optional[Path] = None,
):
    """Create FastAPI app instance."""
    deps = _ensure_fastapi_imports()
    FastAPI = deps["FastAPI"]
    HTTPException = deps["HTTPException"]
    Query = deps["Query"]
    CORSMiddleware = deps["CORSMiddleware"]
    FileResponse = deps["FileResponse"]
    StaticFiles = deps["StaticFiles"]
    BaseModel = deps["BaseModel"]
    Field = deps["Field"]

    class WorkspaceUpdate(BaseModel):
        workspace_root: str

    class ProjectCreate(BaseModel):
        name: Optional[str] = None
        config_path: Optional[str] = None
        description: Optional[str] = None

    class ProjectImport(BaseModel):
        config_path: str
        name: Optional[str] = None
        description: Optional[str] = None

    class ProjectUpdate(BaseModel):
        name: Optional[str] = None
        description: Optional[str] = None

    class ProjectCloneRequest(BaseModel):
        mode: str = "schema_only"
        name: Optional[str] = None
        description: Optional[str] = None

    class SpecUpdate(BaseModel):
        yaml_text: Optional[str] = None
        form: Optional[Dict[str, Any]] = None

    class PipelineRunRequest(BaseModel):
        mode: str = Field(default="run")
        output_folder: Optional[str] = None
        verbose: bool = False
        dry_run: bool = False
        debug: bool = False
        num_workers: int = 1
        force_reextract_incomplete_fulltext: bool = False
        apply_default_email: bool = True

    class MetaRunRequest(BaseModel):
        output_folder: str
        estimator: str = "mkdadensity"
        estimator_args: str = "{}"
        corrector: str = "fdr"
        corrector_args: str = "{}"
        include_ids: Optional[str] = None
        run_reports: bool = False
        fail_fast: bool = False
        debug: bool = False

    class SecretsUpdate(BaseModel):
        OPENAI_API_KEY: Optional[str] = None
        OPENAI_API_GATEWAY: Optional[str] = None
        PUBGET_API_KEY: Optional[str] = None
        NCBI_EMAIL: Optional[str] = None

    class PreferencesUpdate(BaseModel):
        preferred_models: Optional[list[str]] = None
        default_model: Optional[str] = None

    class ProjectDeleteRequest(BaseModel):
        mode: str

    state = WorkspaceState(workspace_root)
    secrets = SecretsManager(env_path=env_path)
    preferences = PreferencesManager(preferences_path=preferences_path)
    run_manager = RunManager(state=state, secrets_provider=secrets.load)

    app = FastAPI(title="Autonima UI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _is_supported_meta_artifact(path: Path) -> bool:
        lower_name = path.name.lower()
        return lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")

    def _resolve_run_output_folder(run_id: str) -> tuple[Dict[str, Any], Path]:
        metadata = state.load_run_metadata(run_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        output_folder_raw = str(metadata.get("output_folder") or "").strip()
        if not output_folder_raw:
            raise HTTPException(
                status_code=400,
                detail="Run does not declare an output_folder",
            )

        output_folder = Path(output_folder_raw).expanduser().resolve(strict=False)
        if not output_folder.exists() or not output_folder.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Output folder not found: {output_folder}",
            )

        return metadata, output_folder

    def _resolve_meta_results_root(output_folder: Path) -> Path:
        candidates = [
            output_folder / "outputs" / "meta_analysis_results",
            output_folder / "meta_analysis_results",
            output_folder,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return output_folder

    def _collect_meta_artifacts(artifacts_root: Path) -> list[Dict[str, Any]]:
        files: list[Dict[str, Any]] = []
        for path in artifacts_root.rglob("*"):
            if not path.is_file() or not _is_supported_meta_artifact(path):
                continue
            relative_path = path.relative_to(artifacts_root).as_posix()
            files.append(
                {
                    "name": path.name,
                    "relative_path": relative_path,
                    "size_bytes": path.stat().st_size,
                    "absolute_path": path,
                }
            )
        files.sort(key=lambda item: item.get("relative_path", ""))
        return files

    def _normalize_relative_artifact_path(raw_path: str) -> str:
        text = str(raw_path or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Artifact path is required")
        cleaned = PurePosixPath(text).as_posix().lstrip("/")
        if cleaned in {"", "."}:
            raise HTTPException(status_code=400, detail="Artifact path is required")
        if cleaned.startswith("../") or "/../" in cleaned or cleaned == "..":
            raise HTTPException(status_code=400, detail="Invalid artifact path")
        return cleaned

    def _find_meta_artifact_path(run_id: str, relative_path: str) -> tuple[Dict[str, Any], Path, Path, Dict[str, Any]]:
        metadata, output_folder = _resolve_run_output_folder(run_id)
        artifacts_root = _resolve_meta_results_root(output_folder)
        normalized_relative_path = _normalize_relative_artifact_path(relative_path)
        files = _collect_meta_artifacts(artifacts_root)
        artifact = next(
            (item for item in files if item.get("relative_path") == normalized_relative_path),
            None,
        )
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        absolute_path = artifact.get("absolute_path")
        if not isinstance(absolute_path, Path) or not absolute_path.exists() or not absolute_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return metadata, output_folder, artifacts_root, artifact

    def _resolve_missing_fulltexts_artifact_path(run_id: str, filename: str) -> tuple[Dict[str, Any], Path]:
        if filename not in {"missing_fulltexts.txt", "missing_fulltexts.csv"}:
            raise HTTPException(status_code=400, detail="Unsupported artifact")
        metadata, output_folder = _resolve_run_output_folder(run_id)
        artifact_path = output_folder / "outputs" / filename
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return metadata, artifact_path

    @app.get("/")
    async def root():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/workspace")
    async def get_workspace():
        info = state.get_workspace()
        info["projects_count"] = len(state.list_projects())
        return info

    @app.post("/api/workspace")
    async def set_workspace(payload: WorkspaceUpdate):
        nonlocal state, run_manager
        new_state = WorkspaceState(Path(payload.workspace_root).expanduser())
        state = new_state
        run_manager = RunManager(state=state, secrets_provider=secrets.load)
        return state.get_workspace()

    @app.get("/api/projects")
    async def list_projects():
        return {"projects": state.list_projects()}

    @app.post("/api/projects")
    async def create_project(payload: ProjectCreate):
        try:
            if payload.config_path:
                project = state.import_project(
                    payload.config_path,
                    payload.name,
                    payload.description,
                )
            else:
                if not payload.name:
                    raise ValueError(
                        "Provide `name` for a new project or `config_path` for import."
                    )
                project = state.create_project(
                    name=payload.name,
                    description=payload.description,
                )
            return state.get_project(project["id"])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/import")
    async def import_project(payload: ProjectImport):
        try:
            project = state.import_project(
                payload.config_path,
                payload.name,
                payload.description,
            )
            return project
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str):
        project = state.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    @app.put("/api/projects/{project_id}")
    async def update_project(project_id: str, payload: ProjectUpdate):
        try:
            updates: Dict[str, Any] = {}
            if payload.name is not None:
                name = payload.name.strip()
                if not name:
                    raise ValueError("Project name cannot be empty")
                updates["name"] = name
            if payload.description is not None:
                updates["description"] = payload.description.strip()
            if not updates:
                raise ValueError("No updates provided")
            return state.update_project(project_id, updates)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/clone")
    async def clone_project(project_id: str, payload: ProjectCloneRequest):
        try:
            return state.clone_project(
                project_id=project_id,
                mode=payload.mode,
                name=payload.name,
                description=payload.description,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/projects/{project_id}/clone")
    async def clone_project_put(project_id: str, payload: ProjectCloneRequest):
        return await clone_project(project_id, payload)

    @app.get("/api/projects/{project_id}/delete-preview")
    async def get_project_delete_preview(project_id: str):
        try:
            return state.get_project_delete_preview(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/delete")
    async def delete_project(project_id: str, payload: ProjectDeleteRequest):
        try:
            report = state.delete_project(project_id, payload.mode)
            return report
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/spec")
    async def get_project_spec(project_id: str):
        try:
            spec_payload = state.get_project_spec(project_id)
            yaml_text = spec_payload["yaml_text"]
            config_dict = yaml.safe_load(yaml_text) or {}
            return {
                "project_id": project_id,
                "config_path": spec_payload["config_path"],
                "yaml_text": yaml_text,
                "form": _config_to_form(config_dict),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/projects/{project_id}/spec")
    async def put_project_spec(project_id: str, payload: SpecUpdate):
        if payload.yaml_text is None and payload.form is None:
            raise HTTPException(status_code=400, detail="Provide yaml_text or form")

        yaml_text = payload.yaml_text
        if yaml_text is None:
            yaml_text = _form_to_yaml_text(payload.form or {})

        try:
            state.save_project_spec(project_id, yaml_text)
            config_dict = yaml.safe_load(yaml_text) or {}
            return {
                "project_id": project_id,
                "yaml_text": yaml_text,
                "form": _config_to_form(config_dict),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/validate")
    async def validate_spec(project_id: str):
        try:
            spec_payload = state.get_project_spec(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        yaml_text = spec_payload["yaml_text"]
        try:
            config_dict = yaml.safe_load(yaml_text)
            if config_dict is None:
                raise ConfigurationError("Configuration file is empty")
            manager = ConfigManager()
            validated = manager.load_from_dict(config_dict)
            config_out = validated.to_dict()
            return {
                "ok": True,
                "message": "Configuration is valid",
                "config": config_out,
            }
        except ConfigurationError as exc:
            return {
                "ok": False,
                "message": str(exc),
            }
        except Exception as exc:
            return {
                "ok": False,
                "message": f"Unexpected validation error: {exc}",
            }

    @app.post("/api/projects/{project_id}/runs")
    async def start_pipeline_run(project_id: str, payload: PipelineRunRequest):
        project = state.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        allowed_modes = {"run", "run-search", "run-abstract"}
        if payload.mode not in allowed_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{payload.mode}'. Expected one of {sorted(allowed_modes)}",
            )

        try:
            metadata = run_manager.start_pipeline_run(
                project=project,
                mode=payload.mode,
                output_folder=payload.output_folder,
                verbose=payload.verbose,
                dry_run=payload.dry_run,
                debug=payload.debug,
                num_workers=payload.num_workers,
                force_reextract_incomplete_fulltext=payload.force_reextract_incomplete_fulltext,
                apply_default_email=payload.apply_default_email,
            )
            return metadata
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/meta-runs")
    async def start_meta_run(project_id: str, payload: MetaRunRequest):
        project = state.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        try:
            metadata = run_manager.start_meta_run(
                project=project,
                output_folder=payload.output_folder,
                estimator=payload.estimator,
                estimator_args=payload.estimator_args,
                corrector=payload.corrector,
                corrector_args=payload.corrector_args,
                include_ids=payload.include_ids,
                run_reports=payload.run_reports,
                fail_fast=payload.fail_fast,
                debug=payload.debug,
            )
            return metadata
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    async def list_runs(project_id: Optional[str] = None):
        runs = state.list_runs(project_id=project_id)
        enriched = []
        for run in runs:
            try:
                enriched.append(run_manager.get_run(run["id"]))
            except Exception:
                enriched.append(run)
        return {"runs": enriched}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        try:
            return run_manager.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/logs")
    async def get_run_logs(run_id: str, offset: int = Query(default=0, ge=0)):
        try:
            return run_manager.get_logs(run_id, offset=offset)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/meta-artifacts")
    async def list_meta_artifacts(run_id: str):
        metadata, output_folder = _resolve_run_output_folder(run_id)
        artifacts_root = _resolve_meta_results_root(output_folder)
        files = _collect_meta_artifacts(artifacts_root)
        return {
            "run_id": run_id,
            "run_kind": metadata.get("kind"),
            "run_status": metadata.get("status"),
            "artifacts_root": str(artifacts_root),
            "files": [
                {
                    "name": item.get("name"),
                    "relative_path": item.get("relative_path"),
                    "size_bytes": item.get("size_bytes"),
                }
                for item in files
            ],
        }

    @app.get("/api/runs/{run_id}/meta-artifact")
    async def get_meta_artifact_by_query(run_id: str, path: str = Query(default="")):
        _, _, _, artifact = _find_meta_artifact_path(run_id, path)
        absolute_path = artifact["absolute_path"]
        return FileResponse(absolute_path, filename=str(artifact.get("name") or absolute_path.name))

    @app.get("/api/runs/{run_id}/meta-artifacts/{artifact_path:path}")
    async def get_meta_artifact(run_id: str, artifact_path: str):
        _, _, _, artifact = _find_meta_artifact_path(run_id, artifact_path)
        absolute_path = artifact["absolute_path"]
        return FileResponse(absolute_path, filename=str(artifact.get("name") or absolute_path.name))

    @app.get("/api/runs/{run_id}/missing-fulltexts.txt")
    async def get_missing_fulltexts_txt(run_id: str):
        _, artifact_path = _resolve_missing_fulltexts_artifact_path(run_id, "missing_fulltexts.txt")
        return FileResponse(artifact_path, filename="missing_fulltexts.txt")

    @app.get("/api/runs/{run_id}/missing-fulltexts.csv")
    async def get_missing_fulltexts_csv(run_id: str):
        _, artifact_path = _resolve_missing_fulltexts_artifact_path(run_id, "missing_fulltexts.csv")
        return FileResponse(artifact_path, filename="missing_fulltexts.csv")

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        try:
            return run_manager.cancel_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/settings/secrets")
    async def get_secrets():
        values = secrets.load()
        masked = secrets.load_masked()
        return {
            "masked": masked,
            "values": {key: values.get(key, "") for key in SECRETS_KEYS},
        }

    @app.put("/api/settings/secrets")
    async def put_secrets(payload: SecretsUpdate):
        updates = payload.model_dump()
        saved = secrets.save(updates)
        return {
            "saved": {key: saved.get(key, "") for key in SECRETS_KEYS},
            "masked": secrets.load_masked(),
        }

    @app.get("/api/settings/preferences")
    async def get_preferences():
        return preferences.load()

    @app.put("/api/settings/preferences")
    async def put_preferences(payload: PreferencesUpdate):
        updates = payload.model_dump(exclude_unset=True)
        return preferences.save(updates)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(static_dir / "index.html")

    return app


def run_ui_server(
    workspace: Optional[str],
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Run the local FastAPI UI server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Uvicorn is required for `autonima ui`. Install with `pip install -e .[ui]`."
        ) from exc

    workspace_path = Path(workspace).expanduser() if workspace else Path.cwd()
    app = create_app(workspace_path)

    if open_browser:
        webbrowser.open(f"http://{host}:{port}", new=1)

    uvicorn.run(app, host=host, port=port, log_level=os.getenv("AUTONIMA_UI_LOG_LEVEL", "info"))
