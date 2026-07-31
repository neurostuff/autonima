"""Workspace/project persistence for the Autonima web UI."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkspacePaths:
    """Resolved filesystem paths for web UI state."""

    root: Path
    state_dir: Path
    runs_dir: Path
    projects_dir: Path
    workspace_file: Path
    projects_file: Path


class WorkspaceState:
    """Filesystem-backed state for workspace, projects, and runs."""

    def __init__(self, workspace_root: Path):
        self.paths = WorkspacePaths(
            root=workspace_root.resolve(),
            state_dir=(workspace_root / ".autonima-ui").resolve(),
            runs_dir=(workspace_root / ".autonima-ui" / "runs").resolve(),
            projects_dir=(workspace_root / ".autonima-ui" / "projects").resolve(),
            workspace_file=(workspace_root / ".autonima-ui" / "workspace.json").resolve(),
            projects_file=(workspace_root / ".autonima-ui" / "projects.json").resolve(),
        )
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.state_dir.mkdir(parents=True, exist_ok=True)
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.projects_dir.mkdir(parents=True, exist_ok=True)

        if not self.paths.workspace_file.exists():
            self._save_json(
                self.paths.workspace_file,
                {
                    "workspace_root": str(self.paths.root),
                    "state_dir": str(self.paths.state_dir),
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )

        if not self.paths.projects_file.exists():
            self._save_json(
                self.paths.projects_file,
                {
                    "projects": [],
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )

    def _load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)

    def get_workspace(self) -> Dict[str, Any]:
        info = self._load_json(self.paths.workspace_file, {})
        info["workspace_root"] = str(self.paths.root)
        info["state_dir"] = str(self.paths.state_dir)
        return info

    def set_workspace(self, workspace_root: Path) -> Dict[str, Any]:
        self.paths = WorkspaceState(workspace_root).paths
        return self.get_workspace()

    def list_projects(self) -> List[Dict[str, Any]]:
        payload = self._load_json(self.paths.projects_file, {"projects": []})
        projects = payload.get("projects", [])
        for project in projects:
            if "description" not in project:
                project["description"] = ""
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def _save_projects(self, projects: List[Dict[str, Any]]) -> None:
        payload = self._load_json(self.paths.projects_file, {"projects": []})
        payload["projects"] = projects
        payload["updated_at"] = utc_now_iso()
        self._save_json(self.paths.projects_file, payload)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        for project in self.list_projects():
            if project.get("id") == project_id:
                return project
        return None

    def create_project(
        self,
        name: str,
        config_path: Optional[str] = None,
        imported: bool = False,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not name.strip():
            raise ValueError("Project name cannot be empty")

        project_id = str(uuid4())
        project_folder = self.paths.projects_dir / project_id
        project_folder.mkdir(parents=True, exist_ok=True)

        if config_path:
            config_file = Path(config_path).expanduser().resolve()
        else:
            config_file = project_folder / "config.yaml"
            if not config_file.exists():
                config_file.write_text("", encoding="utf-8")

        now = utc_now_iso()
        description_text = (description or "").strip()
        project = {
            "id": project_id,
            "name": name.strip(),
            "description": description_text,
            "config_path": str(config_file),
            "source": "imported" if imported else "created",
            "created_at": now,
            "updated_at": now,
            "last_output_folder": None,
            "run_ids": [],
        }

        projects = self.list_projects()
        projects.append(project)
        self._save_projects(projects)
        return project

    def import_project(
        self,
        config_path: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        project_name = name.strip() if name and name.strip() else config_file.stem
        for project in self.list_projects():
            if Path(project.get("config_path", "")) == config_file:
                return project

        return self.create_project(
            name=project_name,
            config_path=str(config_file),
            imported=True,
            description=description,
        )

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        projects = self.list_projects()
        updated_project: Optional[Dict[str, Any]] = None

        for project in projects:
            if project.get("id") != project_id:
                continue
            project.update(updates)
            project["updated_at"] = utc_now_iso()
            updated_project = project
            break

        if updated_project is None:
            raise KeyError(f"Project not found: {project_id}")

        self._save_projects(projects)
        return updated_project

    def clone_project(
        self,
        project_id: str,
        mode: str = "schema_only",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        allowed_modes = {"schema_only", "schema_and_cached_results"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid clone mode '{mode}'. Expected one of {sorted(allowed_modes)}"
            )

        source_project = self.get_project(project_id)
        if not source_project:
            raise KeyError(f"Project not found: {project_id}")

        source_spec = self.get_project_spec(project_id)
        source_yaml_text = source_spec.get("yaml_text", "")

        clone_name = (name or "").strip() or f'{source_project.get("name", "Project")} copy'
        clone_description = (
            (description or "").strip()
            if description is not None
            else str(source_project.get("description") or "")
        )

        cloned_project = self.create_project(
            name=clone_name,
            imported=False,
            description=clone_description,
        )
        cloned_config_path = Path(cloned_project["config_path"]).expanduser().resolve()
        cloned_config_path.write_text(source_yaml_text, encoding="utf-8")

        cloned_run_ids: List[str] = []
        skipped_active_runs = 0
        if mode == "schema_and_cached_results":
            active_statuses = {"queued", "running", "canceling"}
            source_runs = self.list_runs(project_id=project_id)
            for source_run in source_runs:
                status = str(source_run.get("status") or "").strip().lower()
                if status in active_statuses:
                    skipped_active_runs += 1
                    continue
                new_run_id = str(uuid4())
                cloned_run = dict(source_run)
                cloned_run["id"] = new_run_id
                cloned_run["project_id"] = cloned_project["id"]
                cloned_run["created_at"] = utc_now_iso()
                self.save_run_metadata(new_run_id, cloned_run)
                cloned_run_ids.append(new_run_id)

            self.update_project(
                cloned_project["id"],
                {
                    "run_ids": cloned_run_ids,
                    "last_output_folder": source_project.get("last_output_folder"),
                },
            )
            cloned_project = self.get_project(cloned_project["id"]) or cloned_project

        cloned_project = dict(cloned_project)
        cloned_project["clone_report"] = {
            "mode": mode,
            "cloned_runs_count": len(cloned_run_ids),
            "skipped_active_runs_count": skipped_active_runs,
        }
        return cloned_project

    def get_project_spec(self, project_id: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if not project:
            raise KeyError(f"Project not found: {project_id}")

        config_path = Path(project["config_path"])
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("", encoding="utf-8")

        return {
            "project": project,
            "config_path": str(config_path),
            "yaml_text": config_path.read_text(encoding="utf-8"),
        }

    def save_project_spec(self, project_id: str, yaml_text: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if not project:
            raise KeyError(f"Project not found: {project_id}")

        config_path = Path(project["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml_text, encoding="utf-8")

        return self.update_project(project_id, {})

    def run_metadata_path(self, run_id: str) -> Path:
        return self.paths.runs_dir / f"{run_id}.json"

    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        self._save_json(self.run_metadata_path(run_id), metadata)

    def load_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.run_metadata_path(run_id)
        if not path.exists():
            return None
        return self._load_json(path, None)

    def list_runs(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for run_file in sorted(self.paths.runs_dir.glob("*.json")):
            payload = self._load_json(run_file, None)
            if not payload:
                continue
            if project_id and payload.get("project_id") != project_id:
                continue
            runs.append(payload)
        return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)

    def _is_within_workspace(self, candidate: Path) -> bool:
        workspace_root = self.paths.root.resolve()
        resolved = candidate.expanduser().resolve(strict=False)
        return workspace_root == resolved or workspace_root in resolved.parents

    def _collect_project_output_folders(self, project: Dict[str, Any]) -> List[str]:
        output_folders: List[str] = []
        seen = set()

        for run in self.list_runs(project_id=project["id"]):
            output_folder = (run.get("output_folder") or "").strip()
            if not output_folder or output_folder in seen:
                continue
            seen.add(output_folder)
            output_folders.append(output_folder)

        last_output_folder = (project.get("last_output_folder") or "").strip()
        if last_output_folder and last_output_folder not in seen:
            output_folders.append(last_output_folder)

        return output_folders

    def get_project_delete_preview(self, project_id: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if not project:
            raise KeyError(f"Project not found: {project_id}")

        runs = self.list_runs(project_id=project_id)
        active_statuses = {"queued", "running", "canceling"}
        active_run_ids = [
            run.get("id")
            for run in runs
            if (run.get("status") or "").strip() in active_statuses
        ]

        config_path = Path(project["config_path"]).expanduser().resolve(strict=False)
        output_folders_detected = self._collect_project_output_folders(project)
        output_folders_deletable = [
            folder
            for folder in output_folders_detected
            if self._is_within_workspace(Path(folder))
        ]

        return {
            "project_id": project_id,
            "project_name": project.get("name"),
            "project_source": project.get("source"),
            "config_path": str(config_path),
            "config_deletable": self._is_within_workspace(config_path),
            "run_metadata_count": len(runs),
            "has_active_runs": bool(active_run_ids),
            "active_run_ids": active_run_ids,
            "has_outputs": bool(output_folders_detected),
            "output_folders_detected": output_folders_detected,
            "output_folders_deletable": output_folders_deletable,
        }

    def delete_project(self, project_id: str, mode: str) -> Dict[str, Any]:
        allowed_modes = {
            "metadata_only",
            "metadata_and_config",
            "metadata_config_and_outputs",
        }
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid delete mode '{mode}'. Expected one of {sorted(allowed_modes)}"
            )

        preview = self.get_project_delete_preview(project_id)
        if preview["has_active_runs"]:
            raise RuntimeError(
                "Cannot delete project while runs are active. Cancel or wait for runs to finish."
            )

        project = self.get_project(project_id)
        if not project:
            raise KeyError(f"Project not found: {project_id}")

        # Remove project metadata from projects.json
        projects = self.list_projects()
        remaining_projects = [
            item for item in projects if item.get("id") != project_id
        ]
        self._save_projects(remaining_projects)

        removed_run_metadata_files: List[str] = []
        for run in self.list_runs(project_id=project_id):
            run_id = run.get("id")
            if not run_id:
                continue
            run_path = self.run_metadata_path(str(run_id))
            if run_path.exists():
                run_path.unlink()
                removed_run_metadata_files.append(str(run_path))

        removed_config_path: Optional[str] = None
        skipped_config_reason: Optional[str] = None
        config_path = Path(preview["config_path"])
        include_config = mode in {"metadata_and_config", "metadata_config_and_outputs"}
        if include_config:
            if preview["config_deletable"]:
                if config_path.exists():
                    config_path.unlink()
                    removed_config_path = str(config_path)
            else:
                skipped_config_reason = "Config path is outside workspace boundary"

        removed_output_folders: List[str] = []
        skipped_output_folders: List[Dict[str, str]] = []
        include_outputs = mode == "metadata_config_and_outputs"
        if include_outputs:
            deletable = set(preview["output_folders_deletable"])
            for output_folder in preview["output_folders_detected"]:
                if output_folder not in deletable:
                    skipped_output_folders.append(
                        {
                            "path": output_folder,
                            "reason": "Outside workspace boundary",
                        }
                    )
                    continue

                output_path = Path(output_folder).expanduser().resolve(strict=False)
                if not output_path.exists():
                    continue
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
                removed_output_folders.append(str(output_path))

        return {
            "project_id": project_id,
            "project_name": preview.get("project_name"),
            "mode": mode,
            "removed": {
                "project_metadata": True,
                "run_metadata_files": removed_run_metadata_files,
                "config_path": removed_config_path,
                "output_folders": removed_output_folders,
            },
            "skipped": {
                "config": skipped_config_reason,
                "output_folders": skipped_output_folders,
            },
        }
