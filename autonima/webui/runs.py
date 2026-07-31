"""Run orchestration for Autonima web UI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from autonima.config import ConfigManager
from autonima.execution import complete_execution_progress, preview_execution_changes

from .progress import build_stage_status
from .state import WorkspaceState, utc_now_iso


@dataclass
class ManagedRun:
    """In-memory runtime process state."""

    run_id: str
    metadata: Dict[str, Any]
    process: subprocess.Popen[str]
    log_lines: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_requested: bool = False


class RunManager:
    """Manage subprocess-backed pipeline/meta runs."""

    def __init__(
        self,
        state: WorkspaceState,
        secrets_provider,
    ):
        self.state = state
        self.secrets_provider = secrets_provider
        self._managed: Dict[str, ManagedRun] = {}
        self._lock = threading.Lock()

    def _reconcile_stale_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Mark persisted active-looking runs as canceled when no process exists."""
        status = str(metadata.get("status") or "").lower()
        if status not in {"queued", "running", "canceling"}:
            return metadata

        updated = dict(metadata)
        now = utc_now_iso()
        updated["status"] = "canceled"
        updated["completed_at"] = updated.get("completed_at") or now
        updated["stale_recovered_at"] = now
        updated["status_message"] = (
            "Run process is no longer active; marked canceled after app restart."
        )
        self.state.save_run_metadata(str(updated["id"]), updated)

        output_folder = updated.get("output_folder")
        if output_folder:
            try:
                complete_execution_progress(
                    Path(output_folder),
                    status="canceled",
                    error=updated["status_message"],
                )
            except Exception:
                pass

        return updated

    def _resolve_output_folder(self, config_path: Path, output_folder: Optional[str]) -> Path:
        if output_folder:
            return Path(output_folder).expanduser().resolve()
        return config_path.with_suffix("").resolve()

    def _load_spec_for_run(
        self,
        project: Dict[str, Any],
        run_id: str,
        apply_default_email: bool,
        default_email: Optional[str],
    ) -> Path:
        config_path = Path(project["config_path"]).expanduser().resolve()
        if not apply_default_email or not default_email:
            return config_path

        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return config_path

        search = data.get("search")
        if not isinstance(search, dict):
            return config_path

        existing_email = (search.get("email") or "").strip()
        if existing_email:
            return config_path

        search["email"] = default_email
        data["search"] = search

        temp_config_dir = self.state.paths.state_dir / "temp-configs"
        temp_config_dir.mkdir(parents=True, exist_ok=True)
        temp_config_path = temp_config_dir / f"{run_id}.yaml"
        temp_config_path.write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
        return temp_config_path

    def _build_pipeline_command(
        self,
        mode: str,
        config_path: Path,
        output_folder: Optional[str],
        verbose: bool,
        dry_run: bool,
        debug: bool,
        num_workers: int,
        force_reextract_incomplete_fulltext: bool,
        cache_policy: str,
        clear_cache: List[str],
        copy_valid_cache_from: Optional[str],
    ) -> List[str]:
        cmd = [sys.executable, "-m", "autonima", mode, str(config_path)]
        if output_folder:
            cmd.append(output_folder)
        if verbose:
            cmd.append("--verbose")
        if dry_run:
            cmd.append("--dry-run")
        if debug:
            cmd.append("--debug")
        if num_workers and num_workers > 0:
            cmd.extend(["--num-workers", str(num_workers)])
        if force_reextract_incomplete_fulltext and mode == "run":
            cmd.append("--force-reextract-incomplete-fulltext")
        if cache_policy:
            cmd.extend(["--cache-policy", cache_policy])
        for stage in clear_cache or []:
            cmd.extend(["--clear-cache", stage])
        if copy_valid_cache_from:
            cmd.extend(["--copy-valid-cache-from", copy_valid_cache_from])
        return cmd

    def _maybe_create_execution_output(
        self,
        runtime_config_path: Path,
        resolved_output: Path,
        execution_mode: str,
        cache_source_output: Optional[Path] = None,
    ) -> tuple[Path, Optional[str], Dict[str, Any]]:
        """Default UI behavior: branch to a new output when signatures changed."""
        if execution_mode != "auto_new_on_change":
            return resolved_output, None, {}
        source_output = cache_source_output or resolved_output
        try:
            config = ConfigManager().load_from_file(runtime_config_path)
            config.output.directory = str(source_output)
            preview = preview_execution_changes(config, source_output)
        except Exception:
            return resolved_output, None, {}

        if not preview.get("changed_stages"):
            return source_output, None, preview

        execution_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + "-"
            + str(preview.get("stage_hashes", {}).get("output", ""))[:8]
        )
        branched_output = resolved_output / "executions" / execution_name
        return branched_output, str(source_output), preview

    def _build_meta_command(
        self,
        output_folder: str,
        estimator: str,
        estimator_args: str,
        corrector: str,
        corrector_args: str,
        include_ids: Optional[str],
        run_reports: bool,
        fail_fast: bool,
        debug: bool,
    ) -> List[str]:
        cmd = [
            sys.executable,
            "-m",
            "autonima",
            "meta",
            output_folder,
            "--estimator",
            estimator,
            "--estimator-args",
            estimator_args,
            "--corrector",
            corrector,
            "--corrector-args",
            corrector_args,
        ]
        if include_ids:
            cmd.extend(["--include-ids", include_ids])
        if run_reports:
            cmd.append("--run-reports")
        if fail_fast:
            cmd.append("--fail-fast")
        if debug:
            cmd.append("--debug")
        return cmd

    def _append_log(self, managed: ManagedRun, line: str) -> None:
        cleaned = line.rstrip("\n")
        with managed.lock:
            managed.log_lines.append(cleaned)
            if len(managed.log_lines) > 8000:
                managed.log_lines = managed.log_lines[-8000:]

    def _reader_worker(self, managed: ManagedRun) -> None:
        assert managed.process.stdout is not None
        buffer = ""
        while True:
            chunk = managed.process.stdout.read(1)
            if chunk == "":
                break
            if chunk in {"\n", "\r"}:
                if buffer:
                    self._append_log(managed, buffer)
                    buffer = ""
                continue
            buffer += chunk
        if buffer:
            self._append_log(managed, buffer)

    def _watcher_worker(self, managed: ManagedRun) -> None:
        return_code = managed.process.wait()
        with managed.lock:
            metadata = managed.metadata
            metadata["return_code"] = return_code
            metadata["completed_at"] = utc_now_iso()
            if managed.cancel_requested:
                metadata["status"] = "canceled"
            elif return_code == 0:
                metadata["status"] = "completed"
            else:
                metadata["status"] = "failed"
        self.state.save_run_metadata(managed.run_id, managed.metadata)

    def _start_process(
        self,
        run_id: str,
        metadata: Dict[str, Any],
        cmd: List[str],
        env: Dict[str, str],
    ) -> ManagedRun:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.state.paths.root),
            env=env,
        )

        metadata["status"] = "running"
        metadata["started_at"] = utc_now_iso()
        metadata["command"] = cmd

        managed = ManagedRun(run_id=run_id, metadata=metadata, process=process)
        self.state.save_run_metadata(run_id, metadata)

        reader_thread = threading.Thread(
            target=self._reader_worker,
            args=(managed,),
            daemon=True,
        )
        watcher_thread = threading.Thread(
            target=self._watcher_worker,
            args=(managed,),
            daemon=True,
        )
        reader_thread.start()
        watcher_thread.start()

        with self._lock:
            self._managed[run_id] = managed

        return managed

    def _base_metadata(
        self,
        run_id: str,
        project_id: str,
        run_kind: str,
        mode: str,
        output_folder: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "id": run_id,
            "project_id": project_id,
            "kind": run_kind,
            "mode": mode,
            "status": "queued",
            "created_at": utc_now_iso(),
            "started_at": None,
            "completed_at": None,
            "return_code": None,
            "output_folder": output_folder,
            "command": [],
        }

    def start_pipeline_run(
        self,
        project: Dict[str, Any],
        mode: str,
        output_folder: Optional[str],
        verbose: bool,
        dry_run: bool,
        debug: bool,
        num_workers: int,
        force_reextract_incomplete_fulltext: bool,
        apply_default_email: bool,
        cache_policy: str = "auto",
        clear_cache: Optional[List[str]] = None,
        copy_valid_cache_from: Optional[str] = None,
        execution_mode: str = "auto_new_on_change",
    ) -> Dict[str, Any]:
        run_id = str(uuid4())
        config_path = Path(project["config_path"]).expanduser().resolve()

        secrets = self.secrets_provider()
        runtime_config_path = self._load_spec_for_run(
            project=project,
            run_id=run_id,
            apply_default_email=apply_default_email,
            default_email=secrets.get("NCBI_EMAIL"),
        )

        resolved_output = self._resolve_output_folder(config_path, output_folder)
        cache_preview: Dict[str, Any] = {}
        branched_from: Optional[str] = None
        if not output_folder:
            previous_output_raw = str(project.get("last_output_folder") or "").strip()
            previous_output = (
                Path(previous_output_raw).expanduser().resolve()
                if previous_output_raw
                else None
            )
            if previous_output and not previous_output.exists():
                previous_output = None
            resolved_output, branched_from, cache_preview = (
                self._maybe_create_execution_output(
                    runtime_config_path,
                    resolved_output,
                    execution_mode,
                    cache_source_output=previous_output,
                )
            )
            if branched_from and not copy_valid_cache_from:
                copy_valid_cache_from = branched_from
        metadata = self._base_metadata(
            run_id=run_id,
            project_id=project["id"],
            run_kind="pipeline",
            mode=mode,
            output_folder=str(resolved_output),
        )

        cmd = self._build_pipeline_command(
            mode=mode,
            config_path=runtime_config_path,
            output_folder=str(resolved_output),
            verbose=verbose,
            dry_run=dry_run,
            debug=debug,
            num_workers=num_workers,
            force_reextract_incomplete_fulltext=force_reextract_incomplete_fulltext,
            cache_policy=cache_policy,
            clear_cache=clear_cache or [],
            copy_valid_cache_from=copy_valid_cache_from,
        )
        metadata["cache_preview"] = cache_preview
        metadata["branched_from_output_folder"] = branched_from
        metadata["execution_mode"] = execution_mode

        env = os.environ.copy()
        env.update({k: v for k, v in secrets.items() if v})
        managed = self._start_process(run_id, metadata, cmd, env)

        run_ids = list(project.get("run_ids", []))
        run_ids.append(run_id)
        self.state.update_project(
            project["id"],
            {
                "run_ids": run_ids,
                "last_output_folder": str(resolved_output),
            },
        )

        return managed.metadata

    def start_meta_run(
        self,
        project: Dict[str, Any],
        output_folder: str,
        source_run_id: Optional[str],
        estimator: str,
        estimator_args: str,
        corrector: str,
        corrector_args: str,
        include_ids: Optional[str],
        run_reports: bool,
        fail_fast: bool,
        debug: bool,
    ) -> Dict[str, Any]:
        run_id = str(uuid4())
        output_path = str(Path(output_folder).expanduser().resolve())
        metadata = self._base_metadata(
            run_id=run_id,
            project_id=project["id"],
            run_kind="meta",
            mode="meta",
            output_folder=output_path,
        )
        metadata["source_run_id"] = source_run_id
        metadata["source_output_folder"] = output_path

        cmd = self._build_meta_command(
            output_folder=output_path,
            estimator=estimator,
            estimator_args=estimator_args,
            corrector=corrector,
            corrector_args=corrector_args,
            include_ids=include_ids,
            run_reports=run_reports,
            fail_fast=fail_fast,
            debug=debug,
        )
        env = os.environ.copy()
        env.update({k: v for k, v in self.secrets_provider().items() if v})

        managed = self._start_process(run_id, metadata, cmd, env)

        run_ids = list(project.get("run_ids", []))
        run_ids.append(run_id)
        self.state.update_project(
            project["id"],
            {
                "run_ids": run_ids,
                "last_meta_output_folder": output_path,
            },
        )

        return managed.metadata

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            managed = self._managed.get(run_id)

        if managed is None:
            metadata = self.state.load_run_metadata(run_id)
            if not metadata:
                raise KeyError(f"Run not found: {run_id}")
            return self._reconcile_stale_metadata(metadata)

        if managed.process.poll() is not None:
            return managed.metadata

        managed.cancel_requested = True
        with managed.lock:
            managed.metadata["status"] = "canceling"
            managed.metadata["cancel_requested_at"] = utc_now_iso()
        self.state.save_run_metadata(run_id, managed.metadata)

        managed.process.terminate()

        def _graceful_kill() -> None:
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if managed.process.poll() is not None:
                    return
                time.sleep(0.1)
            if managed.process.poll() is None:
                managed.process.kill()

        threading.Thread(target=_graceful_kill, daemon=True).start()
        return managed.metadata

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            managed = self._managed.get(run_id)

        if managed:
            with managed.lock:
                metadata = dict(managed.metadata)
                logs = list(managed.log_lines)
            self.state.save_run_metadata(run_id, metadata)
        else:
            metadata = self.state.load_run_metadata(run_id)
            if not metadata:
                raise KeyError(f"Run not found: {run_id}")
            metadata = self._reconcile_stale_metadata(metadata)
            logs = []

        progress = build_stage_status(
            run_status=str(metadata.get("status", "")),
            output_folder=metadata.get("output_folder"),
            log_lines=logs,
        )
        metadata["progress"] = progress
        metadata["log_line_count"] = len(logs)
        return metadata

    def get_logs(self, run_id: str, offset: int = 0) -> Dict[str, Any]:
        with self._lock:
            managed = self._managed.get(run_id)

        if managed:
            with managed.lock:
                lines = list(managed.log_lines)
                status = managed.metadata.get("status")
        else:
            metadata = self.state.load_run_metadata(run_id)
            if not metadata:
                raise KeyError(f"Run not found: {run_id}")
            metadata = self._reconcile_stale_metadata(metadata)
            lines = []
            status = metadata.get("status")

        safe_offset = max(0, min(offset, len(lines)))
        new_lines = lines[safe_offset:]
        return {
            "run_id": run_id,
            "offset": safe_offset,
            "next_offset": len(lines),
            "lines": new_lines,
            "completed": status in {"completed", "failed", "canceled"},
        }
