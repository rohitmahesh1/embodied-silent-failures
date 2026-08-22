from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import file_sha256, load_json


class InfrastructureError(RuntimeError):
    """The campaign cannot make useful progress until its environment changes."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_storage(output_dir: Path, minimum_free_gb: float) -> float:
    free_gb = shutil.disk_usage(output_dir).free / 1_000_000_000
    if free_gb < minimum_free_gb:
        raise InfrastructureError(
            f"output volume has {free_gb:.1f} GB free; "
            f"{minimum_free_gb:.1f} GB is required before starting another trial"
        )
    return free_gb


def _health(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/healthz", timeout=2
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _record_server_session(output_dir: Path, record: dict[str, Any]) -> None:
    run_path = output_dir / "run.json"
    metadata = load_json(run_path)
    metadata.setdefault("server_sessions", []).append(record)
    write_json_atomic(run_path, metadata)


class PolicyServer:
    """Own one restartable policy process and its immutable session records."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process: subprocess.Popen[Any] | None = None
        self.metadata_path: Path | None = None
        self.log_file: Any | None = None

    def stop(self) -> None:
        stop_process(self.process)
        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def ready(self) -> bool:
        return (
            self.process is not None
            and self.process.poll() is None
            and self.metadata_path is not None
            and self.metadata_path.is_file()
            and _health(self.args.host, self.args.port)
        )

    def ensure(self) -> Path:
        if self.ready():
            assert self.metadata_path is not None
            return self.metadata_path
        self.stop()
        errors = []
        for attempt in range(1, self.args.server_start_attempts + 1):
            session_id = uuid4().hex
            session_dir = self.args.output_dir / "server-sessions"
            session_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = session_dir / f"{session_id}.json"
            log_path = session_dir / f"{session_id}.log"
            self.log_file = log_path.open("ab", buffering=0)
            command = [
                str(self.args.policy_python),
                "-u",
                "-m",
                "embodied_silent_failures.serve_pi05",
                "--openpi-root",
                str(self.args.openpi_root),
                "--checkpoint",
                self.args.checkpoint,
                "--config",
                self.args.config,
                "--port",
                str(self.args.port),
                "--metadata-output",
                str(metadata_path),
            ]
            started_at = now()
            self.process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.metadata_path = metadata_path
            deadline = time.monotonic() + self.args.server_startup_seconds
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    errors.append(
                        f"attempt {attempt} exited with {self.process.returncode}; "
                        f"see {log_path}"
                    )
                    break
                if metadata_path.is_file() and _health(self.args.host, self.args.port):
                    record = {
                        "session_id": session_id,
                        "started_at": started_at,
                        "ready_at": now(),
                        "metadata": str(
                            metadata_path.relative_to(self.args.output_dir)
                        ),
                        "metadata_sha256": file_sha256(metadata_path),
                        "log": str(log_path.relative_to(self.args.output_dir)),
                    }
                    _record_server_session(self.args.output_dir, record)
                    return metadata_path
                time.sleep(min(self.args.poll_seconds, 5))
            else:
                errors.append(
                    f"attempt {attempt} did not become ready within "
                    f"{self.args.server_startup_seconds} seconds; see {log_path}"
                )
            self.stop()
        raise InfrastructureError("policy server could not start: " + "; ".join(errors))


def unresolved_path(output_dir: Path, trial: Trial) -> Path:
    return output_dir / f"task{trial.task_id}--ep{trial.episode_index}.unresolved.json"


def append_attempt(output_dir: Path, trial: Trial, attempt: dict[str, Any]) -> None:
    directory = output_dir / "attempts"
    path = directory / f"task{trial.task_id}--ep{trial.episode_index}.json"
    if path.exists():
        ledger = load_json(path)
    else:
        ledger = {"schema_version": 1, **trial.to_dict(), "attempts": []}
    ledger["attempts"].append(attempt)
    write_json_atomic(path, ledger)


def log_tail(path: Path, limit: int = 8000) -> str:
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - limit))
        return file.read().decode("utf-8", errors="replace")


def execute_resilient_plan(
    plan: list[Trial],
    max_attempts: int,
    already_complete: Callable[[Trial], bool],
    run_attempt: Callable[[Trial, int], dict[str, Any]],
    mark_unresolved: Callable[[Trial, list[str]], None],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, list[Trial]]:
    """Finish every reachable trial even when individual trials raise errors."""
    complete = []
    unresolved = []
    for index, trial in enumerate(plan, start=1):
        if already_complete(trial):
            complete.append(trial)
            if progress is not None:
                progress({"index": index, "trial": trial, "state": "already_complete"})
            continue
        errors = []
        for attempt in range(1, max_attempts + 1):
            try:
                run_attempt(trial, attempt)
                complete.append(trial)
                if progress is not None:
                    progress({"index": index, "trial": trial, "state": "complete"})
                break
            except Exception as error:
                if isinstance(error, InfrastructureError):
                    raise
                errors.append(f"{type(error).__name__}: {error}")
                if progress is not None:
                    progress(
                        {
                            "index": index,
                            "trial": trial,
                            "state": "retrying"
                            if attempt < max_attempts
                            else "unresolved",
                            "attempt": attempt,
                            "error": errors[-1],
                        }
                    )
        else:
            mark_unresolved(trial, errors)
            unresolved.append(trial)
    return {"complete": complete, "unresolved": unresolved}


def trial_command(
    args: argparse.Namespace,
    trial: Trial,
    server_metadata: Path,
    heartbeat: Path,
    compare_reference: bool,
) -> list[str]:
    command = [
        str(args.libero_python),
        "-u",
        "-m",
        "embodied_silent_failures.run_pi05_trial",
        "--openpi-root",
        str(args.openpi_root),
        "--libero-root",
        str(args.libero_root),
        "--output-dir",
        str(args.output_dir),
        "--server-metadata",
        str(server_metadata),
        "--heartbeat",
        str(heartbeat),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--task-suite",
        args.task_suite,
        "--task-id",
        str(trial.task_id),
        "--episode-index",
        str(trial.episode_index),
        "--seed",
        str(args.seed),
        "--wait-steps",
        str(args.wait_steps),
        "--replan-steps",
        str(args.replan_steps),
        "--save-video" if args.save_video else "--no-save-video",
        "--resume",
    ]
    if compare_reference:
        command.append("--compare-reference-first-decision")
    return command
