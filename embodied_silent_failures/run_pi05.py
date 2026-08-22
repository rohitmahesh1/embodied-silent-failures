from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    completion_path,
    prepare_trial,
    write_json_atomic,
)
from embodied_silent_failures.pi05_contract import (
    CHECKPOINT,
    DEFAULT_REPLAN_STEPS,
    DEFAULT_WAIT_STEPS,
    LIBERO_REVISION,
    MAX_STEPS,
    OPENPI_REVISION,
    POLICY_CONFIG,
    validate_replan_steps,
)
from embodied_silent_failures.pi05_supervisor import (
    InfrastructureError,
    PolicyServer,
    append_attempt,
    execute_resilient_plan,
    log_tail,
    require_storage,
    stop_process,
    trial_command,
    unresolved_path,
)
from embodied_silent_failures.plan import (
    Trial,
    build_trial_plan,
    load_trial_manifest,
    parse_task_ids,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    git_state,
    load_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable pi0.5 baseline campaign on LIBERO."
    )
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--libero-python", type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--config", default=POLICY_CONFIG)
    parser.add_argument("--task-suite", choices=sorted(MAX_STEPS), default="libero_10")
    parser.add_argument("--task-ids", default="0-9")
    parser.add_argument("--trial-manifest", type=Path)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-stop", type=int, default=50)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=DEFAULT_WAIT_STEPS)
    parser.add_argument("--replan-steps", type=int, default=DEFAULT_REPLAN_STEPS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--server-start-attempts", type=int, default=2)
    parser.add_argument("--server-startup-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--minimum-free-gb", type=float, default=15.0)
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compare-reference-first-decision",
        action="store_true",
        help="compare official and instrumented samplers on the campaign's first decision",
    )
    args = parser.parse_args()
    args.policy_python = args.policy_python or args.openpi_root / ".venv/bin/python"
    args.libero_python = (
        args.libero_python or args.openpi_root / "examples/libero/.venv/bin/python"
    )
    if args.seed < 0 or args.wait_steps < 0:
        raise ValueError("seed and wait steps must be non-negative")
    validate_replan_steps(args.replan_steps)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.max_attempts <= 0 or args.server_start_attempts <= 0:
        raise ValueError("attempt counts must be positive")
    if args.server_startup_seconds <= 0 or args.poll_seconds <= 0:
        raise ValueError("server startup and poll intervals must be positive")
    if args.minimum_free_gb <= 0:
        raise ValueError("minimum free space must be positive")
    return args


def _trial_plan(args: argparse.Namespace) -> list[Trial]:
    if args.trial_manifest is not None:
        return load_trial_manifest(args.trial_manifest)
    return build_trial_plan(
        parse_task_ids(args.task_ids),
        args.episode_start,
        args.episode_stop,
        args.episode_stride,
    )


def _validate_environment(args: argparse.Namespace) -> None:
    for name, path in {
        "OpenPI root": args.openpi_root,
        "LIBERO root": args.libero_root,
        "policy Python": args.policy_python,
        "LIBERO Python": args.libero_python,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for name, path, revision in (
        ("OpenPI", args.openpi_root, OPENPI_REVISION),
        ("LIBERO", args.libero_root, LIBERO_REVISION),
    ):
        actual = git_revision(path)
        if actual != revision:
            raise RuntimeError(f"{name} revision is {actual}, expected {revision}")
        if git_dirty(path):
            raise RuntimeError(f"{name} has uncommitted changes: {path}")
    project_root = Path(__file__).resolve().parents[1]
    if git_dirty(project_root):
        raise RuntimeError(f"experiment code has uncommitted changes: {project_root}")


def _scientific_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "pi0.5",
        "checkpoint": args.checkpoint,
        "policy_config": args.config,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "wait_steps": args.wait_steps,
        "replan_steps": args.replan_steps,
        "save_video": args.save_video,
        "trial_isolation": "one simulator process and environment per trial",
        "compare_reference_first_decision": (args.compare_reference_first_decision),
        "openpi_root": str(args.openpi_root.resolve()),
        "libero_root": str(args.libero_root.resolve()),
    }


def _code_hashes(project_root: Path) -> dict[str, str]:
    names = (
        "artifacts.py",
        "pi05_contract.py",
        "pi05_policy.py",
        "pi05_rollout.py",
        "pi05_supervisor.py",
        "plan.py",
        "run_pi05.py",
        "run_pi05_trial.py",
        "serve_pi05.py",
    )
    return {
        f"embodied_silent_failures/{name}": file_sha256(
            project_root / "embodied_silent_failures" / name
        )
        for name in names
    }


def _run_metadata(args: argparse.Namespace, plan: list[Trial]) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "condition": "clean",
        "created_at": _now(),
        "configuration": _scientific_configuration(args),
        "trial_count": len(plan),
        "trial_plan": [trial.to_dict() for trial in plan],
        "repository_states": {
            "experiment_code": git_state(project_root),
            "openpi": git_state(args.openpi_root),
            "libero": git_state(args.libero_root),
        },
        "code_sha256": _code_hashes(project_root),
        "trial_manifest_sha256": (
            file_sha256(args.trial_manifest) if args.trial_manifest else None
        ),
        "server_sessions": [],
        "launches": [
            {
                "started_at": _now(),
                "policy_python": str(args.policy_python.resolve()),
                "libero_python": str(args.libero_python.resolve()),
                "max_attempts": args.max_attempts,
                "server_start_attempts": args.server_start_attempts,
                "minimum_free_gb": args.minimum_free_gb,
            }
        ],
    }


def _prepare_run(args: argparse.Namespace, plan: list[Trial]) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "run.json"
    metadata = _run_metadata(args, plan)
    if not path.exists():
        write_json_atomic(path, metadata)
        return metadata
    if not args.resume:
        raise FileExistsError(f"output directory already contains run.json: {path}")
    existing = load_json(path)
    if existing.get("configuration") != metadata["configuration"]:
        raise ValueError("resume configuration does not match the existing run")
    if existing.get("trial_plan") != metadata["trial_plan"]:
        raise ValueError("resume trial plan does not match the existing run")
    existing.setdefault("launches", []).append(metadata["launches"][0])
    write_json_atomic(path, existing)
    return existing


def running_status(
    output_dir: Path,
    planned_trials: int,
    campaign_started: str,
    update: dict[str, Any],
) -> dict[str, Any]:
    trial = update.get("trial")
    trial_progress = update.get("state")
    value = {
        "schema_version": 1,
        "state": "running",
        "started_at": campaign_started,
        "updated_at": _now(),
        "planned_trials": planned_trials,
        "completed_trials": len(list(output_dir.glob("*.complete.json"))),
        "unresolved_trials": len(list(output_dir.glob("*.unresolved.json"))),
        **{key: item for key, item in update.items() if key not in {"state", "trial"}},
    }
    if trial_progress is not None:
        value["trial_progress"] = trial_progress
    if isinstance(trial, Trial):
        value["trial"] = trial.to_dict()
    return value


def main() -> None:
    args = _arguments()
    _validate_environment(args)
    plan = _trial_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "campaign.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another pi0.5 campaign holds the run lock") from error
    _prepare_run(args, plan)

    server = PolicyServer(args)
    active_trial: subprocess.Popen[Any] | None = None
    status_path = args.output_dir / "status.json"
    campaign_started = _now()

    def write_status(update: dict[str, Any]) -> None:
        write_json_atomic(
            status_path,
            running_status(args.output_dir, len(plan), campaign_started, update),
        )

    def already_complete(trial: Trial) -> bool:
        try:
            return prepare_trial(args.output_dir, trial, resume=True) == "complete"
        except Exception as error:
            write_json_atomic(
                unresolved_path(args.output_dir, trial),
                {
                    "schema_version": 1,
                    "status": "unresolved",
                    **trial.to_dict(),
                    "reason": "existing completion record failed validation",
                    "errors": [f"{type(error).__name__}: {error}"],
                    "updated_at": _now(),
                    "counted_as_policy_failure": False,
                },
            )
            return False

    def run_attempt(trial: Trial, attempt: int) -> dict[str, Any]:
        nonlocal active_trial
        free_gb = require_storage(args.output_dir, args.minimum_free_gb)
        server_metadata = server.ensure()
        heartbeat = (
            args.output_dir
            / "heartbeats"
            / (f"task{trial.task_id}--ep{trial.episode_index}.json")
        )
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (
            f"task{trial.task_id}--ep{trial.episode_index}--attempt{attempt}.log"
        )
        compare_reference = args.compare_reference_first_decision and trial == plan[0]
        command = trial_command(
            args, trial, server_metadata, heartbeat, compare_reference
        )
        started_at = _now()
        with log_path.open("ab", buffering=0) as log:
            active_trial = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while active_trial.poll() is None:
                time.sleep(args.poll_seconds)
                heartbeat_value = None
                if heartbeat.is_file():
                    try:
                        heartbeat_value = load_json(heartbeat)
                    except (json.JSONDecodeError, ValueError):
                        heartbeat_value = None
                write_status(
                    {
                        "trial": trial,
                        "attempt": attempt,
                        "trial_state": heartbeat_value,
                        "filesystem_reported_free_gb_before_trial": free_gb,
                    }
                )
            return_code = active_trial.wait()
            active_trial = None
        attempt_record = {
            "attempt": attempt,
            "started_at": started_at,
            "finished_at": _now(),
            "return_code": return_code,
            "log": str(log_path.relative_to(args.output_dir)),
            "server_metadata": str(server_metadata.relative_to(args.output_dir)),
            "server_metadata_sha256": file_sha256(server_metadata),
        }
        if return_code != 0:
            attempt_record["error_tail"] = log_tail(log_path)
            append_attempt(args.output_dir, trial, attempt_record)
            raise RuntimeError(
                f"trial process exited with {return_code}; see {log_path}"
            )
        try:
            if prepare_trial(args.output_dir, trial, resume=True) != "complete":
                raise ValueError(
                    "trial process returned successfully without completion"
                )
        except Exception as error:
            attempt_record["validation_error"] = f"{type(error).__name__}: {error}"
            append_attempt(args.output_dir, trial, attempt_record)
            raise
        append_attempt(args.output_dir, trial, attempt_record)
        unresolved_path(args.output_dir, trial).unlink(missing_ok=True)
        return load_json(completion_path(args.output_dir, trial))

    def mark_unresolved(trial: Trial, errors: list[str]) -> None:
        write_json_atomic(
            unresolved_path(args.output_dir, trial),
            {
                "schema_version": 1,
                "status": "unresolved",
                **trial.to_dict(),
                "attempts_this_launch": len(errors),
                "errors": errors,
                "updated_at": _now(),
                "counted_as_policy_failure": False,
            },
        )

    blocked_error = None
    try:
        results = execute_resilient_plan(
            plan,
            args.max_attempts,
            already_complete,
            run_attempt,
            mark_unresolved,
            write_status,
        )
    except InfrastructureError as error:
        blocked_error = error
        results = {"complete": [], "unresolved": []}
    finally:
        stop_process(active_trial)
        server.stop()

    if blocked_error is not None:
        blocked = {
            "schema_version": 1,
            "state": "blocked",
            "started_at": campaign_started,
            "updated_at": _now(),
            "reason": str(blocked_error),
            "completed_trials": len(list(args.output_dir.glob("*.complete.json"))),
            "unresolved_trials": len(list(args.output_dir.glob("*.unresolved.json"))),
        }
        write_json_atomic(status_path, blocked)
        raise blocked_error

    summary = {
        "schema_version": 1,
        "state": "complete" if not results["unresolved"] else "partial",
        "started_at": campaign_started,
        "finished_at": _now(),
        "planned_trials": len(plan),
        "completed_trials": len(results["complete"]),
        "unresolved_trials": [trial.to_dict() for trial in results["unresolved"]],
        "policy_successes": sum(
            bool(load_json(completion_path(args.output_dir, trial))["success"])
            for trial in results["complete"]
        ),
    }
    write_json_atomic(status_path, summary)
    print(json.dumps(summary, indent=2))
    if summary["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
