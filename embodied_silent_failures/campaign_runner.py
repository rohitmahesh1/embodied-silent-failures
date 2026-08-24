from __future__ import annotations

import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from embodied_silent_failures.artifacts import (
    completion_path,
    prepare_trial,
    write_json_atomic,
)
from embodied_silent_failures.pi05_supervisor import (
    InfrastructureError,
    PolicyServer,
    append_attempt,
    execute_resilient_plan,
    log_tail,
    require_storage,
    stop_process,
    unresolved_path,
)
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import file_sha256, load_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def trial_process_error(
    return_code: int, blocking_exit_codes: tuple[int, ...], log_path: Path
) -> RuntimeError:
    if return_code in blocking_exit_codes:
        return InfrastructureError(
            f"trial process exited with blocking code {return_code}; see {log_path}"
        )
    return RuntimeError(f"trial process exited with {return_code}; see {log_path}")


def run_campaign(
    args: Any,
    plan: list[Trial],
    prepare_run: Callable[[Any, list[Trial]], dict[str, Any]],
    server: PolicyServer,
    trial_command_builder: Callable[
        [Any, Trial, Path, Path, bool], list[str]
    ],
) -> None:
    """Run a process-isolated rollout plan with retry and resume accounting."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "campaign.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another campaign holds the run lock") from error
    prepare_run(args, plan)

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
        write_status(
            {
                "trial": trial,
                "attempt": attempt,
                "state": "starting_server",
                "filesystem_reported_free_gb_before_trial": free_gb,
            }
        )
        server_metadata = server.ensure()
        heartbeat = (
            args.output_dir
            / "heartbeats"
            / f"task{trial.task_id}--ep{trial.episode_index}.json"
        )
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (
            log_dir
            / f"task{trial.task_id}--ep{trial.episode_index}--attempt{attempt}.log"
        )
        compare_reference = args.compare_reference_first_decision and trial == plan[0]
        command = trial_command_builder(
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
            raise trial_process_error(
                return_code,
                getattr(args, "blocking_trial_exit_codes", ()),
                log_path,
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
        write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "state": "blocked",
                "started_at": campaign_started,
                "updated_at": _now(),
                "reason": str(blocked_error),
                "completed_trials": len(
                    list(args.output_dir.glob("*.complete.json"))
                ),
                "unresolved_trials": len(
                    list(args.output_dir.glob("*.unresolved.json"))
                ),
            },
        )
        raise blocked_error

    summary = {
        "schema_version": 1,
        "state": "complete" if not results["unresolved"] else "partial",
        "started_at": campaign_started,
        "finished_at": _now(),
        "planned_trials": len(plan),
        "completed_trials": len(results["complete"]),
        "unresolved_trials": [
            trial.to_dict() for trial in results["unresolved"]
        ],
        "policy_successes": sum(
            bool(load_json(completion_path(args.output_dir, trial))["success"])
            for trial in results["complete"]
        ),
    }
    write_json_atomic(status_path, summary)
    print(json.dumps(summary, indent=2))
    if summary["state"] != "complete":
        raise SystemExit(1)
