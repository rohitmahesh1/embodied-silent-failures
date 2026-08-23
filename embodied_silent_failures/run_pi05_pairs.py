from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi05_contract import (
    CHECKPOINT,
    LIBERO_REVISION,
    OPENPI_REVISION,
    POLICY_CONFIG,
)
from embodied_silent_failures.pi05_pair import (
    PAIR_CONDITIONS,
    pair_terminal_state,
    prepare_pair,
)
from embodied_silent_failures.pi05_stale_manifest import load_manifest
from embodied_silent_failures.pi05_supervisor import (
    PolicyServer,
    append_attempt,
    execute_resilient_plan,
    log_tail,
    require_storage,
    stop_process,
    unresolved_path,
)
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_state,
    load_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resilient paired pi0.5 stale-camera or null trials."
    )
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stale-manifest", required=True, type=Path)
    parser.add_argument("--pair-condition", required=True, choices=PAIR_CONDITIONS)
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--libero-python", type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--config", default=POLICY_CONFIG)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--server-start-attempts", type=int, default=2)
    parser.add_argument("--server-startup-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--minimum-free-gb", type=float, default=10.0)
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.policy_python = args.policy_python or args.openpi_root / ".venv/bin/python"
    args.libero_python = (
        args.libero_python or args.openpi_root / "examples/libero/.venv/bin/python"
    )
    if args.seed < 0 or args.wait_steps < 0:
        raise ValueError("seed and wait steps must be non-negative")
    if args.replan_steps != 5:
        raise ValueError("primary pi0.5 stale-camera campaign requires replan_steps=5")
    if args.max_attempts <= 0 or args.server_start_attempts <= 0:
        raise ValueError("attempt counts must be positive")
    return args


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "pi0.5",
        "pair_condition": args.pair_condition,
        "checkpoint": args.checkpoint,
        "policy_config": args.config,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "wait_steps": args.wait_steps,
        "replan_steps": args.replan_steps,
        "save_video": args.save_video,
        "branching": (
            "one live policy prefix, exact action replay into two fresh simulators"
        ),
        "monitor_response": "none; freshness and SAFE are shadow observations",
        "openpi_root": str(args.openpi_root.resolve()),
        "libero_root": str(args.libero_root.resolve()),
    }


def _prepare_run(
    args: argparse.Namespace, plan: list[Trial], source_run_sha256: str
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "run.json"
    project_root = Path(__file__).resolve().parents[1]
    value = {
        "schema_version": 1,
        "condition": args.pair_condition,
        "created_at": _now(),
        "configuration": _configuration(args),
        "trial_count": len(plan),
        "trial_plan": [trial.to_dict() for trial in plan],
        "stale_manifest": str(args.stale_manifest.resolve()),
        "stale_manifest_sha256": file_sha256(args.stale_manifest),
        "source_clean_run_json_sha256": source_run_sha256,
        "repository_states": {
            "experiment_code": git_state(project_root),
            "openpi": git_state(args.openpi_root),
            "libero": git_state(args.libero_root),
        },
        "server_sessions": [],
        "launches": [{"started_at": _now(), "max_attempts": args.max_attempts}],
    }
    if not path.exists():
        write_json_atomic(path, value)
        return value
    if not args.resume:
        raise FileExistsError(f"output directory already contains run.json: {path}")
    existing = load_json(path)
    for key in (
        "configuration",
        "trial_plan",
        "stale_manifest_sha256",
        "source_clean_run_json_sha256",
    ):
        if existing.get(key) != value.get(key):
            raise ValueError(f"resume run disagrees on {key}")
    existing.setdefault("launches", []).append(value["launches"][0])
    write_json_atomic(path, existing)
    return existing


def _trial_command(
    args: argparse.Namespace,
    trial: Trial,
    server_metadata: Path,
    heartbeat: Path,
) -> list[str]:
    return [
        str(args.libero_python),
        "-u",
        "-m",
        "embodied_silent_failures.run_pi05_pair_trial",
        "--openpi-root",
        str(args.openpi_root),
        "--libero-root",
        str(args.libero_root),
        "--output-dir",
        str(args.output_dir),
        "--server-metadata",
        str(server_metadata),
        "--stale-manifest",
        str(args.stale_manifest),
        "--heartbeat",
        str(heartbeat),
        "--pair-condition",
        args.pair_condition,
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


def main() -> None:
    args = _arguments()
    project_root = Path(__file__).resolve().parents[1]
    for name, path, revision in (
        ("OpenPI", args.openpi_root, OPENPI_REVISION),
        ("LIBERO", args.libero_root, LIBERO_REVISION),
    ):
        state = git_state(path)
        if state["revision"] != revision or state["dirty"]:
            raise RuntimeError(f"{name} must be clean and pinned at {revision}")
    if git_dirty(project_root):
        raise RuntimeError("experiment code has uncommitted changes")
    for path in (args.policy_python, args.libero_python, args.stale_manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_manifest(args.stale_manifest)
    plan = sorted(manifest.specs)
    _prepare_run(args, plan, manifest.source_run_json_sha256)
    lock_path = args.output_dir / "campaign.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("another process owns this pi0.5 pair campaign") from error

    status_path = args.output_dir / "status.json"
    campaign_started = _now()
    server = PolicyServer(args)
    active_trial: subprocess.Popen[Any] | None = None

    def status(update: dict[str, Any]) -> None:
        trial = update.get("trial")
        write_json_atomic(
            status_path,
            {
                "schema_version": 1,
                "state": "running",
                "started_at": campaign_started,
                "updated_at": _now(),
                "planned_pairs": len(plan),
                "terminal_pairs": sum(
                    pair_terminal_state(args.output_dir, item) is not None
                    for item in plan
                ),
                "current_trial": trial.to_dict() if isinstance(trial, Trial) else None,
                **{
                    key: value
                    for key, value in update.items()
                    if key != "trial"
                },
            },
        )

    def already_terminal(trial: Trial) -> bool:
        return prepare_pair(args.output_dir, trial, True) is not None

    def run_attempt(trial: Trial, attempt: int) -> dict[str, Any]:
        nonlocal active_trial
        free_gb = require_storage(args.output_dir, args.minimum_free_gb)
        server_metadata = server.ensure()
        heartbeat = (
            args.output_dir
            / "heartbeats"
            / f"task{trial.task_id}--ep{trial.episode_index}.json"
        )
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        log_path = (
            args.output_dir
            / "logs"
            / f"task{trial.task_id}--ep{trial.episode_index}--attempt{attempt}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _trial_command(args, trial, server_metadata, heartbeat)
        started_at = _now()
        with log_path.open("ab", buffering=0) as log:
            active_trial = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while active_trial.poll() is None:
                time.sleep(args.poll_seconds)
                status(
                    {
                        "trial": trial,
                        "attempt": attempt,
                        "filesystem_reported_free_gb_before_trial": free_gb,
                        "trial_state": load_json(heartbeat) if heartbeat.is_file() else None,
                    }
                )
            return_code = active_trial.wait()
            active_trial = None
        record = {
            "attempt": attempt,
            "started_at": started_at,
            "finished_at": _now(),
            "return_code": return_code,
            "log": str(log_path.relative_to(args.output_dir)),
            "server_metadata_sha256": file_sha256(server_metadata),
        }
        if return_code != 0:
            record["error_tail"] = log_tail(log_path)
            append_attempt(args.output_dir, trial, record)
            raise RuntimeError(f"pair process exited with {return_code}; see {log_path}")
        terminal = prepare_pair(args.output_dir, trial, True)
        if terminal is None:
            record["validation_error"] = "pair process wrote no terminal marker"
            append_attempt(args.output_dir, trial, record)
            raise ValueError(record["validation_error"])
        record["terminal_state"] = terminal
        append_attempt(args.output_dir, trial, record)
        unresolved_path(args.output_dir, trial).unlink(missing_ok=True)
        return {"status": terminal}

    def mark_unresolved(trial: Trial, errors: list[str]) -> None:
        write_json_atomic(
            unresolved_path(args.output_dir, trial),
            {
                "schema_version": 1,
                "status": "unresolved",
                **trial.to_dict(),
                "errors": errors,
                "updated_at": _now(),
                "counted_as_policy_failure": False,
            },
        )

    try:
        results = execute_resilient_plan(
            plan,
            args.max_attempts,
            already_terminal,
            run_attempt,
            mark_unresolved,
            status,
        )
    finally:
        stop_process(active_trial)
        server.stop()
        lock.close()

    states = {trial: pair_terminal_state(args.output_dir, trial) for trial in plan}
    summary = {
        "schema_version": 1,
        "state": "complete" if not results["unresolved"] else "partial",
        "started_at": campaign_started,
        "finished_at": _now(),
        "planned_pairs": len(plan),
        "completed_pairs": sum(value == "complete" for value in states.values()),
        "excluded_pairs": sum(value == "excluded" for value in states.values()),
        "unresolved_pairs": [trial.to_dict() for trial in results["unresolved"]],
    }
    write_json_atomic(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
