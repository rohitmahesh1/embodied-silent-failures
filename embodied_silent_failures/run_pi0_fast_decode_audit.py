from __future__ import annotations

import argparse
import collections
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi05_supervisor import (
    PolicyServer,
    append_attempt,
    log_tail,
    require_storage,
)
from embodied_silent_failures.pi0_fast_contract import (
    LIBERO_REVISION,
    SAFE_OPENPI_REVISION,
)
from embodied_silent_failures.plan import Trial, load_trial_manifest
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    git_state,
    load_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_path(output_dir: Path, trial: Trial) -> Path:
    return (
        output_dir
        / f"task{trial.task_id}--ep{trial.episode_index}.decode-audit.complete.json"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute a frozen cohort of malformed pi0-FAST decodes."
    )
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--trial-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--libero-python", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--server-start-attempts", type=int, default=2)
    parser.add_argument("--server-startup-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.policy_python = args.policy_python or args.openpi_root / ".venv/bin/python"
    args.libero_python = (
        args.libero_python or args.openpi_root / "examples/libero/.venv/bin/python"
    )
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.max_attempts <= 0 or args.server_start_attempts <= 0:
        raise ValueError("attempt counts must be positive")
    if args.minimum_free_gb <= 0:
        raise ValueError("minimum free space must be positive")
    return args


def _validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    for name, path in {
        "SAFE OpenPI root": args.openpi_root,
        "LIBERO root": args.libero_root,
        "baseline run": args.baseline_run_dir / "run.json",
        "trial manifest": args.trial_manifest,
        "policy Python": args.policy_python,
        "LIBERO Python": args.libero_python,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for name, path, revision in (
        ("SAFE OpenPI", args.openpi_root, SAFE_OPENPI_REVISION),
        ("LIBERO", args.libero_root, LIBERO_REVISION),
    ):
        if git_revision(path) != revision or git_dirty(path):
            raise RuntimeError(f"{name} does not match its clean pinned revision")
    project_root = Path(__file__).resolve().parents[1]
    if git_dirty(project_root):
        raise RuntimeError(f"experiment code has uncommitted changes: {project_root}")

    baseline = load_json(args.baseline_run_dir / "run.json")
    manifest = load_json(args.trial_manifest)
    baseline_hash = file_sha256(args.baseline_run_dir / "run.json")
    if manifest.get("source_campaign_run_json_sha256") != baseline_hash:
        raise ValueError("audit manifest does not identify the supplied baseline run")
    configuration = baseline.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("model") != "pi0-FAST":
        raise ValueError("baseline run is not a pi0-FAST campaign")
    args.checkpoint = configuration["checkpoint"]
    args.config = configuration["policy_config"]
    return baseline, manifest


def _prepare_run(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    manifest: dict[str, Any],
    plan: list[Trial],
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = args.baseline_run_dir / "run.json"
    record = {
        "schema_version": 1,
        "condition": "malformed_decode_attribution",
        "created_at": _now(),
        "selection_rule": manifest.get("selection_rule"),
        "trial_plan": [trial.to_dict() for trial in plan],
        "source_campaign": {
            "directory": str(args.baseline_run_dir.resolve()),
            "run_json_sha256": file_sha256(baseline_path),
            "experiment_revision": baseline.get("repository_states", {})
            .get("experiment_code", {})
            .get("revision"),
        },
        "configuration": {
            key: baseline["configuration"][key]
            for key in (
                "checkpoint",
                "policy_config",
                "task_suite",
                "seed",
                "wait_steps",
                "replan_steps",
            )
        },
        "repository_states": {
            "experiment_code": git_state(project_root),
            "openpi": git_state(args.openpi_root),
            "libero": git_state(args.libero_root),
        },
        "trial_manifest_sha256": file_sha256(args.trial_manifest),
        "server_sessions": [],
    }
    path = args.output_dir / "run.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"audit output already contains run.json: {path}")
        existing = load_json(path)
        for key in (
            "condition",
            "selection_rule",
            "trial_plan",
            "source_campaign",
            "configuration",
            "trial_manifest_sha256",
        ):
            if existing.get(key) != record.get(key):
                raise ValueError(f"audit resume disagrees on {key}")
        return existing
    write_json_atomic(path, record)
    return record


def _valid_audit(path: Path, trial: Trial) -> dict[str, Any]:
    record = load_json(path)
    if (
        record.get("status") != "complete"
        or record.get("condition") != "malformed_decode_attribution"
        or record.get("task_id") != trial.task_id
        or record.get("episode_index") != trial.episode_index
        or not isinstance(record.get("audit"), dict)
    ):
        raise ValueError(f"invalid decode audit marker: {path}")
    return record


def _trial_command(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    trial: Trial,
    server_metadata: Path,
    heartbeat: Path,
) -> list[str]:
    configuration = baseline["configuration"]
    return [
        str(args.libero_python),
        "-u",
        "-m",
        "embodied_silent_failures.run_pi0_fast_trial",
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
        configuration["task_suite"],
        "--task-id",
        str(trial.task_id),
        "--episode-index",
        str(trial.episode_index),
        "--seed",
        str(configuration["seed"]),
        "--wait-steps",
        str(configuration["wait_steps"]),
        "--replan-steps",
        str(configuration["replan_steps"]),
        "--no-save-video",
        "--audit-malformed-decodes",
        "--resume",
    ]


def run_audit(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    plan: list[Trial],
) -> dict[str, Any]:
    server = PolicyServer(
        args,
        "embodied_silent_failures.serve_pi0_fast",
        health_mode="tcp",
    )
    unresolved = []
    try:
        for index, trial in enumerate(plan, start=1):
            marker = audit_path(args.output_dir, trial)
            if marker.is_file():
                _valid_audit(marker, trial)
                continue
            errors = []
            for attempt in range(1, args.max_attempts + 1):
                require_storage(args.output_dir, args.minimum_free_gb)
                metadata = server.ensure()
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
                write_json_atomic(
                    args.output_dir / "status.json",
                    {
                        "schema_version": 1,
                        "state": "running",
                        "updated_at": _now(),
                        "planned_trials": len(plan),
                        "trial_index": index,
                        "attempt": attempt,
                        **trial.to_dict(),
                    },
                )
                started_at = _now()
                with log_path.open("ab", buffering=0) as log:
                    process = subprocess.run(
                        _trial_command(
                            args, baseline, trial, metadata, heartbeat
                        ),
                        cwd=Path(__file__).resolve().parents[1],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                attempt_record = {
                    "attempt": attempt,
                    "started_at": started_at,
                    "finished_at": _now(),
                    "return_code": process.returncode,
                    "log": str(log_path.relative_to(args.output_dir)),
                    "server_metadata": str(metadata.relative_to(args.output_dir)),
                    "server_metadata_sha256": file_sha256(metadata),
                }
                if marker.is_file():
                    _valid_audit(marker, trial)
                    append_attempt(args.output_dir, trial, attempt_record)
                    break
                attempt_record["error_tail"] = log_tail(log_path)
                append_attempt(args.output_dir, trial, attempt_record)
                errors.append(
                    f"attempt {attempt} exited with {process.returncode} without an audit marker"
                )
            else:
                unresolved.append({**trial.to_dict(), "errors": errors})
    finally:
        server.stop()

    records = [
        _valid_audit(audit_path(args.output_dir, trial), trial)
        for trial in plan
        if audit_path(args.output_dir, trial).is_file()
    ]
    classifications = collections.Counter(
        record["audit"]["classification"] for record in records
    )
    summary = {
        "schema_version": 1,
        "state": "complete" if not unresolved else "partial",
        "finished_at": _now(),
        "planned_trials": len(plan),
        "completed_trials": len(records),
        "unresolved_trials": unresolved,
        "classification_counts": dict(sorted(classifications.items())),
    }
    write_json_atomic(args.output_dir / "status.json", summary)
    return summary


def main() -> None:
    args = _arguments()
    baseline, manifest = _validate(args)
    plan = load_trial_manifest(args.trial_manifest)
    _prepare_run(args, baseline, manifest, plan)
    summary = run_audit(args, baseline, plan)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
