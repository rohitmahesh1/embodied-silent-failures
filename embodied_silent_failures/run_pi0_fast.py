from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.campaign_runner import run_campaign
from embodied_silent_failures.pi05_supervisor import PolicyServer
from embodied_silent_failures.pi0_fast_contract import (
    CHECKPOINT,
    DEFAULT_REPLAN_STEPS,
    DEFAULT_WAIT_STEPS,
    FAST_TOKENIZER_REVISION,
    LIBERO_REVISION,
    MAX_STEPS,
    POLICY_CONFIG,
    SAFE_OPENPI_PARENT_REVISION,
    SAFE_OPENPI_REVISION,
    SAFE_REVISION,
    validate_replan_steps,
)
from embodied_silent_failures.pi0_fast_rollout import (
    EXACT_PARITY_EXIT_CODE,
    FALLBACK_CONTINUATION_CONDITION,
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
        description="Run the pinned pi0-FAST baseline on LIBERO."
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
        "--record-decode-fallbacks",
        action="store_true",
        help="record FAST decoder fallbacks and execute its returned actions",
    )
    parser.add_argument(
        "--compare-reference-first-decision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the instrumented first decision to equal the parent sampler",
    )
    args = parser.parse_args()
    args.policy_python = args.policy_python or args.openpi_root / ".venv/bin/python"
    args.libero_python = (
        args.libero_python or args.openpi_root / "examples/libero/.venv/bin/python"
    )
    args.blocking_trial_exit_codes = (EXACT_PARITY_EXIT_CODE,)
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
    task_major = build_trial_plan(
        parse_task_ids(args.task_ids),
        args.episode_start,
        args.episode_stop,
        args.episode_stride,
    )
    # Trial processes are isolated and each initial state is fixed, so execution
    # order is not part of the scientific condition. Episode-major order covers
    # every LIBERO task before spending the next rollout on any one task.
    return sorted(task_major, key=lambda trial: (trial.episode_index, trial.task_id))


def _validate_environment(args: argparse.Namespace) -> None:
    for name, path in {
        "SAFE OpenPI root": args.openpi_root,
        "LIBERO root": args.libero_root,
        "policy Python": args.policy_python,
        "LIBERO Python": args.libero_python,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    for name, path, revision in (
        ("SAFE OpenPI", args.openpi_root, SAFE_OPENPI_REVISION),
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
        "model": "pi0-FAST",
        "checkpoint": args.checkpoint,
        "policy_config": args.config,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "wait_steps": args.wait_steps,
        "replan_steps": args.replan_steps,
        "save_video": args.save_video,
        "trial_isolation": "one simulator process and environment per trial",
        "exact_parent_parity_on_first_decision": (
            args.compare_reference_first_decision
        ),
        "record_decode_fallbacks": args.record_decode_fallbacks,
        "openpi_root": str(args.openpi_root.resolve()),
        "libero_root": str(args.libero_root.resolve()),
    }


def _code_hashes(project_root: Path) -> dict[str, str]:
    names = (
        "artifacts.py",
        "campaign_runner.py",
        "pi05_supervisor.py",
        "pi0_fast_contract.py",
        "pi0_fast_policy.py",
        "pi0_fast_rollout.py",
        "plan.py",
        "run_pi0_fast.py",
        "run_pi0_fast_trial.py",
        "serve_pi0_fast.py",
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
        "condition": (
            FALLBACK_CONTINUATION_CONDITION
            if args.record_decode_fallbacks
            else "clean"
        ),
        "created_at": _now(),
        "configuration": _scientific_configuration(args),
        "trial_count": len(plan),
        "trial_plan": [trial.to_dict() for trial in plan],
        "pinned_sources": {
            "safe_openpi_revision": SAFE_OPENPI_REVISION,
            "safe_openpi_parent_revision": SAFE_OPENPI_PARENT_REVISION,
            "safe_revision": SAFE_REVISION,
            "fast_tokenizer_revision": FAST_TOKENIZER_REVISION,
        },
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


def _trial_command(
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
    if args.record_decode_fallbacks:
        command.append("--record-decode-fallbacks")
    return command


def main() -> None:
    args = _arguments()
    _validate_environment(args)
    plan = _trial_plan(args)
    run_campaign(
        args,
        plan,
        _prepare_run,
        PolicyServer(
            args,
            "embodied_silent_failures.serve_pi0_fast",
            health_mode="tcp",
        ),
        _trial_command,
    )


if __name__ == "__main__":
    main()
