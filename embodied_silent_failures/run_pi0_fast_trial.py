from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from embodied_silent_failures.artifacts import prepare_trial, write_json_atomic
from embodied_silent_failures.pi0_fast_contract import LIBERO_REVISION
from embodied_silent_failures.pi0_fast_rollout import (
    EXACT_PARITY_EXIT_CODE,
    DecodeAuditComplete,
    ExactParityError,
    RolloutConfig,
    array_sha256,
    run_trial,
)
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one pinned pi0-FAST trial.")
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--server-metadata", required=True, type=Path)
    parser.add_argument("--heartbeat", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--save-video", dest="save_video", action="store_true")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.set_defaults(save_video=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-reference-first-decision", action="store_true")
    decode_mode = parser.add_mutually_exclusive_group()
    decode_mode.add_argument("--audit-malformed-decodes", action="store_true")
    decode_mode.add_argument("--record-decode-fallbacks", action="store_true")
    return parser.parse_args()


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        universal_newlines=True,
    )
    return result.stdout.strip()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = _arguments()
    trial = Trial(args.task_id, args.episode_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = (
        args.output_dir
        / f"task{trial.task_id}--ep{trial.episode_index}.decode-audit.complete.json"
    )
    if args.audit_malformed_decodes:
        if audit_path.exists():
            if not args.resume:
                raise FileExistsError(f"decode audit already exists: {audit_path}")
            print(json.dumps({"state": "already_complete", **trial.to_dict()}))
            return
    else:
        state = prepare_trial(args.output_dir, trial, args.resume)
        if state == "complete":
            print(json.dumps({"state": "already_complete", **trial.to_dict()}))
            return

    if _git_revision(args.libero_root) != LIBERO_REVISION:
        raise RuntimeError(f"LIBERO must be at {LIBERO_REVISION}: {args.libero_root}")
    sys.path.insert(0, str(args.libero_root))
    sys.path.insert(0, str(args.openpi_root / "packages" / "openpi-client" / "src"))

    import numpy as np
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    if not _is_relative_to(
        Path(benchmark.__file__).resolve(), args.libero_root.resolve()
    ):
        raise RuntimeError("imported LIBERO does not come from --libero-root")

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    if not 0 <= trial.task_id < suite.n_tasks:
        raise ValueError(
            f"task {trial.task_id} is outside suite with {suite.n_tasks} tasks"
        )
    task = suite.get_task(trial.task_id)
    initial_states = suite.get_task_init_states(trial.task_id)
    if not 0 <= trial.episode_index < len(initial_states):
        raise ValueError(
            f"episode {trial.episode_index} is outside the task's "
            f"{len(initial_states)} initial states"
        )

    server_metadata = load_json(args.server_metadata)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    if client.get_server_metadata() != server_metadata:
        raise ValueError("policy server metadata disagrees with its frozen record")
    project_root = Path(__file__).resolve().parents[1]
    execution = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment_revision": _git_revision(project_root),
        "server_metadata": str(args.server_metadata.relative_to(args.output_dir)),
        "server_metadata_sha256": file_sha256(args.server_metadata),
        "libero_revision": LIBERO_REVISION,
        "initial_state_sha256": array_sha256(initial_states[trial.episode_index]),
        "machine": {
            "python": platform.python_version(),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "mujoco", "robosuite", "openpi-client")
        },
    }

    def heartbeat(value):
        write_json_atomic(
            args.heartbeat,
            {**value, "updated_at": datetime.now(timezone.utc).isoformat()},
        )

    try:
        result = run_trial(
            RolloutConfig(
                output_dir=args.output_dir,
                task_suite=args.task_suite,
                base_seed=args.seed,
                wait_steps=args.wait_steps,
                replan_steps=args.replan_steps,
                save_video=args.save_video,
                compare_reference_first_decision=(
                    args.compare_reference_first_decision
                ),
                audit_malformed_decodes=args.audit_malformed_decodes,
                record_decode_fallbacks=args.record_decode_fallbacks,
            ),
            client,
            trial,
            task,
            initial_states[trial.episode_index],
            execution,
            heartbeat,
        )
    except DecodeAuditComplete as complete:
        result = {
            "schema_version": 1,
            "status": "complete",
            "condition": "malformed_decode_attribution",
            "model": "pi0-FAST",
            "task_suite_name": args.task_suite,
            **trial.to_dict(),
            "initial_state_sha256": array_sha256(
                initial_states[trial.episode_index]
            ),
            "replan_steps": args.replan_steps,
            "execution": execution,
            "audit": complete.record,
        }
        write_json_atomic(audit_path, result)
        heartbeat(
            {
                "state": "complete",
                **trial.to_dict(),
                "classification": complete.record["classification"],
            }
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except ExactParityError as error:
        print(f"pi0-FAST parity gate failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(EXACT_PARITY_EXIT_CODE) from error
