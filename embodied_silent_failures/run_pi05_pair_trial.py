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

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi05_contract import LIBERO_REVISION
from embodied_silent_failures.pi05_pair import (
    PAIR_CONDITIONS,
    PairConfig,
    PrefixTerminated,
    pair_directory,
    prepare_pair,
    run_pair,
)
from embodied_silent_failures.pi05_rollout import array_sha256
from embodied_silent_failures.pi05_stale_manifest import load_manifest
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paired pi0.5 camera trial.")
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--server-metadata", required=True, type=Path)
    parser.add_argument("--stale-manifest", required=True, type=Path)
    parser.add_argument("--heartbeat", required=True, type=Path)
    parser.add_argument("--pair-condition", choices=PAIR_CONDITIONS, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = _arguments()
    trial = Trial(args.task_id, args.episode_index)
    state = prepare_pair(args.output_dir, trial, args.resume)
    if state is not None:
        print(json.dumps({"state": f"already_{state}", **trial.to_dict()}))
        return
    if _git_revision(args.libero_root) != LIBERO_REVISION:
        raise RuntimeError(f"LIBERO must be at {LIBERO_REVISION}: {args.libero_root}")
    manifest = load_manifest(args.stale_manifest)
    if trial not in manifest.specs:
        raise ValueError(f"trial is absent from stale manifest: {trial}")
    spec = manifest.specs[trial]

    sys.path.insert(0, str(args.libero_root))
    sys.path.insert(0, str(args.openpi_root / "packages" / "openpi-client" / "src"))
    import numpy as np
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(trial.task_id)
    initial_states = suite.get_task_init_states(trial.task_id)
    initial_state = initial_states[trial.episode_index]
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
        "stale_manifest_sha256": file_sha256(args.stale_manifest),
        "libero_revision": LIBERO_REVISION,
        "initial_state_sha256": array_sha256(initial_state),
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
        result = run_pair(
            PairConfig(
                output_dir=args.output_dir,
                task_suite=args.task_suite,
                base_seed=args.seed,
                wait_steps=args.wait_steps,
                replan_steps=args.replan_steps,
                save_video=args.save_video,
                pair_condition=args.pair_condition,
            ),
            client,
            trial,
            task,
            initial_state,
            spec,
            execution,
            heartbeat,
        )
    except PrefixTerminated as error:
        directory = pair_directory(args.output_dir, trial)
        directory.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": 1,
            "status": "excluded",
            "reason": "live_common_prefix_terminated_before_intervention",
            "detail": str(error),
            "model": "pi0.5",
            "pair_condition": args.pair_condition,
            **trial.to_dict(),
            "initial_state_sha256": array_sha256(initial_state),
            "intervention_decision": spec.intervention_decision,
            "counted_as_policy_failure": False,
            "execution": execution,
        }
        write_json_atomic(directory / "pair.excluded.json", result)
        heartbeat({"state": "excluded", **trial.to_dict(), "reason": result["reason"]})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
