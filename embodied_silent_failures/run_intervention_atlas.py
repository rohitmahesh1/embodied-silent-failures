from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.atlas_worker import run_atlas_context
from embodied_silent_failures.intervention_atlas import (
    manifest_sha256,
    validate_intervention_atlas_manifest,
)
from embodied_silent_failures.language_worker import error_record
from embodied_silent_failures.openvla_runtime import (
    CHECKPOINT_REVISION,
    array_sha256,
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json
from embodied_silent_failures.temporal_values import TemporalValueCollector


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable graph-derived OpenVLA intervention atlas."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-shard", required=True, type=int)
    parser.add_argument("--context-id", action="append", default=[])
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--maximum-contexts", type=int)
    parser.add_argument("--maximum-faulted-terminal-branches", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _select_contexts(
    manifest: dict[str, Any], worker_shard: int, context_ids: list[str]
) -> list[dict[str, Any]]:
    worker_count = int(manifest["counts"]["worker_count"])
    if not 0 <= worker_shard < worker_count:
        raise ValueError(f"worker shard must be between 0 and {worker_count - 1}")
    selected = [
        context
        for context in manifest["contexts"]
        if int(context["worker_shard"]) == worker_shard
    ]
    if not context_ids:
        return selected
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("a requested context ID was repeated")
    known = {str(context["context_id"]) for context in manifest["contexts"]}
    unknown = sorted(set(context_ids) - known)
    if unknown:
        raise ValueError(f"requested context IDs are absent from the manifest: {unknown}")
    result = [
        context for context in selected if str(context["context_id"]) in context_ids
    ]
    if len(result) != len(context_ids):
        raise ValueError("a requested context belongs to another worker shard")
    return result


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    entries = [
        {
            "path": str(path.relative_to(checkpoint)),
            "size": path.stat().st_size,
            "resolved_name": path.resolve().name,
        }
        for path in sorted(item for item in checkpoint.rglob("*") if item.is_file())
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(entries),
        "basis": "relative path, byte size, and resolved Hugging Face blob name",
    }


def _gpu_record() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _execution(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    import torch

    return {
        "started_at": _now(),
        "experiment_code": git_state(root),
        "openvla": git_state(args.openvla_root),
        "libero": git_state(args.libero_root),
        "libero_config": {
            "path": str(args.libero_config.resolve()),
            "sha256": file_sha256(args.libero_config / "config.yaml"),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "revision": CHECKPOINT_REVISION,
            "manifest": _checkpoint_manifest(args.checkpoint),
        },
        "runtime": {
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpus": _gpu_record(),
            "packages": _package_versions(
                (
                    "accelerate",
                    "bddl",
                    "flash-attn",
                    "huggingface-hub",
                    "libero",
                    "mujoco",
                    "numpy",
                    "robosuite",
                    "safetensors",
                    "torch",
                    "transformers",
                )
            ),
        },
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_content_sha256": manifest_sha256(manifest),
        "implementation_files": {
            path.name: file_sha256(path)
            for path in (
                root / "embodied_silent_failures" / "intervention_atlas.py",
                root / "embodied_silent_failures" / "atlas_context.py",
                root / "embodied_silent_failures" / "atlas_policy.py",
                root / "embodied_silent_failures" / "atlas_worker.py",
                root / "embodied_silent_failures" / "temporal_fault.py",
                root / "embodied_silent_failures" / "temporal_values.py",
            )
        },
        "worker_shard": args.worker_shard,
    }


def _immutable_identity(record: dict[str, Any]) -> dict[str, Any]:
    execution = dict(record.get("execution", {}))
    execution.pop("started_at", None)
    return {
        "campaign": record.get("campaign"),
        "worker_shard": record.get("worker_shard"),
        "context_ids": record.get("context_ids"),
        "limits": record.get("limits"),
        "execution": execution,
    }


def main() -> None:
    args = _arguments()
    if args.maximum_contexts is not None and args.maximum_contexts <= 0:
        raise ValueError("maximum contexts must be positive")
    if (
        args.maximum_faulted_terminal_branches is not None
        and args.maximum_faulted_terminal_branches < 0
    ):
        raise ValueError("maximum faulted terminal branches cannot be negative")
    for utility in ("git", "nvidia-smi"):
        if shutil.which(utility) is None:
            raise RuntimeError(f"required runtime utility is missing: {utility}")
    config_file = args.libero_config / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"LIBERO configuration is missing: {config_file}")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    manifest = load_json(args.manifest)
    validate_intervention_atlas_manifest(manifest)
    project_root = Path(__file__).resolve().parents[1]
    validate_pinned_runtime(
        args.checkpoint,
        args.openvla_root,
        args.libero_root,
        project_root=project_root,
    )
    contexts = _select_contexts(manifest, args.worker_shard, args.context_id)
    if args.maximum_contexts is not None:
        contexts = contexts[: args.maximum_contexts]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution = _execution(args, manifest)
    run_record = {
        "schema_version": 1,
        "campaign": manifest["campaign"],
        "condition": "graph_atlas_temporal_fault",
        "worker_shard": args.worker_shard,
        "planned_contexts": len(contexts),
        "context_ids": [str(context["context_id"]) for context in contexts],
        "limits": {
            "maximum_contexts": args.maximum_contexts,
            "maximum_faulted_terminal_branches": args.maximum_faulted_terminal_branches,
        },
        "execution": execution,
    }
    run_path = args.output_dir / "run.json"
    if run_path.exists() and not args.resume:
        raise FileExistsError(f"atlas output already exists: {run_path}")
    if run_path.exists():
        if _immutable_identity(load_json(run_path)) != _immutable_identity(run_record):
            raise ValueError("resume output belongs to a different atlas execution")
    else:
        write_json_atomic(run_path, run_record)

    runtime = load_runtime(args.openvla_root, args.libero_root)
    policy_config = model_config(args.checkpoint, "libero_10")
    runtime.set_seed_everywhere(int(manifest["seed"]))
    model = runtime.get_model(policy_config)
    model.eval()
    processor = runtime.get_processor(policy_config)
    if "libero_10" not in model.norm_stats:
        policy_config.unnorm_key = "libero_10_no_noops"
    sites = manifest["sites"]
    collector = TemporalValueCollector(runtime.torch, runtime.np, sites)
    collector.install(model)

    suite = runtime.benchmark.get_benchmark_dict()["libero_10"]()
    task_ids = sorted({int(context["task_id"]) for context in contexts})
    states = {task_id: suite.get_task_init_states(task_id) for task_id in task_ids}
    environments = {
        task_id: runtime.get_libero_env(
            suite.get_task(task_id), "openvla", resolution=256
        )
        for task_id in task_ids
    }
    counts = Counter()
    started = time.perf_counter()
    consecutive_oom = 0
    try:
        for index, context in enumerate(contexts, start=1):
            context_id = str(context["context_id"])
            result = None
            errors = []
            for attempt in (1, 2):
                try:
                    task_id = int(context["task_id"])
                    episode_index = int(context["episode_index"])
                    initial_state = states[task_id][episode_index]
                    if array_sha256(runtime, initial_state) != context["initial_state_sha256"]:
                        raise ValueError(f"LIBERO initial state changed for {context_id}")
                    env, task_description = environments[task_id]
                    result = run_atlas_context(
                        output_dir=args.output_dir,
                        wait_steps=args.wait_steps,
                        maximum_faulted_terminal_branches=(
                            args.maximum_faulted_terminal_branches
                        ),
                        context=context,
                        sites=sites,
                        runtime=runtime,
                        policy_config=policy_config,
                        model=model,
                        processor=processor,
                        collector=collector,
                        env=env,
                        task_description=task_description,
                        initial_state=initial_state,
                        execution=execution,
                    )
                    break
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
                    write_json_atomic(
                        args.output_dir
                        / "contexts"
                        / context_id
                        / f"context-attempt-{attempt}.error.json",
                        {
                            **error_record(
                                "context_exception", error, attempt=attempt
                            ),
                            "context": context,
                        },
                    )
                    consecutive_oom = (
                        consecutive_oom + 1
                        if "out of memory" in str(error).lower()
                        else 0
                    )
                    runtime.torch.cuda.empty_cache()
                    if consecutive_oom >= 3:
                        raise RuntimeError("systematic CUDA out-of-memory failures")
            if result is None:
                counts["unresolved"] += 1
                write_json_atomic(
                    args.output_dir / "contexts" / context_id / "context.unresolved.json",
                    {
                        "schema_version": 1,
                        "status": "unresolved",
                        "context": context,
                        "errors": errors,
                        "updated_at": _now(),
                    },
                )
            else:
                consecutive_oom = 0
                counts["complete"] += 1
                counts["local_unresolved"] += int(result["local_unresolved"])
                counts["terminal_unresolved"] += int(result["terminal_unresolved"])
            write_json_atomic(
                args.output_dir / "status.json",
                {
                    "schema_version": 1,
                    "state": "running",
                    "worker_shard": args.worker_shard,
                    "planned_contexts": len(contexts),
                    "processed_contexts": counts["complete"] + counts["unresolved"],
                    "counts": dict(sorted(counts.items())),
                    "last_context": context_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_at": _now(),
                },
            )
            print(
                f"[{index}/{len(contexts)}] {context_id} "
                f"{result['status'] if result else 'unresolved'}",
                flush=True,
            )
    finally:
        collector.close()
        for env, _description in environments.values():
            env.close()

    final = {
        "schema_version": 1,
        "state": "complete" if not counts["unresolved"] else "partial",
        "worker_shard": args.worker_shard,
        "planned_contexts": len(contexts),
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": _now(),
    }
    write_json_atomic(args.output_dir / "status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
