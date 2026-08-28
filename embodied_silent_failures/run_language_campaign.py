from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_campaign import (
    manifest_sha256,
    validate_language_campaign_manifest,
)
from embodied_silent_failures.language_fault import LanguageBlockInjector
from embodied_silent_failures.language_worker import error_record, run_context
from embodied_silent_failures.openvla_runtime import (
    array_sha256,
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable OpenVLA language-block residual-risk contexts."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-shard", required=True, type=int, choices=(0, 1))
    parser.add_argument(
        "--context-id",
        action="append",
        default=[],
        help="Run only this manifest context; repeat to select more than one.",
    )
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--maximum-contexts", type=int)
    parser.add_argument("--maximum-faulted-terminal-branches", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _select_contexts(
    manifest: dict[str, Any], worker_shard: int, context_ids: list[str]
) -> list[dict[str, Any]]:
    contexts = [
        context
        for context in manifest["contexts"]
        if int(context["worker_shard"]) == worker_shard
    ]
    if not context_ids:
        return contexts
    if len(set(context_ids)) != len(context_ids):
        raise ValueError("a requested context ID was repeated")
    known = {str(context["context_id"]) for context in manifest["contexts"]}
    unknown = sorted(set(context_ids) - known)
    if unknown:
        raise ValueError(f"requested context IDs are absent from the manifest: {unknown}")
    selected = [
        context for context in contexts if str(context["context_id"]) in context_ids
    ]
    if len(selected) != len(context_ids):
        raise ValueError("a requested context belongs to the other worker shard")
    return selected


def _immutable_run_identity(record: dict[str, Any]) -> dict[str, Any]:
    execution = dict(record.get("execution", {}))
    execution.pop("started_at", None)
    return {
        "campaign": record.get("campaign"),
        "condition": record.get("condition"),
        "worker_shard": record.get("worker_shard"),
        "context_ids": record.get("context_ids"),
        "limits": record.get("limits"),
        "execution": execution,
    }


def _execution(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "started_at": _now(),
        "experiment_code": git_state(root),
        "openvla": git_state(args.openvla_root),
        "libero": git_state(args.libero_root),
        "libero_config": {
            "path": str(args.libero_config.resolve()),
            "sha256": file_sha256(args.libero_config / "config.yaml"),
        },
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_content_sha256": manifest_sha256(manifest),
        "language_campaign_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_campaign.py"
        ),
        "language_context_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_context.py"
        ),
        "language_fault_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_fault.py"
        ),
        "language_policy_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_policy.py"
        ),
        "language_worker_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_worker.py"
        ),
        "worker_shard": args.worker_shard,
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
    libero_config_file = args.libero_config / "config.yaml"
    if not libero_config_file.is_file():
        raise FileNotFoundError(
            f"LIBERO configuration is missing: {libero_config_file}"
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    manifest = load_json(args.manifest)
    validate_language_campaign_manifest(manifest)
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
    sites = {
        (int(site["layer_index"]), int(site["action_token_position"])): site
        for site in manifest["sites"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    execution = _execution(args, manifest)
    run_path = args.output_dir / "run.json"
    run_record = {
        "schema_version": 1,
        "campaign": manifest["campaign"],
        "condition": "activation_fault",
        "worker_shard": args.worker_shard,
        "planned_contexts": len(contexts),
        "context_ids": [str(context["context_id"]) for context in contexts],
        "limits": {
            "maximum_contexts": args.maximum_contexts,
            "maximum_faulted_terminal_branches": (
                args.maximum_faulted_terminal_branches
            ),
        },
        "execution": execution,
    }
    if run_path.exists() and not args.resume:
        raise FileExistsError(f"language campaign output already exists: {run_path}")
    if run_path.exists():
        existing_run = load_json(run_path)
        if _immutable_run_identity(existing_run) != _immutable_run_identity(run_record):
            raise ValueError(
                "resume output belongs to a different language campaign execution"
            )
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
    injector = LanguageBlockInjector(runtime.torch)
    injector.install(model)

    suite = runtime.benchmark.get_benchmark_dict()["libero_10"]()
    states = {
        task_id: suite.get_task_init_states(task_id)
        for task_id in sorted({int(context["task_id"]) for context in contexts})
    }
    environments = {
        task_id: runtime.get_libero_env(
            suite.get_task(task_id), "openvla", resolution=256
        )
        for task_id in states
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
                    initial_hash = array_sha256(runtime, initial_state)
                    if initial_hash != context["initial_state_sha256"]:
                        raise ValueError(f"LIBERO initial state changed for {context_id}")
                    env, task_description = environments[task_id]
                    result = run_context(
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
                        injector=injector,
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
                        error_record(
                            "context_exception",
                            error,
                            context=context,
                            attempt=attempt,
                        ),
                    )
                    if "out of memory" in str(error).lower():
                        consecutive_oom += 1
                    else:
                        consecutive_oom = 0
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
                counts["terminal_unresolved"] += int(result["terminal_unresolved"])
            write_json_atomic(
                args.output_dir / "status.json",
                {
                    "schema_version": 1,
                    "state": "running",
                    "worker_shard": args.worker_shard,
                    "planned_contexts": len(contexts),
                    "processed_contexts": sum(
                        value for key, value in counts.items() if key in {"complete", "unresolved"}
                    ),
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
        injector.close()
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
