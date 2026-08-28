from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.command_interpolation_worker import (
    CONDITION,
    run_planned_interpolations,
)
from embodied_silent_failures.language_campaign import (
    manifest_sha256,
    validate_language_campaign_manifest,
)
from embodied_silent_failures.language_fault import LanguageBlockInjector
from embodied_silent_failures.language_worker import error_record
from embodied_silent_failures.openvla_runtime import (
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_state,
    load_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable state-blocked command interpolation canary."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-campaign-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-shard", required=True, type=int, choices=(0, 1))
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _execution(args: argparse.Namespace) -> dict[str, Any]:
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
        "campaign_manifest_sha256": file_sha256(args.campaign_manifest),
        "interpolation_plan_sha256": file_sha256(args.plan),
        "source_campaign_run_sha256": file_sha256(
            args.source_campaign_dir / "run.json"
        ),
        "runner_sha256": file_sha256(Path(__file__)),
        "interpolation_method_sha256": file_sha256(
            root / "embodied_silent_failures" / "command_interpolation.py"
        ),
        "interpolation_worker_sha256": file_sha256(
            root
            / "embodied_silent_failures"
            / "command_interpolation_worker.py"
        ),
        "language_context_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_context.py"
        ),
        "language_policy_sha256": file_sha256(
            root / "embodied_silent_failures" / "language_policy.py"
        ),
        "worker_shard": args.worker_shard,
    }


def _validate_inputs(
    args: argparse.Namespace,
    plan: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    if plan.get("experiment") != "state-blocked command-boundary interpolation canary":
        raise ValueError("plan is not a command-boundary interpolation canary")
    validate_language_campaign_manifest(manifest)
    source = plan["source"]["campaign_manifest"]
    if source["file_sha256"] != file_sha256(args.campaign_manifest):
        raise ValueError("plan and supplied campaign manifest files differ")
    if source["content_sha256"] != manifest_sha256(manifest):
        raise ValueError("plan and supplied campaign manifest contents differ")
    source_run = load_json(args.source_campaign_dir / "run.json")
    if int(source_run["worker_shard"]) != args.worker_shard:
        raise ValueError("source campaign belongs to another worker shard")
    if (
        source_run["execution"]["manifest_content_sha256"]
        != source["content_sha256"]
    ):
        raise ValueError("source campaign used another context manifest")
    selected = [
        branch
        for branch in plan["branches"]
        if int(branch["worker_shard"]) == args.worker_shard
    ]
    if not selected:
        raise ValueError("plan has no branches for this worker")
    return selected


def _immutable_run_identity(record: dict[str, Any]) -> dict[str, Any]:
    execution = dict(record.get("execution", {}))
    execution.pop("started_at", None)
    return {
        "experiment": record.get("experiment"),
        "condition": record.get("condition"),
        "worker_shard": record.get("worker_shard"),
        "branch_ids": record.get("branch_ids"),
        "planned_terminal_rollouts": record.get("planned_terminal_rollouts"),
        "execution": execution,
    }


def main() -> None:
    args = _arguments()
    for utility in ("git", "nvidia-smi"):
        if shutil.which(utility) is None:
            raise RuntimeError(f"required runtime utility is missing: {utility}")
    config_file = args.libero_config / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"LIBERO configuration is missing: {config_file}")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    plan = load_json(args.plan)
    manifest = load_json(args.campaign_manifest)
    branches = _validate_inputs(args, plan, manifest)
    project_root = Path(__file__).resolve().parents[1]
    validate_pinned_runtime(
        args.checkpoint,
        args.openvla_root,
        args.libero_root,
        project_root=project_root,
    )
    execution = _execution(args)
    run_record = {
        "schema_version": 1,
        "experiment": plan["experiment"],
        "condition": CONDITION,
        "worker_shard": args.worker_shard,
        "branch_ids": [branch["physical_run"] for branch in branches],
        "planned_terminal_rollouts": sum(
            len(branch["lambdas"]) for branch in branches
        ),
        "execution": execution,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    if run_path.exists() and not args.resume:
        raise FileExistsError(f"interpolation output already exists: {run_path}")
    if run_path.exists():
        existing = load_json(run_path)
        if _immutable_run_identity(existing) != _immutable_run_identity(run_record):
            raise ValueError("resume output belongs to another interpolation execution")
    else:
        write_json_atomic(run_path, run_record)

    runtime = load_runtime(args.openvla_root, args.libero_root)
    policy_config = model_config(args.checkpoint, "libero_10")
    runtime.set_seed_everywhere(int(plan["selection"]["seed"]))
    model = runtime.get_model(policy_config)
    model.eval()
    processor = runtime.get_processor(policy_config)
    if "libero_10" not in model.norm_stats:
        policy_config.unnorm_key = "libero_10_no_noops"
    injector = LanguageBlockInjector(runtime.torch)
    injector.install(model)

    suite = runtime.benchmark.get_benchmark_dict()["libero_10"]()
    task_ids = sorted({int(branch["context"]["task_id"]) for branch in branches})
    states = {task_id: suite.get_task_init_states(task_id) for task_id in task_ids}
    environments = {
        task_id: runtime.get_libero_env(
            suite.get_task(task_id), "openvla", resolution=256
        )
        for task_id in task_ids
    }
    counts = Counter()
    started = time.perf_counter()
    try:
        for branch_plan in branches:
            context = branch_plan["context"]
            context_id = str(context["context_id"])
            task_id = int(context["task_id"])
            episode_index = int(context["episode_index"])
            initial_state = states[task_id][episode_index]
            env, task_description = environments[task_id]
            results = None
            context_errors = []
            for attempt in (1, 2):
                try:
                    results = run_planned_interpolations(
                        output_root=args.output_dir,
                        source_campaign_dir=args.source_campaign_dir,
                        wait_steps=args.wait_steps,
                        branch_plan=branch_plan,
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
                    context_errors.append(f"{type(error).__name__}: {error}")
                    write_json_atomic(
                        args.output_dir
                        / "contexts"
                        / context_id
                        / f"context-attempt-{attempt}.error.json",
                        error_record(
                            "command_interpolation_context_exception",
                            error,
                            context=context,
                            attempt=attempt,
                        ),
                    )
                    runtime.torch.cuda.empty_cache()
            if results is not None:
                for result in results:
                    state = (
                        "complete"
                        if result.get("status") == "complete"
                        else "unresolved"
                    )
                    counts[state] += 1
            else:
                counts["context_unresolved"] += 1
                write_json_atomic(
                    args.output_dir
                    / "contexts"
                    / context_id
                    / "context.unresolved.json",
                    {
                        "schema_version": 1,
                        "status": "unresolved",
                        "context": context,
                        "errors": context_errors,
                        "updated_at": _now(),
                    },
                )
            write_json_atomic(
                args.output_dir / "status.json",
                {
                    "schema_version": 1,
                    "state": "running",
                    "worker_shard": args.worker_shard,
                    "counts": dict(sorted(counts.items())),
                    "last_context": context_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_at": _now(),
                },
            )
    finally:
        injector.close()
        for env, _description in environments.values():
            env.close()

    partial = bool(counts["context_unresolved"] or counts["unresolved"])
    final = {
        "schema_version": 1,
        "state": "partial" if partial else "complete",
        "worker_shard": args.worker_shard,
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": _now(),
    }
    write_json_atomic(args.output_dir / "status.json", final)
    print(final, flush=True)


if __name__ == "__main__":
    main()
