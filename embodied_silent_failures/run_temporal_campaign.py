from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    completion_path,
    prepare_trial,
    write_json_atomic,
)
from embodied_silent_failures.openvla_rollout import (
    RolloutConfig,
    TemporalPrefixDivergence,
    run_trial,
)
from embodied_silent_failures.openvla_runtime import (
    array_sha256,
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import (
    file_sha256,
    git_state,
    load_json,
)
from embodied_silent_failures.replay import load_clean_trace
from embodied_silent_failures.temporal_campaign import (
    manifest_sha256,
    validate_campaign_manifest,
)
from embodied_silent_failures.temporal_fault import (
    TemporalReplacementInjector,
    TemporalReplacementSpec,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resilient paired OpenVLA temporal-fault pilot."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--canary-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--maximum-new-attempts", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _canary_status(path: Path) -> dict[str, str]:
    status = load_json(path / "status.json")
    if status.get("state") != "complete":
        raise ValueError("temporal canary census has not completed")
    records = {}
    with (path / "canary.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            value = json.loads(line)
            site_id = str(value["site_id"])
            if site_id in records:
                raise ValueError(f"duplicate temporal canary site: {site_id}")
            records[site_id] = str(value["status"])
    if len(records) != int(status["site_count"]):
        raise ValueError("temporal canary records do not cover the declared census")
    return records


def _baseline_result(
    baseline_dir: Path,
    expected: dict[str, Any],
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    task_id = int(expected["task_id"])
    episode = int(expected["episode_index"])
    completion = baseline_dir / f"task{task_id}--ep{episode}.complete.json"
    result = load_json(completion)
    if result.get("condition") != "clean" or result.get("success") is not True:
        raise ValueError(f"staged paired baseline is not a clean success: {completion}")
    if int(result["trial_seed"]) != int(expected["trial_seed"]):
        raise ValueError(f"staged paired baseline has the wrong seed: {completion}")
    if result["initial_state_sha256"] != expected["initial_state_sha256"]:
        raise ValueError(f"staged paired baseline has the wrong initial state: {completion}")
    for name, artifact in expected["artifacts"].items():
        path = baseline_dir / artifact["staged_name"]
        if not path.is_file():
            raise FileNotFoundError(f"staged baseline {name} is missing: {path}")
        if path not in hash_cache:
            hash_cache[path] = file_sha256(path)
        digest = hash_cache[path]
        if digest != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"staged baseline {name} failed its artifact hash: {path}")
    return {**result, "_source_dir": str(baseline_dir.resolve())}


def _execution_record(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "run_started_at": _now(),
        "experiment_code": git_state(project_root),
        "openvla": git_state(args.openvla_root),
        "libero": git_state(args.libero_root),
        "campaign_manifest_sha256": file_sha256(args.manifest),
        "campaign_manifest_content_sha256": manifest_sha256(manifest),
        "temporal_injector_sha256": file_sha256(
            project_root / "embodied_silent_failures" / "temporal_fault.py"
        ),
        "rollout_sha256": file_sha256(
            project_root / "embodied_silent_failures" / "openvla_rollout.py"
        ),
    }


def _attempt_run(
    attempt: dict[str, Any],
    site: dict[str, Any],
    clean: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _now(),
        "condition": "activation_fault",
        "fault_model": {
            "kind": "single_temporal_value_replacement",
            "operator": "replace x_t with x_(t-1)",
            "site_id": site["site_id"],
        },
        "attempt": attempt,
        "site": site,
        "paired_clean": {
            key: clean[key]
            for key in (
                "task_id",
                "episode_index",
                "policy_steps",
                "trial_seed",
                "initial_state_sha256",
            )
        },
        "execution": execution,
    }


def main() -> None:
    args = _arguments()
    manifest = load_json(args.manifest)
    validate_campaign_manifest(manifest)
    if args.maximum_new_attempts is not None and args.maximum_new_attempts <= 0:
        raise ValueError("maximum new attempts must be positive")
    project_root = Path(__file__).resolve().parents[1]
    validate_pinned_runtime(
        args.checkpoint,
        args.openvla_root,
        args.libero_root,
        project_root=project_root,
    )
    canaries = _canary_status(args.canary_dir)
    sites = {site["site_id"]: site for site in manifest["sites"]}
    clean_metadata = {
        (int(value["task_id"]), int(value["episode_index"])): value
        for value in manifest["clean_trajectories"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    execution = _execution_record(manifest, args)
    campaign_run = {
        "schema_version": 1,
        "condition": "activation_fault",
        "campaign": manifest["campaign"],
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": file_sha256(args.manifest),
        },
        "canary": {
            "directory": str(args.canary_dir.resolve()),
            "records_sha256": file_sha256(args.canary_dir / "canary.jsonl"),
        },
        "planned_attempts": len(manifest["attempts"]),
        "execution": execution,
    }
    if run_path.exists():
        if not args.resume:
            raise FileExistsError(f"temporal campaign already exists: {run_path}")
        existing = load_json(run_path)
        for key in ("condition", "campaign", "manifest", "planned_attempts"):
            if existing.get(key) != campaign_run.get(key):
                raise ValueError(f"temporal campaign resume changed {key}")
    else:
        write_json_atomic(run_path, campaign_run)

    hash_cache: dict[Path, str] = {}
    clean_results = {
        key: _baseline_result(args.baseline_dir, value, hash_cache)
        for key, value in clean_metadata.items()
    }
    runtime = load_runtime(args.openvla_root, args.libero_root)
    policy_config = model_config(args.checkpoint, args.task_suite)
    runtime.set_seed_everywhere(int(manifest["seed"]))
    model = runtime.get_model(policy_config)
    model.eval()
    processor = runtime.get_processor(policy_config)
    if args.task_suite not in model.norm_stats:
        policy_config.unnorm_key = f"{args.task_suite}_no_noops"
    suite = runtime.benchmark.get_benchmark_dict()[args.task_suite]()
    task_ids = sorted({int(item["task_id"]) for item in manifest["attempts"]})
    states = {task_id: suite.get_task_init_states(task_id) for task_id in task_ids}
    environments = {}
    for task_id in task_ids:
        environments[task_id] = runtime.get_libero_env(
            suite.get_task(task_id), "openvla", resolution=256
        )

    counts = Counter()
    new_attempts = 0
    stopped_at_limit = False
    campaign_started = time.perf_counter()
    for index, attempt in enumerate(manifest["attempts"], start=1):
        attempt_id = str(attempt["attempt_id"])
        site = sites[str(attempt["site_id"])]
        output_dir = args.output_dir / "attempts" / attempt_id
        output_dir.mkdir(parents=True, exist_ok=True)
        trial = Trial(int(attempt["task_id"]), int(attempt["episode_index"]))
        marker = completion_path(output_dir, trial)
        error_path = output_dir / "attempt.error.json"
        if args.resume and marker.is_file():
            prepare_trial(output_dir, trial, resume=True)
            counts["complete"] += 1
            continue
        if (
            args.maximum_new_attempts is not None
            and new_attempts >= args.maximum_new_attempts
        ):
            stopped_at_limit = True
            break
        new_attempts += 1
        prepare_trial(output_dir, trial, resume=False)
        clean_result = clean_results[(trial.task_id, trial.episode_index)]
        write_json_atomic(
            output_dir / "run.json",
            _attempt_run(attempt, site, clean_result, execution),
        )
        if canaries.get(site["site_id"]) != "passed":
            write_json_atomic(
                error_path,
                {
                    "schema_version": 1,
                    "status": "not_run",
                    "reason": "selected_site_did_not_pass_current_value_canary",
                    "attempt": attempt,
                    "site_id": site["site_id"],
                    "canary_status": canaries.get(site["site_id"], "missing"),
                },
            )
            counts["canary_ineligible"] += 1
            continue

        trial_seed = int(clean_result["trial_seed"])
        initial_state = states[trial.task_id][trial.episode_index]
        initial_hash = array_sha256(runtime, initial_state)
        if initial_hash != clean_result["initial_state_sha256"]:
            raise ValueError(f"LIBERO initial state changed for {attempt_id}")
        clean_trace = load_clean_trace(clean_result)
        injector = TemporalReplacementInjector(
            runtime.torch,
            runtime.np,
            TemporalReplacementSpec(
                site_id=site["site_id"],
                identity=site["identity"],
                policy_step=int(attempt["policy_step"]),
                source_policy_step=int(attempt["source_policy_step"]),
            ),
        )
        env, task_description = environments[trial.task_id]
        started = time.perf_counter()
        try:
            runtime.set_seed_everywhere(trial_seed)
            injector.install(model)
            result = run_trial(
                RolloutConfig(
                    output_dir=output_dir,
                    task_suite=args.task_suite,
                    wait_steps=args.wait_steps,
                    save_video=False,
                    image_input_mode="stale",
                ),
                runtime,
                policy_config,
                model,
                processor,
                trial,
                env,
                task_description,
                initial_state,
                trial_seed,
                initial_hash,
                "activation_fault",
                injector,
                clean_trace,
                None,
                execution,
                replay_clean_prefix=False,
            )
            error_path.unlink(missing_ok=True)
            counts["complete"] += 1
            print(
                f"[{index}/{len(manifest['attempts'])}] {attempt_id} complete "
                f"success={result['success']} steps={result['policy_steps']}",
                flush=True,
            )
        except Exception as error:
            reason = getattr(error, "reason", "attempt_exception")
            write_json_atomic(
                error_path,
                {
                    "schema_version": 1,
                    "status": "error",
                    "reason": reason,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=16),
                    "attempt": attempt,
                    "site_id": site["site_id"],
                    "seconds": time.perf_counter() - started,
                    "retryable": not isinstance(error, TemporalPrefixDivergence),
                },
            )
            counts["error"] += 1
            print(
                f"[{index}/{len(manifest['attempts'])}] {attempt_id} error: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            injector.close()
            runtime.torch.cuda.empty_cache()

        write_json_atomic(
            args.output_dir / "status.json",
            {
                "schema_version": 1,
                "state": "running",
                "updated_at": _now(),
                "planned_attempts": len(manifest["attempts"]),
                "processed_attempts": sum(counts.values()),
                "counts": dict(sorted(counts.items())),
                "elapsed_seconds": time.perf_counter() - campaign_started,
                "last_attempt": attempt_id,
            },
        )

    final = {
        "schema_version": 1,
        "state": "paused_after_requested_limit" if stopped_at_limit else "complete",
        "finished_at": _now(),
        "planned_attempts": len(manifest["attempts"]),
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.perf_counter() - campaign_started,
        "new_attempts_this_launch": new_attempts,
    }
    write_json_atomic(args.output_dir / "status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
