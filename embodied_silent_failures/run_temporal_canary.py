from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.openvla_rollout import _extract_hidden_states
from embodied_silent_failures.openvla_runtime import (
    array_sha256,
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json
from embodied_silent_failures.temporal_campaign import eligible_sites
from embodied_silent_failures.temporal_fault import (
    TemporalProcessor,
    TemporalReplacementInjector,
    TemporalReplacementSpec,
    decode_action_tokens,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canary every structurally eligible OpenVLA temporal fault site."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--site-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--baseline-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _digest(torch: Any, np: Any, value: Any) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().contiguous().cpu().view(torch.uint8).numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(getattr(value, "dtype", array.dtype)).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _summary(runtime: Any, raw_action: Any, generated: Any, command: Any) -> dict[str, Any]:
    hidden = _extract_hidden_states(runtime, generated)
    return {
        "raw_action": runtime.np.asarray(raw_action).copy(),
        "command": runtime.np.asarray(command).copy(),
        "sequences": generated["sequences"].detach().cpu(),
        "safe_features": hidden,
        "digests": {
            "raw_action": _digest(runtime.torch, runtime.np, raw_action),
            "command": _digest(runtime.torch, runtime.np, command),
            "sequences": _digest(
                runtime.torch, runtime.np, generated["sequences"]
            ),
            "safe_features": _digest(runtime.torch, runtime.np, hidden),
        },
    }


def _query(
    runtime: Any,
    config: Any,
    model: Any,
    processor: Any,
    observation: dict[str, Any],
    task_description: str,
    resize_size: int,
    injector: TemporalReplacementInjector | None,
) -> dict[str, Any]:
    policy_step = 0
    source = observation
    if injector is not None:
        source = injector.boundary(
            "libero.current_observation", observation, policy_step=policy_step
        )
    image = runtime.get_libero_image(source, resize_size)
    if injector is not None:
        image = injector.boundary(
            "libero.current_image", image, policy_step=policy_step
        )
        image = injector.boundary(
            "policy.selected_image", image, policy_step=policy_step
        )
    state = runtime.np.concatenate(
        (
            source["robot0_eef_pos"],
            runtime.quat2axisangle(source["robot0_eef_quat"]),
            source["robot0_gripper_qpos"],
        )
    )
    policy_observation = {"full_image": image, "state": state}
    selected_processor = (
        TemporalProcessor(processor, injector) if injector is not None else processor
    )
    context = injector.inference(policy_step) if injector is not None else None
    if context is None:
        raw_action, generated = runtime.get_action(
            config,
            model,
            policy_observation,
            task_description,
            processor=selected_processor,
            n_samples=1,
        )
    else:
        with context:
            raw_action, generated = runtime.get_action(
                config,
                model,
                policy_observation,
                task_description,
                processor=selected_processor,
                n_samples=1,
            )
            generated = injector.boundary("openvla.policy_call", generated)
            action_tokens = generated["sequences"][:, -model.get_action_dim(
                config.unnorm_key
            ) :]
            action_tokens = injector.boundary(
                "openvla.action_tokens", action_tokens
            )
            if injector.requires_action_redecode():
                raw_action = decode_action_tokens(
                    model, action_tokens, config.unnorm_key, runtime.np
                )
    raw_action = runtime.np.asarray(raw_action).copy()
    if injector is not None:
        raw_action = injector.boundary(
            "openvla.raw_action", raw_action, policy_step=policy_step
        )
    command = runtime.normalize_gripper_action(raw_action.copy(), binarize=True)
    command = runtime.invert_gripper_action(command)
    if injector is not None:
        command = injector.boundary(
            "libero.executed_command", command, policy_step=policy_step
        )
    return _summary(runtime, raw_action, generated, command)


def _matches(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    return {
        name: reference["digests"][name] == observed["digests"][name]
        for name in reference["digests"]
    }


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                values.add(str(json.loads(line)["site_id"]))
    return values


def main() -> None:
    args = _arguments()
    if args.baseline_interval <= 0:
        raise ValueError("baseline interval must be positive")
    table = load_json(args.site_table)
    sites = sorted(eligible_sites(table), key=lambda value: value["site_id"])
    project_root = Path(__file__).resolve().parents[1]
    validate_pinned_runtime(
        args.checkpoint,
        args.openvla_root,
        args.libero_root,
        project_root=project_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "canary.jsonl"
    run_path = args.output_dir / "run.json"
    if run_path.exists() and not args.resume:
        raise FileExistsError(f"canary output already exists: {run_path}")
    if not run_path.exists():
        write_json_atomic(
            run_path,
            {
                "schema_version": 1,
                "started_at": _now(),
                "site_table": {
                    "path": str(args.site_table.resolve()),
                    "sha256": file_sha256(args.site_table),
                },
                "site_count": len(sites),
                "operator": "return a cloned current value at each eligible site",
                "task_suite": args.task_suite,
                "task_id": args.task_id,
                "episode_index": args.episode_index,
                "seed": args.seed,
                "wait_steps": args.wait_steps,
                "repositories": {
                    "experiment_code": git_state(project_root),
                    "openvla": git_state(args.openvla_root),
                    "libero": git_state(args.libero_root),
                },
                "checkpoint_revision": args.checkpoint.resolve().name,
                "implementation": {
                    "entrypoint_sha256": file_sha256(Path(__file__)),
                    "injector_sha256": file_sha256(
                        project_root
                        / "embodied_silent_failures"
                        / "temporal_fault.py"
                    ),
                },
            },
        )
    completed = _completed(records_path) if args.resume else set()

    runtime = load_runtime(args.openvla_root, args.libero_root)
    config = model_config(args.checkpoint, args.task_suite)
    runtime.set_seed_everywhere(args.seed)
    model = runtime.get_model(config)
    model.eval()
    processor = runtime.get_processor(config)
    if args.task_suite not in model.norm_stats:
        config.unnorm_key = f"{args.task_suite}_no_noops"
    suite = runtime.benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_state = suite.get_task_init_states(args.task_id)[args.episode_index]
    env, task_description = runtime.get_libero_env(
        task, "openvla", resolution=256
    )
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(args.wait_steps):
        observation, _, _, _ = env.step(runtime.get_libero_dummy_action("openvla"))
    resize_size = runtime.get_image_resize_size(config)
    with runtime.torch.inference_mode():
        reference = _query(
            runtime,
            config,
            model,
            processor,
            observation,
            task_description,
            resize_size,
            None,
        )
    started = time.perf_counter()
    passed = failed = 0
    for index, site in enumerate(sites, start=1):
        if site["site_id"] in completed:
            continue
        injector = TemporalReplacementInjector(
            runtime.torch,
            runtime.np,
            TemporalReplacementSpec(
                site_id=site["site_id"],
                identity=site["identity"],
                policy_step=0,
                source_policy_step=0,
                mode="current_value_canary",
            ),
        )
        record: dict[str, Any] = {
            "site_id": site["site_id"],
            "identity": site["identity"],
            "started_at": _now(),
        }
        site_started = time.perf_counter()
        try:
            injector.install(model)
            injector.begin_trial(args.seed)
            with runtime.torch.inference_mode():
                observed = _query(
                    runtime,
                    config,
                    model,
                    processor,
                    observation,
                    task_description,
                    resize_size,
                    injector,
                )
            injection = injector.require_injected()
            matches = _matches(reference, observed)
            schemas = site["schemas"]
            schema_match = injection["comparison"]["schema"] in schemas
            ok = all(matches.values()) and schema_match
            record.update(
                {
                    "status": "passed" if ok else "failed",
                    "output_matches": matches,
                    "schema_matches_table": schema_match,
                    "injection": injection,
                }
            )
            passed += int(ok)
            failed += int(not ok)
        except Exception as error:
            failed += 1
            record.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        finally:
            injector.close()
            runtime.torch.cuda.empty_cache()
        record["seconds"] = time.perf_counter() - site_started
        _append(records_path, record)
        if index % args.baseline_interval == 0:
            with runtime.torch.inference_mode():
                check = _query(
                    runtime,
                    config,
                    model,
                    processor,
                    observation,
                    task_description,
                    resize_size,
                    None,
                )
            if not all(_matches(reference, check).values()):
                raise RuntimeError("unmodified baseline changed during canary census")
        write_json_atomic(
            args.output_dir / "status.json",
            {
                "schema_version": 1,
                "state": "running",
                "updated_at": _now(),
                "planned": len(sites),
                "recorded": len(completed) + passed + failed,
                "passed_this_launch": passed,
                "failed_this_launch": failed,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        if index % 50 == 0:
            print(
                f"[{index}/{len(sites)}] passed={passed} failed={failed}",
                flush=True,
            )

    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    counts = {}
    for status in ("passed", "failed", "error"):
        counts[status] = sum(row["status"] == status for row in rows)
    final = {
        "schema_version": 1,
        "state": "complete",
        "finished_at": _now(),
        "counts": counts,
        "site_count": len(sites),
        "initial_state_sha256": array_sha256(runtime, initial_state),
        "baseline_digests": reference["digests"],
        "records_sha256": file_sha256(records_path),
    }
    write_json_atomic(args.output_dir / "status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
