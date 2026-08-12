import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.plan import load_trial_manifest, seed_for_trial
from embodied_silent_failures.probe_openvla import (
    _load_feature_bounds,
    _load_safe_monitor,
    _probe_step,
    _step_environment,
)
from embodied_silent_failures.replay import (
    CleanTrace,
    load_clean_trace,
    observation_error,
    replay_action,
)
from embodied_silent_failures.run_openvla import (
    CHECKPOINT_REVISION,
    LIBERO_REVISION,
    OPENVLA_REVISION,
    REPLAY_OBSERVATION_TOLERANCE,
    _array_sha256,
    _git_revision,
    _load_runtime,
    _model_config,
    _paired_clean_results,
    _sha256,
)


def _parse_lags(value: str) -> list[int]:
    try:
        lags = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("image lags must be comma-separated integers") from error
    if not lags or any(lag <= 0 for lag in lags):
        raise ValueError("image lags must be positive")
    if len(lags) != len(set(lags)):
        raise ValueError("image lags must not contain duplicates")
    return sorted(lags)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure OpenVLA and SAFE responses to one stale camera frame."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--trial-manifest", required=True, type=Path)
    parser.add_argument(
        "--paired-clean-dir",
        dest="paired_clean_dirs",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument("--calibration-clean-dir", required=True, type=Path)
    parser.add_argument("--calibration-split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--minimum-policy-step", type=int, default=20)
    parser.add_argument("--image-lags", default="1,2,4,8,16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    args = parser.parse_args()
    args.image_lags = _parse_lags(args.image_lags)
    if args.minimum_policy_step < 0:
        raise ValueError("minimum policy step must be non-negative")
    return args


def _reconstruct_images(
    runtime: Any,
    env: Any,
    initial_state: Any,
    trace: CleanTrace,
    step: int,
    lags: list[int],
    wait_steps: int,
    resize_size: int,
) -> tuple[dict[int, Any], dict[str, Any], float]:
    requested = {step, *(step - lag for lag in lags if step >= lag)}
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        observation, _, _, _ = env.step(runtime.get_libero_dummy_action("openvla"))

    images = {}
    maximum_error = 0.0
    target_observation = None
    for policy_step in range(step + 1):
        error = observation_error(runtime.np, trace, observation, policy_step)
        maximum_error = max(maximum_error, error)
        if error > REPLAY_OBSERVATION_TOLERANCE:
            raise RuntimeError(
                f"clean replay diverged at step {policy_step}: {error:.3g}"
            )
        if policy_step in requested:
            images[policy_step] = runtime.get_libero_image(
                observation, resize_size
            ).copy()
        if policy_step == step:
            target_observation = observation
            break
        action = replay_action(runtime.np, trace, policy_step)
        observation, done, _, _ = _step_environment(env, action)
        if done:
            raise RuntimeError("clean replay terminated before the probe step")
    if target_observation is None or step not in images:
        raise RuntimeError("failed to reconstruct the target observation")
    return images, target_observation, maximum_error


def _executed_action(runtime: Any, raw_action: Any) -> Any:
    action = runtime.normalize_gripper_action(
        runtime.np.asarray(raw_action).copy(), binarize=True
    )
    return runtime.invert_gripper_action(action)


def _action_metrics(np: Any, clean: Any, stale: Any) -> dict[str, Any]:
    delta = np.asarray(stale, dtype=float) - np.asarray(clean, dtype=float)
    return {
        "translation_l2": float(np.linalg.norm(delta[:3])),
        "rotation_l2": float(np.linalg.norm(delta[3:6])),
        "continuous_l2": float(np.linalg.norm(delta[:6])),
        "continuous_maximum_absolute": float(np.max(np.abs(delta[:6]))),
        "gripper_changed": bool(stale[-1] != clean[-1]),
        "any_executed_action_changed": bool(np.any(stale != clean)),
    }


def _feature_metrics(torch: Any, hidden: Any, bounds: dict[str, Any]) -> dict[str, Any]:
    feature = hidden[-1].float()
    finite = torch.isfinite(feature)
    outside_coordinate = finite & (
        (feature < bounds["lower"]) | (feature > bounds["upper"])
    )
    maximum = float(torch.max(torch.abs(feature)).item())
    return {
        "all_finite": bool(finite.all()),
        "maximum_absolute": maximum,
        "exceeds_calibration_global_maximum": (
            maximum > bounds["global_absolute_maximum"]
        ),
        "outside_calibration_coordinate_count": int(
            outside_coordinate.sum().item()
        ),
    }


def _inference_record(
    runtime: Any,
    model_config: Any,
    model: Any,
    processor: Any,
    monitor: Any,
    image: Any,
    state: Any,
    task_description: str,
    bounds: dict[str, Any],
) -> dict[str, Any]:
    with runtime.torch.inference_mode():
        raw_action, generated = runtime.get_action(
            model_config,
            model,
            {"full_image": image, "state": state},
            task_description,
            processor=processor,
            n_samples=1,
        )
        hidden = runtime.torch.stack(
            [token[-1][0, -1, :] for token in generated["hidden_states"]], dim=0
        ).detach()
        safe_score = float(
            monitor.projector(hidden[-1].float().view(1, 1, -1)).item()
        )
    executed = _executed_action(runtime, raw_action)
    return {
        "raw_action": runtime.np.asarray(raw_action),
        "executed_action": runtime.np.asarray(executed),
        "hidden": hidden,
        "safe_raw_score": safe_score,
        "feature_range": _feature_metrics(runtime.torch, hidden, bounds),
    }


def _plain_record(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_action": value["raw_action"].astype(float).tolist(),
        "executed_action": value["executed_action"].astype(float).tolist(),
        "safe_raw_score": value["safe_raw_score"],
        "feature_range": value["feature_range"],
    }


def main() -> None:
    args = _parse_arguments()
    if CHECKPOINT_REVISION not in args.checkpoint.resolve().parts:
        raise RuntimeError("probe requires the pinned OpenVLA checkpoint")
    for name, actual, expected in (
        ("OpenVLA", _git_revision(args.openvla_root), OPENVLA_REVISION),
        ("LIBERO", _git_revision(args.libero_root), LIBERO_REVISION),
    ):
        if actual != expected:
            raise RuntimeError(f"{name} revision is {actual}, expected {expected}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"probe output directory is not empty: {args.output_dir}")

    plan = load_trial_manifest(args.trial_manifest)
    plan, clean_results = _paired_clean_results(args.paired_clean_dirs, plan)
    runtime = _load_runtime(args.openvla_root, args.libero_root)
    monitor = _load_safe_monitor(args.safe_root, args.monitor_dir, runtime.torch)
    bounds = _load_feature_bounds(
        runtime.torch, args.calibration_clean_dir, args.calibration_split
    )
    model_config = _model_config(
        SimpleNamespace(checkpoint=args.checkpoint, task_suite=args.task_suite)
    )
    runtime.set_seed_everywhere(args.seed)
    model = runtime.get_model(model_config)
    model.eval()
    processor = runtime.get_processor(model_config)
    if args.task_suite not in model.norm_stats:
        model_config.unnorm_key = f"{args.task_suite}_no_noops"

    benchmark_class = runtime.benchmark.get_benchmark_dict()[args.task_suite]
    task_suite = benchmark_class()
    initial_states = {
        task_id: task_suite.get_task_init_states(task_id)
        for task_id in sorted({trial.task_id for trial in plan})
    }
    records = []
    for task_id in sorted({trial.task_id for trial in plan}):
        task = task_suite.get_task(task_id)
        env, task_description = runtime.get_libero_env(
            task, "openvla", resolution=256
        )
        try:
            for trial in [value for value in plan if value.task_id == task_id]:
                trial_seed = seed_for_trial(args.seed, trial)
                runtime.set_seed_everywhere(trial_seed)
                clean_result = clean_results[trial]
                trace = load_clean_trace(clean_result)
                step = _probe_step(
                    trace,
                    "first_gripper_transition",
                    fixed_step=0,
                    minimum=args.minimum_policy_step,
                )
                initial_state = initial_states[task_id][trial.episode_index]
                if _array_sha256(runtime, initial_state) != clean_result["initial_state_sha256"]:
                    raise RuntimeError("clean reference initial state does not match LIBERO")
                resize_size = runtime.get_image_resize_size(model_config)
                eligible_lags = [lag for lag in args.image_lags if lag <= step]
                images, observation, replay_error = _reconstruct_images(
                    runtime,
                    env,
                    initial_state,
                    trace,
                    step,
                    eligible_lags,
                    args.wait_steps,
                    resize_size,
                )
                state = runtime.np.concatenate(
                    (
                        observation["robot0_eef_pos"],
                        runtime.quat2axisangle(observation["robot0_eef_quat"]),
                        observation["robot0_gripper_qpos"],
                    )
                )
                current = _inference_record(
                    runtime,
                    model_config,
                    model,
                    processor,
                    monitor,
                    images[step],
                    state,
                    task_description,
                    bounds,
                )
                expected = replay_action(runtime.np, trace, step)
                clean_action_error = float(
                    runtime.np.max(runtime.np.abs(current["executed_action"] - expected))
                )
                expected_hidden = runtime.torch.as_tensor(
                    trace.hidden_states[step]
                ).to(current["hidden"].device)
                clean_hidden_error = float(
                    runtime.torch.max(
                        runtime.torch.abs(current["hidden"].float() - expected_hidden.float())
                    ).item()
                )
                if clean_action_error > 1e-6 or clean_hidden_error != 0.0:
                    raise RuntimeError(
                        "clean inference does not reproduce the saved clean trace: "
                        f"action error {clean_action_error:.3g}, hidden error "
                        f"{clean_hidden_error:.3g}"
                    )

                candidates = []
                for lag in eligible_lags:
                    stale = _inference_record(
                        runtime,
                        model_config,
                        model,
                        processor,
                        monitor,
                        images[step - lag],
                        state,
                        task_description,
                        bounds,
                    )
                    image_delta = images[step - lag].astype(float) - images[step].astype(float)
                    candidates.append(
                        {
                            "image_lag": lag,
                            "source_policy_step": step - lag,
                            "image_mean_absolute_difference": float(
                                runtime.np.mean(runtime.np.abs(image_delta))
                            ),
                            "image_maximum_absolute_difference": float(
                                runtime.np.max(runtime.np.abs(image_delta))
                            ),
                            "action_change": _action_metrics(
                                runtime.np,
                                current["executed_action"],
                                stale["executed_action"],
                            ),
                            **_plain_record(stale),
                        }
                    )
                records.append(
                    {
                        "task_id": trial.task_id,
                        "episode_index": trial.episode_index,
                        "trial_seed": trial_seed,
                        "policy_step": step,
                        "replay_maximum_numeric_observation_error": replay_error,
                        "clean_action_maximum_absolute_error": clean_action_error,
                        "clean_hidden_maximum_absolute_error": clean_hidden_error,
                        "current": _plain_record(current),
                        "candidates": candidates,
                    }
                )
                changed = sum(
                    candidate["action_change"]["gripper_changed"]
                    for candidate in candidates
                )
                print(
                    f"probed task {trial.task_id}, episode {trial.episode_index}, "
                    f"step {step}: {changed}/{len(candidates)} stale frames changed gripper"
                )
        finally:
            env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "probe.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "intervention": (
                "one_policy_inference_uses_an_earlier_camera_frame_while_current_"
                "proprioception_and_all_later_inputs_remain_current"
            ),
            "selection_basis": "none_discovery_records_all_predeclared_image_lags",
            "image_lags": args.image_lags,
            "timing": "first_gripper_transition_after_minimum_policy_step",
            "minimum_policy_step": args.minimum_policy_step,
            "calibration": {
                "clean_directory": str(args.calibration_clean_dir.resolve()),
                "split_manifest": str(args.calibration_split.resolve()),
                "split_manifest_sha256": _sha256(args.calibration_split),
                "split": "train",
                "rollout_count": bounds["rollout_count"],
                "step_count": bounds["step_count"],
                "global_absolute_maximum": bounds["global_absolute_maximum"],
            },
            "records": records,
        },
    )
    print(f"recorded {len(records)} stale-image probes")


if __name__ == "__main__":
    main()
