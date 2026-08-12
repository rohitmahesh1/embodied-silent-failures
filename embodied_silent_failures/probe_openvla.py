import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.faults import FaultSpec, TransientActivationFault
from embodied_silent_failures.plan import load_trial_manifest, seed_for_trial
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
from embodied_silent_failures.score_safe import SAFE_REVISION, _validate_monitor


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find exact shared-tensor bit flips that change OpenVLA actions."
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument(
        "--timing", choices=("fixed", "first_gripper_transition"), default="fixed"
    )
    parser.add_argument("--policy-step", type=int, default=50)
    parser.add_argument("--minimum-policy-step", type=int, default=20)
    parser.add_argument("--calibration-clean-dir", required=True, type=Path)
    parser.add_argument("--calibration-split", required=True, type=Path)
    parser.add_argument("--candidate-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    args = parser.parse_args()
    if args.policy_step < 0 or args.minimum_policy_step < 0:
        raise ValueError("policy steps must be non-negative")
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate batch size must be positive")
    return args


def _probe_step(trace: CleanTrace, timing: str, fixed_step: int, minimum: int) -> int:
    if timing == "fixed":
        step = fixed_step
    else:
        gripper = [float(row["action/dgripper"]) for row in trace.rows]
        changes = [
            index
            for index in range(max(1, minimum), len(gripper))
            if gripper[index] != gripper[index - 1]
        ]
        if not changes:
            raise ValueError("clean trace has no gripper transition after the minimum step")
        step = changes[0]
    if step >= len(trace.rows):
        raise ValueError(f"probe step {step} is outside a {len(trace.rows)}-step trace")
    return step


def _reconstruct_observation(
    runtime: Any,
    env: Any,
    initial_state: Any,
    trace: CleanTrace,
    step: int,
    wait_steps: int,
) -> tuple[dict[str, Any], float]:
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        observation, _, _, _ = env.step(runtime.get_libero_dummy_action("openvla"))

    maximum_error = 0.0
    for policy_step in range(step + 1):
        error = observation_error(runtime.np, trace, observation, policy_step)
        maximum_error = max(maximum_error, error)
        if error > REPLAY_OBSERVATION_TOLERANCE:
            raise RuntimeError(
                f"clean replay diverged at step {policy_step}: {error:.3g}"
            )
        if policy_step < step:
            action = replay_action(runtime.np, trace, policy_step)
            observation, done, _, _ = _step_environment(env, action)
            if done:
                raise RuntimeError("clean replay terminated before the probe step")
    return observation, maximum_error


def _step_environment(env: Any, action: Any) -> tuple[dict[str, Any], bool, Any, Any]:
    observation, reward, done, info = env.step(action.tolist())
    return observation, bool(done), reward, info


def _load_safe_monitor(safe_root: Path, monitor_dir: Path, torch: Any) -> Any:
    if _git_revision(safe_root) != SAFE_REVISION:
        raise RuntimeError(f"SAFE must be checked out at {SAFE_REVISION}")
    _, paths = _validate_monitor(monitor_dir)
    sys.path.insert(0, str(safe_root.resolve()))
    from failure_prob.model import get_model
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(paths["configuration"])
    if (
        cfg.model.name != "indep"
        or int(cfg.model.n_history_steps) != 1
        or float(cfg.dataset.token_idx_rel) != 1.0
    ):
        raise RuntimeError("probe expects the frozen one-step final-token SAFE MLP")
    monitor = get_model(cfg, 4096)
    monitor.load_state_dict(torch.load(paths["checkpoint"], map_location="cpu"))
    monitor.to("cuda")
    monitor.eval()
    return monitor


def _mask(torch: Any, bit_index: int) -> Any:
    unsigned = 1 << bit_index
    signed = unsigned if bit_index < 15 else unsigned - (1 << 16)
    return torch.tensor(signed, dtype=torch.int16, device="cuda")


def _bit_class(bit_index: int) -> str:
    if bit_index < 7:
        return "mantissa"
    if bit_index < 15:
        return "exponent"
    return "sign"


def _decode_final_action(model: Any, token: int, unnorm_key: str) -> float:
    np = __import__("numpy")
    index = int(np.clip(model.vocab_size - token - 1, 0, len(model.bin_centers) - 1))
    normalized = float(model.bin_centers[index])
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    if bool(mask[-1]):
        raw = 0.5 * (normalized + 1) * (stats["q99"][-1] - stats["q01"][-1])
        raw += stats["q01"][-1]
        return float(raw)
    return normalized


def _executed_gripper(runtime: Any, clean_action: Any, raw_gripper: float) -> float:
    action = runtime.np.asarray(clean_action).copy()
    action[-1] = raw_gripper
    action = runtime.normalize_gripper_action(action, binarize=True)
    action = runtime.invert_gripper_action(action)
    return float(action[-1])


def _load_feature_bounds(
    torch: Any, clean_dir: Path, split_path: Path
) -> dict[str, Any]:
    with split_path.open(encoding="utf-8") as file:
        split = json.load(file)
    entries = split.get("splits", {}).get("train")
    if not isinstance(entries, list) or not entries:
        raise ValueError("calibration split has no training rollouts")

    lower = None
    upper = None
    rollout_count = 0
    step_count = 0
    for entry in entries:
        csv_name = entry.get("csv")
        if not isinstance(csv_name, str):
            raise ValueError("calibration split entry has no CSV name")
        path = clean_dir / Path(csv_name).with_suffix(".pkl").name
        with path.open("rb") as file:
            payload = pickle.load(file)
        hidden = torch.as_tensor(payload["hidden_states"])[:, -1, :].float()
        if hidden.ndim != 2 or hidden.shape[1] != 4096:
            raise ValueError(f"unexpected calibration feature shape in {path}")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError(f"non-finite calibration feature in {path}")
        current_lower = hidden.amin(dim=0)
        current_upper = hidden.amax(dim=0)
        lower = current_lower if lower is None else torch.minimum(lower, current_lower)
        upper = current_upper if upper is None else torch.maximum(upper, current_upper)
        rollout_count += 1
        step_count += int(hidden.shape[0])

    return {
        "lower": lower.to("cuda"),
        "upper": upper.to("cuda"),
        "global_absolute_maximum": float(
            torch.maximum(torch.abs(lower), torch.abs(upper)).max().item()
        ),
        "rollout_count": rollout_count,
        "step_count": step_count,
    }


def _candidate_sort_key(record: dict[str, Any]) -> tuple[float, int, int]:
    return (
        record["absolute_feature_change"],
        record["feature_index"],
        record["bit_index"],
    )


def _candidate_records(
    runtime: Any,
    model: Any,
    monitor: Any,
    generated: Any,
    clean_action: Any,
    bounds: dict[str, Any],
    batch_size: int,
    unnorm_key: str,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any] | None]:
    torch = runtime.torch
    generation_step = 6
    hidden = generated["hidden_states"][generation_step][-1][0, -1, :].detach()
    logits = generated["logits"][generation_step][0].detach()
    token = int(generated["sequences"][0, -7 + generation_step].item())
    if token != int(torch.argmax(logits).item()):
        raise RuntimeError("generated token does not match the recorded logits")
    if hidden.dtype != torch.bfloat16 or hidden.shape != (4096,):
        raise RuntimeError(f"unexpected monitored feature: {hidden.dtype} {hidden.shape}")

    hidden_bits = hidden.contiguous().view(torch.int16)
    candidates: list[tuple[int, int, float]] = []
    global_limit = float(bounds["global_absolute_maximum"])
    counts = {
        "native_bit_flips": 4096 * 16,
        "finite": 0,
        "within_global_range": 0,
        "within_coordinate_range": 0,
        "action_changing_within_coordinate_range": 0,
    }

    for bit_index in range(16):
        changed_bits = torch.bitwise_xor(hidden_bits, _mask(torch, bit_index))
        changed = changed_bits.view(torch.bfloat16)
        finite = torch.isfinite(changed)
        in_global = finite & (torch.abs(changed.float()) <= global_limit)
        in_coordinate = in_global & (changed.float() >= bounds["lower"]) & (
            changed.float() <= bounds["upper"]
        )
        counts["finite"] += int(finite.sum().item())
        counts["within_global_range"] += int(in_global.sum().item())
        counts["within_coordinate_range"] += int(in_coordinate.sum().item())
        for feature in torch.nonzero(in_coordinate, as_tuple=False).flatten().tolist():
            candidates.append(
                (
                    int(feature),
                    bit_index,
                    float(changed[feature].item()),
                )
            )

    candidate_tokens: list[int] = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            faulted = hidden.unsqueeze(0).expand(len(chunk), -1).clone()
            for row, (feature, _, after) in enumerate(chunk):
                faulted[row, feature] = after
            changed_logits = model.language_model.lm_head(faulted)
            candidate_tokens.extend(torch.argmax(changed_logits, dim=-1).tolist())
        clean_safe_score = float(
            monitor.projector(hidden.float().view(1, 1, -1)).item()
        )

    clean_raw_gripper = float(clean_action[-1])
    decoded_clean_gripper = _decode_final_action(model, token, unnorm_key)
    if not math.isclose(
        decoded_clean_gripper, clean_raw_gripper, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("clean token decoding disagrees with OpenVLA action")
    clean_executed = _executed_gripper(runtime, clean_action, clean_raw_gripper)
    clean_maximum_absolute_feature = float(torch.max(torch.abs(hidden.float())).item())
    records = []
    for candidate, candidate_token in zip(candidates, candidate_tokens):
        feature, bit_index, after = candidate
        raw_gripper = _decode_final_action(model, candidate_token, unnorm_key)
        executed = _executed_gripper(runtime, clean_action, raw_gripper)
        action_changed = executed != clean_executed
        counts["action_changing_within_coordinate_range"] += int(
            action_changed
        )
        if not action_changed:
            continue
        before_bits = int(hidden_bits[feature].item()) & 0xFFFF
        after_bits = before_bits ^ (1 << bit_index)
        records.append(
            {
                "feature_index": feature,
                "bit_index": bit_index,
                "bit_class": _bit_class(bit_index),
                "before_bits": f"0x{before_bits:04x}",
                "after_bits": f"0x{after_bits:04x}",
                "before_value": float(hidden[feature].item()),
                "after_value": after,
                "absolute_feature_change": abs(after - float(hidden[feature].item())),
                "clean_maximum_absolute_feature": clean_maximum_absolute_feature,
                "fault_to_clean_maximum_ratio": (
                    abs(after) / clean_maximum_absolute_feature
                ),
                "within_calibration_global_range": True,
                "within_calibration_coordinate_range": True,
                "calibration_coordinate_lower": float(bounds["lower"][feature].item()),
                "calibration_coordinate_upper": float(bounds["upper"][feature].item()),
                "clean_token": token,
                "fault_token": candidate_token,
                "token_changed": candidate_token != token,
                "clean_raw_gripper": clean_raw_gripper,
                "fault_raw_gripper": raw_gripper,
                "raw_gripper_change": raw_gripper - clean_raw_gripper,
                "clean_executed_gripper": clean_executed,
                "fault_executed_gripper": executed,
                "executed_action_changed": action_changed,
                "clean_safe_raw_score": clean_safe_score,
            }
        )

    records.sort(key=_candidate_sort_key)
    selected = records[0] if records else None
    if selected is not None:
        faulted = hidden.clone()
        faulted[selected["feature_index"]] = selected["after_value"]
        with torch.no_grad():
            selected["fault_safe_raw_score"] = float(
                monitor.projector(faulted.float().view(1, 1, -1)).item()
            )
    return counts, records, selected


def _verify_selected_fault(
    runtime: Any,
    model_config: Any,
    model: Any,
    processor: Any,
    policy_observation: dict[str, Any],
    task_description: str,
    trial_seed: int,
    spec: FaultSpec,
    expected: dict[str, Any],
) -> dict[str, Any]:
    injector = TransientActivationFault(runtime.torch, spec)
    injector.install(model)
    injector.begin_trial(trial_seed)
    try:
        with runtime.torch.inference_mode(), injector.inference(spec.policy_step):
            action, generated = runtime.get_action(
                model_config,
                model,
                policy_observation,
                task_description,
                processor=processor,
                n_samples=1,
            )
    finally:
        injector.close()
    actual_token = int(generated["sequences"][0, -1].item())
    actual_hidden = generated["hidden_states"][6][-1][0, -1, spec.feature_index]
    if actual_token != expected["fault_token"]:
        raise RuntimeError("hooked fault token disagrees with analytical probe")
    if float(actual_hidden.item()) != expected["after_value"]:
        raise RuntimeError("hooked SAFE feature disagrees with analytical probe")
    if not math.isclose(
        float(action[-1]), expected["fault_raw_gripper"], rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("hooked gripper action disagrees with analytical probe")
    return {
        "fault_token": actual_token,
        "raw_gripper": float(action[-1]),
        "feature_value": float(actual_hidden.item()),
        "injection": injector.require_injected(),
    }


def main() -> None:
    args = _parse_arguments()
    if CHECKPOINT_REVISION not in args.checkpoint.resolve().parts:
        raise RuntimeError("probe requires the pinned OpenVLA checkpoint")
    revisions = {
        "OpenVLA": (_git_revision(args.openvla_root), OPENVLA_REVISION),
        "LIBERO": (_git_revision(args.libero_root), LIBERO_REVISION),
    }
    for name, (actual, expected) in revisions.items():
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
    selected_entries = []
    for task_id in sorted({trial.task_id for trial in plan}):
        task = task_suite.get_task(task_id)
        env, task_description = runtime.get_libero_env(task, "openvla", resolution=256)
        try:
            for trial in [value for value in plan if value.task_id == task_id]:
                trial_seed = seed_for_trial(args.seed, trial)
                runtime.set_seed_everywhere(trial_seed)
                clean_result = clean_results[trial]
                trace = load_clean_trace(clean_result)
                try:
                    step = _probe_step(
                        trace,
                        args.timing,
                        args.policy_step,
                        args.minimum_policy_step,
                    )
                except ValueError as error:
                    if args.timing != "first_gripper_transition":
                        raise
                    records.append(
                        {
                            "task_id": trial.task_id,
                            "episode_index": trial.episode_index,
                            "trial_seed": trial_seed,
                            "timing": args.timing,
                            "status": "ineligible_timing",
                            "reason": str(error),
                        }
                    )
                    print(
                        f"skipped task {trial.task_id}, episode {trial.episode_index}: "
                        f"{error}"
                    )
                    continue
                initial_state = initial_states[task_id][trial.episode_index]
                if _array_sha256(runtime, initial_state) != clean_result["initial_state_sha256"]:
                    raise RuntimeError("clean reference initial state does not match LIBERO")
                observation, replay_error = _reconstruct_observation(
                    runtime, env, initial_state, trace, step, args.wait_steps
                )
                image = runtime.get_libero_image(
                    observation, runtime.get_image_resize_size(model_config)
                )
                state = runtime.np.concatenate(
                    (
                        observation["robot0_eef_pos"],
                        runtime.quat2axisangle(observation["robot0_eef_quat"]),
                        observation["robot0_gripper_qpos"],
                    )
                )
                policy_observation = {"full_image": image, "state": state}
                with runtime.torch.inference_mode():
                    clean_action, generated = runtime.get_action(
                        model_config,
                        model,
                        policy_observation,
                        task_description,
                        processor=processor,
                        n_samples=1,
                    )
                counts, candidates, selected = _candidate_records(
                    runtime,
                    model,
                    monitor,
                    generated,
                    clean_action,
                    bounds,
                    args.candidate_batch_size,
                    model_config.unnorm_key,
                )
                record = {
                    "task_id": trial.task_id,
                    "episode_index": trial.episode_index,
                    "trial_seed": trial_seed,
                    "policy_step": step,
                    "timing": args.timing,
                    "status": (
                        "selected" if selected is not None else "no_in_range_candidate"
                    ),
                    "replay_maximum_numeric_observation_error": replay_error,
                    "candidate_counts": counts,
                    "action_changing_candidates": candidates,
                    "selected": selected,
                }
                if selected is not None:
                    spec = FaultSpec(
                        site="final_hidden",
                        layer=None,
                        policy_step=step,
                        generation_step=6,
                        bit_index=selected["bit_index"],
                        seed=0,
                        feature_index=selected["feature_index"],
                    )
                    record["verification"] = _verify_selected_fault(
                        runtime,
                        model_config,
                        model,
                        processor,
                        policy_observation,
                        task_description,
                        trial_seed,
                        spec,
                        selected,
                    )
                    selected_entries.append(
                        {
                            "task_id": trial.task_id,
                            "episode_index": trial.episode_index,
                            "fault": spec.to_dict(),
                        }
                    )
                records.append(record)
                print(
                    f"probed task {trial.task_id}, episode {trial.episode_index}, "
                    f"step {step}: {counts['action_changing_within_coordinate_range']} "
                    "in-range action-changing candidates"
                )
        finally:
            env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "probe.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_basis": (
                "smallest_coordinate_calibrated_feature_change_that_changes_"
                "executed_action_without_safe_score"
            ),
            "timing": args.timing,
            "candidate_search": (
                "all_4096_features_times_16_native_bfloat16_bits_with_exact_"
                "model_head_evaluation_after_coordinate_range_gate"
            ),
            "candidate_batch_size": args.candidate_batch_size,
            "calibration": {
                "clean_directory": str(args.calibration_clean_dir.resolve()),
                "split_manifest": str(args.calibration_split.resolve()),
                "split_manifest_sha256": _sha256(args.calibration_split),
                "split": "train",
                "rollout_count": bounds["rollout_count"],
                "step_count": bounds["step_count"],
                "global_absolute_maximum": bounds["global_absolute_maximum"],
                "range": "per_feature_observed_minimum_and_maximum",
            },
            "selected_trial_count": len(selected_entries),
            "records": records,
        },
    )
    write_json_atomic(
        args.output_dir / "fault-manifest.json",
        {
            "schema_version": 1,
            "selection_basis": (
                "smallest_coordinate_calibrated_feature_change_that_changes_"
                "executed_action_without_safe_score"
            ),
            "trials": selected_entries,
        },
    )
    print(f"selected {len(selected_entries)} faults from {len(records)} probes")


if __name__ == "__main__":
    main()
