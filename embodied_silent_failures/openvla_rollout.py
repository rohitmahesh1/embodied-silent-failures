import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from embodied_silent_failures.artifacts import (
    completion_path,
    safe_stem,
    temporary_path,
    write_csv_atomic,
    write_json_atomic,
    write_pickle_atomic,
)
from embodied_silent_failures.evidence_graph.rollout import RolloutEvidence
from embodied_silent_failures.faults import TransientActivationFault
from embodied_silent_failures.openvla_runtime import Runtime
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.replay import (
    CleanTrace,
    observation_error,
    replay_action,
)
from embodied_silent_failures.stale_image_manifest import StaleImageSpec


REPLAY_OBSERVATION_TOLERANCE = 1e-6
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass(frozen=True)
class RolloutConfig:
    output_dir: Path
    task_suite: str
    wait_steps: int
    save_video: bool
    image_input_mode: str


class CounterfactualReplayInvalid(RuntimeError):
    reason: str


class CounterfactualReplayDivergence(CounterfactualReplayInvalid):
    reason = "counterfactual_replay_diverged_before_intervention"

    def __init__(self, policy_step: int, error: float):
        self.policy_step = policy_step
        self.error = error
        super().__init__(
            f"counterfactual replay diverged at step {policy_step}: "
            f"maximum numeric observation error {error:.3g} exceeds "
            f"{REPLAY_OBSERVATION_TOLERANCE:.3g}"
        )


class CounterfactualReplayTerminated(CounterfactualReplayInvalid):
    reason = "counterfactual_replay_terminated_before_intervention"

    def __init__(self, policy_step: int, intervention_step: int):
        self.policy_step = policy_step
        self.intervention_step = intervention_step
        super().__init__(
            f"counterfactual replay terminated after step {policy_step}, "
            f"before intervention step {intervention_step}"
        )


def build_image_intervention_record(
    spec: StaleImageSpec, mode: str, trial_seed: int
) -> dict[str, Any]:
    if mode == "stale":
        return {**spec.to_dict(), "trial_seed": trial_seed}
    if mode == "current_control":
        return {
            "kind": "current_image_control",
            "policy_step": spec.policy_step,
            "input_policy_step": spec.policy_step,
            "matched_stale_image_lag": spec.image_lag,
            "matched_stale_source_policy_step": spec.source_policy_step,
            "trial_seed": trial_seed,
        }
    raise ValueError(f"unsupported image input mode: {mode}")


def image_fault_applied(spec: StaleImageSpec, mode: str, policy_step: int) -> bool:
    if mode not in {"stale", "current_control"}:
        raise ValueError(f"unsupported image input mode: {mode}")
    return mode == "stale" and policy_step == spec.policy_step


def _numeric_observation(runtime: Runtime, observation: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in observation.items():
        if "image" in key.lower():
            continue
        array = runtime.np.asarray(value)
        if array.dtype.kind not in "biuf" or array.size > 4096:
            continue
        values[key] = array.copy()
    return values


def _stack_observations(runtime: Runtime, history: list[dict[str, Any]]) -> dict[str, Any]:
    common_keys = set(history[0]) if history else set()
    for observation in history[1:]:
        common_keys.intersection_update(observation)
    return {
        key: runtime.np.stack([observation[key] for observation in history])
        for key in sorted(common_keys)
    }


def _python_value(runtime: Runtime, value: Any) -> Any:
    if isinstance(value, runtime.np.generic):
        return value.item()
    return value


def _extract_hidden_states(runtime: Runtime, generated: Any) -> Any:
    hidden_states = generated["hidden_states"]
    final_layer = [token_states[-1][0, -1, :] for token_states in hidden_states]
    result = runtime.torch.stack(final_layer, dim=0).detach().cpu()
    if result.ndim != 2 or result.shape[0] != 7:
        raise ValueError(f"unexpected OpenVLA hidden-state shape: {tuple(result.shape)}")
    return result


def run_trial(
    config: RolloutConfig,
    runtime: Runtime,
    model_config: SimpleNamespace,
    model: Any,
    processor: Any,
    trial: Trial,
    env: Any,
    task_description: str,
    initial_state: Any,
    trial_seed: int,
    initial_state_sha256: str,
    run_condition: str,
    fault_injector: TransientActivationFault | None,
    clean_trace: CleanTrace | None,
    stale_image_spec: StaleImageSpec | None,
    execution: dict[str, Any],
    evidence: RolloutEvidence | None = None,
) -> dict[str, Any]:
    runtime.torch.cuda.reset_peak_memory_stats()
    rollout_started = time.perf_counter()
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(config.wait_steps):
        observation, _, _, _ = env.step(runtime.get_libero_dummy_action("openvla"))

    resize_size = runtime.get_image_resize_size(model_config)
    hidden_history = []
    observation_history: list[dict[str, Any]] = []
    policy_images = []
    replay_images = []
    rows: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    simulator_seconds: list[float] = []
    success = False
    replay_maximum_error = 0.0
    replayed_steps = 0
    if fault_injector is not None:
        fault_injector.begin_trial(trial_seed)
    intervention_step = None
    if fault_injector is not None:
        intervention_step = fault_injector.spec.policy_step
    elif stale_image_spec is not None:
        intervention_step = stale_image_spec.policy_step
    image_intervention_record = None

    for policy_step in range(MAX_STEPS[config.task_suite]):
        if (
            clean_trace is not None
            and intervention_step is not None
            and policy_step <= intervention_step
        ):
            error = observation_error(runtime.np, clean_trace, observation, policy_step)
            replay_maximum_error = max(replay_maximum_error, error)
            if error > REPLAY_OBSERVATION_TOLERANCE:
                raise CounterfactualReplayDivergence(policy_step, error)

        if (
            clean_trace is not None
            and intervention_step is not None
            and policy_step < intervention_step
        ):
            image = runtime.get_libero_image(observation, resize_size)
            policy_images.append(image.copy())
            if config.save_video:
                replay_images.append(image)
            hidden_history.append(
                runtime.torch.as_tensor(clean_trace.hidden_states[policy_step])
                .detach()
                .cpu()
            )
            observation_history.append(_numeric_observation(runtime, observation))
            row = dict(clean_trace.rows[policy_step])
            row["timing/inference_seconds"] = 0.0
            row["fault/injected"] = False
            action = replay_action(runtime.np, clean_trace, policy_step)

            if evidence is not None:
                evidence.begin_step(
                    policy_step,
                    observation,
                    image,
                    image,
                    policy_step,
                    None,
                    policy_replayed=True,
                )
                action = evidence.replayed_evidence(
                    policy_step,
                    action,
                    runtime.torch.as_tensor(
                        clean_trace.hidden_states[policy_step, -1, :]
                    ),
                )

            simulator_started = time.perf_counter()
            if evidence is not None:
                observation, reward, done, _ = evidence.environment_step(
                    env, action, policy_step, policy_replayed=True
                )
            else:
                observation, reward, done, _ = env.step(action.tolist())
            simulator_seconds.append(time.perf_counter() - simulator_started)
            row["timing/simulator_seconds"] = simulator_seconds[-1]
            row["environment/reward"] = reward
            row["environment/done"] = bool(done)
            rows.append(row)
            if evidence is not None:
                evidence.finish_step(
                    policy_step,
                    fault_applied=False,
                    reward=reward,
                    done=bool(done),
                )
            replayed_steps += 1
            if done:
                raise CounterfactualReplayTerminated(policy_step, intervention_step)
            continue

        image = runtime.get_libero_image(observation, resize_size)
        policy_image = image
        if stale_image_spec is not None and policy_step == stale_image_spec.policy_step:
            if stale_image_spec.source_policy_step != (
                stale_image_spec.policy_step - stale_image_spec.image_lag
            ):
                raise RuntimeError("stale-image source policy step does not match the lag")
            if stale_image_spec.source_policy_step >= len(policy_images):
                raise RuntimeError(
                    f"stale-image source step {stale_image_spec.source_policy_step} "
                    f"is unavailable at policy step {policy_step}"
                )
            if config.image_input_mode == "stale":
                policy_image = policy_images[
                    stale_image_spec.source_policy_step
                ].copy()
            image_intervention_record = build_image_intervention_record(
                stale_image_spec, config.image_input_mode, trial_seed
            )
        state = runtime.np.concatenate(
            (
                observation["robot0_eef_pos"],
                runtime.quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        )
        policy_observation = {"full_image": policy_image, "state": state}

        intervention = None
        source_step = policy_step
        if stale_image_spec is not None and policy_step == stale_image_spec.policy_step:
            intervention = image_intervention_record
            if config.image_input_mode == "stale":
                source_step = stale_image_spec.source_policy_step
        if evidence is not None:
            evidence.begin_step(
                policy_step,
                observation,
                image,
                policy_image,
                source_step,
                intervention,
            )

        runtime.torch.cuda.synchronize()
        inference_started = time.perf_counter()
        fault_context = (
            fault_injector.inference(policy_step)
            if fault_injector is not None
            else nullcontext()
        )
        evidence_model = (
            evidence.policy_model(model, policy_step)
            if evidence is not None
            else model
        )
        evidence_processor = (
            evidence.processor(processor, policy_image, task_description, policy_step)
            if evidence is not None
            else processor
        )
        with runtime.torch.inference_mode(), fault_context:
            raw_actions, generated = runtime.get_action(
                model_config,
                evidence_model,
                policy_observation,
                task_description,
                processor=evidence_processor,
                n_samples=1,
            )
        runtime.torch.cuda.synchronize()
        inference_seconds.append(time.perf_counter() - inference_started)

        raw_action = runtime.np.asarray(raw_actions).copy()
        if raw_action.shape != (7,):
            raise ValueError(f"unexpected OpenVLA action shape: {raw_action.shape}")
        if evidence is not None:
            evidence.policy_outputs(
                model,
                generated,
                raw_actions,
                model_config.unnorm_key,
                policy_step,
                runtime.torch,
            )
            action = evidence.command(runtime, raw_action, policy_step)
        else:
            action = runtime.normalize_gripper_action(raw_action.copy(), binarize=True)
            action = runtime.invert_gripper_action(action)

        hidden_history.append(_extract_hidden_states(runtime, generated))
        observation_history.append(_numeric_observation(runtime, observation))
        policy_images.append(image.copy())
        if config.save_video:
            replay_images.append(image)

        metrics = runtime.compute_token_uncertainty_metrics(generated, model)
        row: dict[str, Any] = {
            "action/timestep": policy_step,
            "action/dx": action[0],
            "action/dy": action[1],
            "action/dz": action[2],
            "action/droll": action[3],
            "action/dpitch": action[4],
            "action/dyaw": action[5],
            "action/dgripper": action[6],
            "raw_action/dx": raw_action[0],
            "raw_action/dy": raw_action[1],
            "raw_action/dz": raw_action[2],
            "raw_action/droll": raw_action[3],
            "raw_action/dpitch": raw_action[4],
            "raw_action/dyaw": raw_action[5],
            "raw_action/dgripper": raw_action[6],
            "robot/eef_x": observation["robot0_eef_pos"][0],
            "robot/eef_y": observation["robot0_eef_pos"][1],
            "robot/eef_z": observation["robot0_eef_pos"][2],
            "timing/inference_seconds": inference_seconds[-1],
        }
        row.update({f"action/{key}": value for key, value in metrics.items()})
        if fault_injector is not None:
            record = fault_injector.record
            row["fault/injected"] = bool(
                record is not None and record["policy_step"] == policy_step
            )
        elif stale_image_spec is not None:
            row["fault/injected"] = image_fault_applied(
                stale_image_spec, config.image_input_mode, policy_step
            )

        simulator_started = time.perf_counter()
        if evidence is not None:
            observation, reward, done, _ = evidence.environment_step(
                env, action, policy_step
            )
        else:
            observation, reward, done, _ = env.step(action.tolist())
        simulator_seconds.append(time.perf_counter() - simulator_started)
        row["timing/simulator_seconds"] = simulator_seconds[-1]
        row["environment/reward"] = reward
        row["environment/done"] = bool(done)
        rows.append({key: _python_value(runtime, value) for key, value in row.items()})
        if evidence is not None:
            evidence.finish_step(
                policy_step,
                fault_applied=bool(row.get("fault/injected", False)),
                reward=reward,
                done=bool(done),
            )

        if done:
            success = True
            break

    rollout_seconds = time.perf_counter() - rollout_started
    fault_record = None
    if fault_injector is not None:
        fault_record = fault_injector.require_injected()
    elif stale_image_spec is not None:
        if image_intervention_record is None:
            raise RuntimeError("image intervention was never applied")
        fault_record = image_intervention_record
    condition = run_condition if fault_record else "clean"
    evidence_result = None

    stem = safe_stem(trial, success)
    csv_path = config.output_dir / f"{stem}.csv"
    pickle_path = config.output_dir / f"{stem}.pkl"
    video_path = config.output_dir / f"{stem}.mp4"

    artifact_started = time.perf_counter()
    write_csv_atomic(csv_path, rows)
    if config.save_video:
        temporary_video = temporary_path(video_path)
        try:
            runtime.save_video(replay_images, temporary_video)
            temporary_video.replace(video_path)
        finally:
            temporary_video.unlink(missing_ok=True)

    hidden_states = runtime.torch.stack(hidden_history, dim=0)
    write_pickle_atomic(
        pickle_path,
        {
            "hidden_states": hidden_states,
            "observations": _stack_observations(runtime, observation_history),
            "condition": condition,
            "fault": fault_record,
            "task_suite_name": config.task_suite,
            "task_id": trial.task_id,
            "task_description": task_description,
            "episode_idx": trial.episode_index,
            "episode_success": success,
            "trial_seed": trial_seed,
            "initial_state_sha256": initial_state_sha256,
            "mp4_path": str(video_path) if config.save_video else None,
        },
    )
    artifact_seconds = time.perf_counter() - artifact_started
    if evidence is not None:
        evidence_result = evidence.close(
            success=success,
            policy_steps=len(rows),
            fault=fault_record,
        )
        evidence_result["directory_relative_to_run"] = os.path.relpath(
            evidence.output_dir, config.output_dir
        )

    result = {
        "schema_version": 1,
        "status": "complete",
        "condition": condition,
        "task_suite_name": config.task_suite,
        "task_id": trial.task_id,
        "task_description": task_description,
        "episode_index": trial.episode_index,
        "trial_seed": trial_seed,
        "initial_state_sha256": initial_state_sha256,
        "success": success,
        "policy_steps": len(rows),
        "maximum_policy_steps": MAX_STEPS[config.task_suite],
        "rollout_seconds": rollout_seconds,
        "inference_seconds": sum(inference_seconds),
        "mean_inference_seconds": sum(inference_seconds) / len(inference_seconds),
        "simulator_seconds": sum(simulator_seconds),
        "artifact_seconds": artifact_seconds,
        "peak_cuda_memory_bytes": runtime.torch.cuda.max_memory_allocated(),
        "execution": execution,
        "fault": fault_record,
        "counterfactual_replay": (
            {
                "enabled": True,
                "replayed_policy_steps": replayed_steps,
                "policy_inferences": len(inference_seconds),
                "maximum_numeric_observation_error": replay_maximum_error,
                "observation_tolerance": REPLAY_OBSERVATION_TOLERANCE,
                "clean_source_directory": str(clean_trace.source_dir),
            }
            if clean_trace is not None
            else {"enabled": False}
        ),
        "evidence_graph": evidence_result,
        "files": {
            "csv": csv_path.name,
            "pickle": pickle_path.name,
            "video": video_path.name if config.save_video else None,
        },
    }
    write_json_atomic(completion_path(config.output_dir, trial), result)
    return result
