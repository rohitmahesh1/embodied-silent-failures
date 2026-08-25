from __future__ import annotations

import collections
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from embodied_silent_failures.artifacts import (
    artifact_record,
    write_csv_atomic,
    write_json_atomic,
    write_pickle_atomic,
)
from embodied_silent_failures.freshness import observe_freshness
from embodied_silent_failures.pi05_contract import MAX_STEPS, decision_noise_seed
from embodied_silent_failures.pi05_policy import REQUEST_KEY
from embodied_silent_failures.pi05_rollout import (
    LIBERO_DUMMY_ACTION,
    _get_libero_env,
    _policy_input,
    _write_video,
    array_sha256,
    numeric_observation,
    stack_observations,
    validate_policy_response,
)
from embodied_silent_failures.pi05_safe_data import (
    FEATURE_PROTOCOL,
    reduce_pre_velocity,
)
from embodied_silent_failures.pi05_stale_manifest import Pi05StaleSpec
from embodied_silent_failures.plan import Trial


PAIR_CONDITIONS = ("current_current_null", "stale_main_camera")
REPLAY_OBSERVATION_TOLERANCE = 1e-6


class PrefixTerminated(RuntimeError):
    pass


@dataclass(frozen=True)
class PairConfig:
    output_dir: Path
    task_suite: str
    base_seed: int
    wait_steps: int
    replan_steps: int
    save_video: bool
    pair_condition: str


def branch_order(pair_condition: str, order_bit: int) -> tuple[str, str]:
    if order_bit not in (0, 1):
        raise ValueError("order bit must be zero or one")
    if pair_condition == "stale_main_camera":
        labels = ("current", "stale")
    elif pair_condition == "current_current_null":
        labels = ("current_a", "current_b")
    else:
        raise ValueError(f"unknown pi0.5 pair condition: {pair_condition}")
    return labels if order_bit == 0 else tuple(reversed(labels))


def pair_directory(output_dir: Path, trial: Trial) -> Path:
    return output_dir / "pairs" / f"task{trial.task_id}--ep{trial.episode_index}"


def pair_completion_path(output_dir: Path, trial: Trial) -> Path:
    return pair_directory(output_dir, trial) / "pair.complete.json"


def pair_terminal_state(
    output_dir: Path, trial: Trial
) -> Literal["complete", "excluded"] | None:
    directory = pair_directory(output_dir, trial)
    if (directory / "pair.complete.json").is_file():
        return "complete"
    if (directory / "pair.excluded.json").is_file():
        return "excluded"
    return None


def prepare_pair(
    output_dir: Path, trial: Trial, resume: bool
) -> Literal["complete", "excluded"] | None:
    directory = pair_directory(output_dir, trial)
    completion = directory / "pair.complete.json"
    if completion.is_file():
        if not resume:
            raise FileExistsError(f"pi0.5 pair is already complete: {directory}")
        value = json.loads(completion.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            raise ValueError(f"invalid pi0.5 pair completion marker: {completion}")
        if (
            value.get("task_id") != trial.task_id
            or value.get("episode_index") != trial.episode_index
        ):
            raise ValueError(f"pi0.5 pair completion has the wrong trial: {completion}")
        branches = value.get("branches")
        condition = value.get("pair_condition")
        expected_labels = (
            {"current", "stale"}
            if condition == "stale_main_camera"
            else {"current_a", "current_b"}
            if condition == "current_current_null"
            else set()
        )
        if not isinstance(branches, dict) or set(branches) != expected_labels:
            raise ValueError(f"pi0.5 pair completion has invalid branches: {completion}")
        for branch in branches.values():
            for record in branch.get("artifact_manifest", []):
                path = directory / record["name"]
                if artifact_record(path) != record:
                    raise ValueError(f"pi0.5 pair artifact disagrees: {path}")
        return "complete"
    exclusion = directory / "pair.excluded.json"
    if exclusion.is_file():
        if not resume:
            raise FileExistsError(f"pi0.5 pair is already excluded: {directory}")
        value = json.loads(exclusion.read_text(encoding="utf-8"))
        if value.get("status") != "excluded":
            raise ValueError(f"invalid pi0.5 pair exclusion marker: {exclusion}")
        if (
            value.get("task_id") != trial.task_id
            or value.get("episode_index") != trial.episode_index
        ):
            raise ValueError(f"pi0.5 pair exclusion has the wrong trial: {exclusion}")
        return "excluded"
    if directory.exists():
        shutil.rmtree(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    for partial in directory.parent.glob(f".{directory.name}.*.tmp"):
        if partial.is_dir():
            shutil.rmtree(partial)
    return None


def _numeric_error(np: Any, expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if set(expected) != set(actual):
        raise ValueError("replayed numeric observation keys changed")
    maximum = 0.0
    for key in expected:
        left = np.asarray(expected[key])
        right = np.asarray(actual[key])
        if left.shape != right.shape:
            raise ValueError(f"replayed numeric observation {key} changed shape")
        if left.size:
            maximum = max(
                maximum,
                float(np.max(np.abs(left.astype(float) - right.astype(float)))),
            )
    return maximum


def _image_difference(np: Any, expected: Any, actual: Any) -> dict[str, Any]:
    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.shape != right.shape:
        return {
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    changed_pixels = np.any(delta != 0, axis=-1)
    return {
        "maximum_absolute_channel_error": int(delta.max()),
        "mean_absolute_channel_error": float(delta.mean()),
        "changed_pixels": int(changed_pixels.sum()),
        "total_pixels": int(changed_pixels.size),
    }


def _image_divergence(
    np: Any,
    *,
    label: str,
    camera: str,
    environment_step: int,
    expected: Any,
    actual: Any,
) -> RuntimeError:
    detail = json.dumps(
        _image_difference(np, expected, actual), sort_keys=True, separators=(",", ":")
    )
    return RuntimeError(
        f"branch {label} {camera} camera diverged before intervention at "
        f"environment step {environment_step}: {detail}"
    )


def _infer(
    np: Any,
    client: Any,
    element: dict[str, Any],
    trial: Trial,
    decision_index: int,
    environment_step: int,
    base_seed: int,
) -> dict[str, Any]:
    noise_seed = decision_noise_seed(base_seed, trial, decision_index)
    element[REQUEST_KEY] = {
        "decision_id": decision_index,
        "noise_seed": noise_seed,
        "compare_reference": False,
    }
    started = time.monotonic()
    response = client.infer(element)
    client_inference_ms = (time.monotonic() - started) * 1000
    arrays = validate_policy_response(
        response,
        decision_index=decision_index,
        noise_seed=noise_seed,
        compare_reference=False,
    )
    selected = reduce_pre_velocity(arrays["pre_velocity"][None], np)[0]
    return {
        "decision_index": decision_index,
        "environment_step": environment_step,
        "noise_seed": noise_seed,
        "sampling_noise": arrays["sampling_noise"].copy(),
        "raw_action_chunk": arrays["raw_actions"].copy(),
        "action_chunk": arrays["actions"].copy(),
        "safe_feature": selected,
        "client_inference_ms": client_inference_ms,
        "policy_timing": response.get("policy_timing"),
        "server_timing": response.get("server_timing"),
    }


def _prefix(
    np: Any,
    config: PairConfig,
    client: Any,
    trial: Trial,
    task: Any,
    initial_state: Any,
    spec: Pi05StaleSpec,
) -> dict[str, Any]:
    env, task_description = _get_libero_env(task, config.base_seed)
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(config.wait_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        decisions = []
        actions = []
        observations = []
        image_hashes = []
        wrist_hashes = []
        image_frames = []
        wrist_frames = []
        environment_step = 0
        for decision_index in range(spec.intervention_decision):
            element, image, wrist = _policy_input(observation, task_description)
            record = _infer(
                np,
                client,
                element,
                trial,
                decision_index,
                environment_step,
                config.base_seed,
            )
            record["policy_image"] = image.copy()
            record["policy_wrist_image"] = wrist.copy()
            decisions.append(record)
            for action in record["action_chunk"][: config.replan_steps]:
                _, step_image, step_wrist = _policy_input(observation, task_description)
                observations.append(numeric_observation(observation))
                image_hashes.append(array_sha256(step_image))
                wrist_hashes.append(array_sha256(step_wrist))
                image_frames.append(step_image.copy())
                wrist_frames.append(step_wrist.copy())
                actions.append(action.copy())
                observation, _, done, _ = env.step(action.tolist())
                environment_step += 1
                if done:
                    raise PrefixTerminated(
                        "live common prefix reached task success before intervention"
                    )

        _, current_image, current_wrist = _policy_input(
            observation, task_description
        )
        return {
            "task_description": task_description,
            "decisions": decisions,
            "actions": actions,
            "observations": observations,
            "image_hashes": image_hashes,
            "wrist_hashes": wrist_hashes,
            "image_frames": image_frames,
            "wrist_frames": wrist_frames,
            "target_numeric": numeric_observation(observation),
            "target_image": current_image.copy(),
            "target_wrist": current_wrist.copy(),
            "previous_policy_image": decisions[-1]["policy_image"].copy(),
            "environment_step": environment_step,
        }
    finally:
        env.close()


def _freshness_record(
    np: Any,
    *,
    decision_index: int,
    current_image: Any,
    selected_image: Any,
    previous_policy_image: Any | None,
    source_decision: int,
    replan_steps: int,
) -> dict[str, Any]:
    signals = observe_freshness(
        np,
        policy_step=decision_index,
        policy_image=selected_image,
        previous_policy_image=previous_policy_image,
        image_source_policy_step=source_decision,
    )
    record = signals.intervention_record("either", response_applied=False)
    return {
        **record,
        "units": "policy decisions",
        "source_age_policy_decisions": decision_index - source_decision,
        "source_age_environment_steps": (
            decision_index - source_decision
        )
        * replan_steps,
        "current_camera_sha256": array_sha256(current_image),
        "selected_camera_sha256": array_sha256(selected_image),
        "response": "shadow observation only",
    }


def _run_branch(
    np: Any,
    config: PairConfig,
    client: Any,
    trial: Trial,
    task: Any,
    initial_state: Any,
    spec: Pi05StaleSpec,
    prefix: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    stale = label == "stale"
    env, task_description = _get_libero_env(task, config.base_seed)
    rollout_started = time.monotonic()
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(config.wait_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        replay_maximum_error = 0.0
        video_images = []
        rows = []
        observation_history = []
        for step, action in enumerate(prefix["actions"]):
            _, image, wrist = _policy_input(observation, task_description)
            if config.save_video:
                video_images.append(image.copy())
            observation_history.append(numeric_observation(observation))
            error = _numeric_error(
                np, prefix["observations"][step], numeric_observation(observation)
            )
            replay_maximum_error = max(replay_maximum_error, error)
            if error > REPLAY_OBSERVATION_TOLERANCE:
                raise RuntimeError(
                    f"branch {label} replay diverged at environment step {step}: {error}"
                )
            if array_sha256(image) != prefix["image_hashes"][step]:
                raise _image_divergence(
                    np,
                    label=label,
                    camera="main",
                    environment_step=step,
                    expected=prefix["image_frames"][step],
                    actual=image,
                )
            if array_sha256(wrist) != prefix["wrist_hashes"][step]:
                raise _image_divergence(
                    np,
                    label=label,
                    camera="wrist",
                    environment_step=step,
                    expected=prefix["wrist_frames"][step],
                    actual=wrist,
                )
            observation, reward, done, _ = env.step(action.tolist())
            source_index, chunk_offset = divmod(step, config.replan_steps)
            rows.append(
                {
                    "environment_step": step,
                    "decision_index": source_index,
                    "chunk_offset": chunk_offset,
                    **{
                        f"action_{index}": float(value)
                        for index, value in enumerate(action)
                    },
                    "reward": float(reward),
                    "done": bool(done),
                }
            )
            if done:
                raise PrefixTerminated(
                    f"branch {label} replay reached success before intervention"
                )

        target_element, target_image, target_wrist = _policy_input(
            observation, task_description
        )
        target_error = _numeric_error(
            np, prefix["target_numeric"], numeric_observation(observation)
        )
        replay_maximum_error = max(replay_maximum_error, target_error)
        if target_error > REPLAY_OBSERVATION_TOLERANCE:
            raise RuntimeError(f"branch {label} target state diverged: {target_error}")
        if not np.array_equal(target_image, prefix["target_image"]):
            raise _image_divergence(
                np,
                label=label,
                camera="target main",
                environment_step=len(prefix["actions"]),
                expected=prefix["target_image"],
                actual=target_image,
            )
        if not np.array_equal(target_wrist, prefix["target_wrist"]):
            raise _image_divergence(
                np,
                label=label,
                camera="target wrist",
                environment_step=len(prefix["actions"]),
                expected=prefix["target_wrist"],
                actual=target_wrist,
            )

        decisions = [dict(record) for record in prefix["decisions"]]
        action_plan = collections.deque()
        previous_policy_image = prefix["previous_policy_image"]
        environment_step = int(prefix["environment_step"])
        decision_index = spec.intervention_decision
        success = False
        intervention = None
        while environment_step < MAX_STEPS[config.task_suite]:
            element, current_image, current_wrist = _policy_input(
                observation, task_description
            )
            selected_image = current_image
            source_decision = decision_index
            if stale and decision_index == spec.intervention_decision:
                selected_image = prefix["previous_policy_image"].copy()
                source_decision = spec.source_decision
                element["observation/image"] = selected_image
            freshness = _freshness_record(
                np,
                decision_index=decision_index,
                current_image=current_image,
                selected_image=selected_image,
                previous_policy_image=previous_policy_image,
                source_decision=source_decision,
                replan_steps=config.replan_steps,
            )
            record = _infer(
                np,
                client,
                element,
                trial,
                decision_index,
                environment_step,
                config.base_seed,
            )
            record["policy_image"] = selected_image.copy()
            record["policy_wrist_image"] = current_wrist.copy()
            record["freshness"] = freshness
            decisions.append(record)
            if decision_index == spec.intervention_decision:
                intervention = {
                    "kind": "stale_main_camera" if stale else "current_camera_control",
                    "decision_index": decision_index,
                    "environment_step": environment_step,
                    "source_decision": source_decision,
                    "main_camera_changed": stale,
                    "wrist_camera_changed": False,
                    "robot_state_changed": False,
                    "diffusion_noise_changed": False,
                    "freshness": freshness,
                    "current_main_camera": current_image.copy(),
                    "selected_main_camera": selected_image.copy(),
                    "current_wrist_camera": current_wrist.copy(),
                    "current_robot_state": np.asarray(
                        element["observation/state"]
                    ).copy(),
                }
            for offset, action in enumerate(
                record["action_chunk"][: config.replan_steps]
            ):
                action_plan.append((decision_index, offset, action.copy()))
            previous_policy_image = selected_image.copy()
            decision_index += 1

            while action_plan and environment_step < MAX_STEPS[config.task_suite]:
                source_index, chunk_offset, action = action_plan.popleft()
                observation_history.append(numeric_observation(observation))
                if config.save_video:
                    _, video_image, _ = _policy_input(observation, task_description)
                    video_images.append(video_image.copy())
                observation, reward, done, _ = env.step(action.tolist())
                rows.append(
                    {
                        "environment_step": environment_step,
                        "decision_index": source_index,
                        "chunk_offset": chunk_offset,
                        **{
                            f"action_{index}": float(value)
                            for index, value in enumerate(action)
                        },
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )
                environment_step += 1
                if done:
                    success = True
                    action_plan.clear()
                    break
            if success:
                break
        if intervention is None:
            raise RuntimeError(f"branch {label} never reached the intervention")
        return {
            "label": label,
            "success": success,
            "environment_steps": environment_step,
            "decisions": decisions,
            "rows": rows,
            "observations": observation_history,
            "video_images": video_images,
            "intervention": intervention,
            "replay_maximum_numeric_observation_error": replay_maximum_error,
            "rollout_seconds": time.monotonic() - rollout_started,
        }
    finally:
        env.close()


def _write_branch(
    np: Any, directory: Path, branch: dict[str, Any], save_video: bool
) -> dict[str, Any]:
    label = branch["label"]
    csv_path = directory / f"{label}.csv"
    pickle_path = directory / f"{label}.pkl"
    video_path = directory / f"{label}.mp4"
    rows = branch["rows"]
    if not rows:
        raise ValueError(f"pi0.5 branch {label} produced no post-intervention steps")
    write_csv_atomic(csv_path, rows)
    decisions = branch["decisions"]
    intervention = dict(branch["intervention"])
    intervention_images = {
        name: intervention.pop(name)
        for name in (
            "current_main_camera",
            "selected_main_camera",
            "current_wrist_camera",
            "current_robot_state",
        )
    }
    write_pickle_atomic(
        pickle_path,
        {
            "schema_version": 1,
            "feature_protocol": FEATURE_PROTOCOL,
            "label": label,
            "success": branch["success"],
            "safe_features": np.stack([item["safe_feature"] for item in decisions]),
            "decision_environment_steps": np.asarray(
                [item["environment_step"] for item in decisions], dtype=np.int32
            ),
            "noise_seeds": np.asarray(
                [item["noise_seed"] for item in decisions], dtype=np.uint32
            ),
            "sampling_noise": np.stack([item["sampling_noise"] for item in decisions]),
            "raw_action_chunks": np.stack(
                [item["raw_action_chunk"] for item in decisions]
            ),
            "action_chunks": np.stack([item["action_chunk"] for item in decisions]),
            "numeric_observations": stack_observations(branch["observations"]),
            "intervention": {**intervention, **intervention_images},
        },
    )
    if save_video:
        _write_video(video_path, branch["video_images"])
    paths = [csv_path, pickle_path] + ([video_path] if save_video else [])
    return {
        "label": label,
        "success": bool(branch["success"]),
        "environment_steps": int(branch["environment_steps"]),
        "model_decisions": len(decisions),
        "rollout_seconds": float(branch["rollout_seconds"]),
        "replay_maximum_numeric_observation_error": float(
            branch["replay_maximum_numeric_observation_error"]
        ),
        "intervention": intervention,
        "files": {
            "csv": csv_path.name,
            "pickle": pickle_path.name,
            "video": video_path.name if save_video else None,
        },
        "artifact_manifest": [artifact_record(path) for path in paths],
    }


def run_pair(
    config: PairConfig,
    client: Any,
    trial: Trial,
    task: Any,
    initial_state: Any,
    spec: Pi05StaleSpec,
    execution: dict[str, Any],
    heartbeat: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import numpy as np

    if config.pair_condition not in PAIR_CONDITIONS:
        raise ValueError(f"unknown pi0.5 pair condition: {config.pair_condition}")
    if config.replan_steps != spec.replan_steps:
        raise ValueError("runner and stale manifest disagree on replan_steps")
    if config.task_suite not in MAX_STEPS:
        raise ValueError(f"unsupported LIBERO suite: {config.task_suite}")

    final_dir = pair_directory(config.output_dir, trial)
    pending_dir = final_dir.parent / f".{final_dir.name}.{uuid4().hex}.tmp"
    pending_dir.mkdir(parents=True)
    try:
        common = _prefix(np, config, client, trial, task, initial_state, spec)
        labels = branch_order(config.pair_condition, spec.order_bit)
        branch_values = {}
        for index, label in enumerate(labels):
            if heartbeat is not None:
                heartbeat(
                    {
                        "state": "running",
                        "branch": label,
                        "branch_index": index,
                        "task_id": trial.task_id,
                        "episode_index": trial.episode_index,
                    }
                )
            branch = _run_branch(
                np, config, client, trial, task, initial_state, spec, common, label
            )
            branch_values[label] = _write_branch(
                np, pending_dir, branch, config.save_video
            )

        result = {
            "schema_version": 1,
            "status": "complete",
            "model": "pi0.5",
            "pair_condition": config.pair_condition,
            "task_suite_name": config.task_suite,
            **trial.to_dict(),
            "initial_state_sha256": array_sha256(initial_state),
            "intervention_decision": spec.intervention_decision,
            "intervention_environment_step": spec.intervention_environment_step,
            "source_decision": spec.source_decision,
            "replan_steps": config.replan_steps,
            "branch_order": list(labels),
            "common_prefix": {
                "policy_decisions": spec.intervention_decision,
                "environment_steps": len(common["actions"]),
                "generated_once_and_replayed_into_each_branch": True,
            },
            "execution": execution,
            "branches": branch_values,
        }
        write_json_atomic(pending_dir / "pair.complete.json", result)
        pending_dir.replace(final_dir)
        if heartbeat is not None:
            heartbeat(
                {
                    "state": "complete",
                    "task_id": trial.task_id,
                    "episode_index": trial.episode_index,
                    "branch_success": {
                        label: branch_values[label]["success"] for label in labels
                    },
                }
            )
        return result
    finally:
        if pending_dir.exists():
            shutil.rmtree(pending_dir)
