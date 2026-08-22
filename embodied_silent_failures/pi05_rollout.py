from __future__ import annotations

import collections
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from embodied_silent_failures.artifacts import (
    artifact_record,
    completion_path,
    safe_stem,
    temporary_path,
    write_csv_atomic,
    write_json_atomic,
    write_pickle_atomic,
)
from embodied_silent_failures.pi05_contract import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    DIFFUSION_STEPS,
    ENVIRONMENT_RESOLUTION,
    IMAGE_SIZE,
    LIBERO_ACTION_DIMENSION,
    MAX_STEPS,
    PROTOCOL_VERSION,
    decision_noise_seed,
    validate_replan_steps,
)
from embodied_silent_failures.pi05_policy import PARITY_ATOL, REQUEST_KEY
from embodied_silent_failures.plan import Trial


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclass(frozen=True)
class RolloutConfig:
    output_dir: Path
    task_suite: str
    base_seed: int
    wait_steps: int
    replan_steps: int
    save_video: bool
    compare_reference_first_decision: bool = False


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("cannot hash an array containing Python objects")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def quaternion_to_axis_angle(quaternion: Any) -> Any:
    import numpy as np

    # Physical Intelligence openpi commit 15a9616, examples/libero/main.py::_quat2axisangle,
    # copies robosuite transform_utils.py at commit eafb81f and clips the scalar
    # quaternion component before applying the axis-angle conversion below.
    quaternion = np.asarray(quaternion).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator


def numeric_observation(observation: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    values = {}
    for key, value in observation.items():
        if "image" in key.lower():
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or array.size > 4096:
            continue
        values[key] = array.copy()
    return values


def stack_observations(history: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    common = set(history[0]) if history else set()
    for observation in history[1:]:
        common.intersection_update(observation)
    return {
        key: np.stack([observation[key] for observation in history])
        for key in sorted(common)
    }


def validate_policy_response(
    response: dict[str, Any],
    *,
    decision_index: int,
    noise_seed: int,
    compare_reference: bool,
) -> dict[str, Any]:
    import numpy as np

    if not isinstance(response, dict):
        raise TypeError("pi0.5 server response is not an object")
    evidence = response.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("pi0.5 server response contains no SAFE evidence")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "decision_id": decision_index,
        "noise_seed": noise_seed,
        "completed_diffusion_steps": DIFFUSION_STEPS,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(
                f"pi0.5 response {key} is {evidence.get(key)!r}, expected {value!r}"
            )

    arrays = {
        "actions": np.asarray(response.get("actions")),
        "raw_actions": np.asarray(response.get("raw_actions")),
        "pre_velocity": np.asarray(evidence.get("pre_velocity")),
        "sampling_noise": np.asarray(evidence.get("sampling_noise")),
    }
    expected_shapes = {
        "actions": (ACTION_HORIZON, LIBERO_ACTION_DIMENSION),
        "raw_actions": (ACTION_HORIZON, ACTION_DIMENSION),
        "sampling_noise": (ACTION_HORIZON, ACTION_DIMENSION),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"pi0.5 {name} has shape {arrays[name].shape}, expected {shape}"
            )
    if arrays["pre_velocity"].ndim != 3 or arrays["pre_velocity"].shape[:2] != (
        DIFFUSION_STEPS,
        ACTION_HORIZON,
    ):
        raise ValueError(
            "pi0.5 pre_velocity must have shape "
            f"({DIFFUSION_STEPS}, {ACTION_HORIZON}, feature_width), got "
            f"{arrays['pre_velocity'].shape}"
        )
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"pi0.5 {name} contains a non-finite value")

    comparison = response.get("reference_comparison")
    if compare_reference:
        if not isinstance(comparison, dict):
            raise ValueError("requested reference comparison is absent")
        maximum_error = comparison.get("maximum_absolute_raw_action_error")
        if comparison.get("passed") is not True or not isinstance(
            maximum_error, (int, float)
        ):
            raise ValueError(f"instrumented sampler failed action parity: {comparison}")
        if maximum_error > PARITY_ATOL:
            raise ValueError(
                f"instrumented sampler exceeded parity tolerance: {comparison}"
            )
    elif comparison is not None:
        raise ValueError("server returned an unrequested reference comparison")
    return arrays


def _policy_input(
    observation: dict[str, Any], task_description: str
) -> tuple[Any, Any, Any]:
    import numpy as np
    from openpi_client import image_tools

    # Physical Intelligence openpi commit 15a9616, examples/libero/main.py::eval_libero,
    # rotates both LIBERO cameras by 180 degrees, then resizes with padding to
    # 224x224 uint8 before constructing the policy request.
    image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, IMAGE_SIZE, IMAGE_SIZE)
    )
    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quaternion_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    element = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(task_description),
    }
    return element, image, wrist_image


def _get_libero_env(task: Any, seed: int) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=ENVIRONMENT_RESOLUTION,
        camera_widths=ENVIRONMENT_RESOLUTION,
    )
    env.seed(seed)
    return env, task_description


def _write_video(path: Path, images: list[Any]) -> None:
    import imageio
    import numpy as np

    if not images:
        raise ValueError("cannot write a video with no frames")
    pending = temporary_path(path)
    try:
        imageio.mimwrite(pending, [np.asarray(image) for image in images], fps=10)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def run_trial(
    config: RolloutConfig,
    client: Any,
    trial: Trial,
    task: Any,
    initial_state: Any,
    execution: dict[str, Any],
    heartbeat: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import numpy as np

    validate_replan_steps(config.replan_steps)
    if config.task_suite not in MAX_STEPS:
        raise ValueError(f"unsupported LIBERO suite: {config.task_suite}")
    if config.wait_steps < 0:
        raise ValueError("wait steps must be non-negative")

    env, task_description = _get_libero_env(task, config.base_seed)
    rollout_started = time.monotonic()
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        for _ in range(config.wait_steps):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        action_plan = collections.deque()
        decisions = []
        observation_history = []
        replay_images = []
        rows = []
        success = False
        for environment_step in range(MAX_STEPS[config.task_suite]):
            element, image, wrist_image = _policy_input(observation, task_description)
            if config.save_video:
                replay_images.append(image)
            observation_history.append(numeric_observation(observation))

            if not action_plan:
                decision_index = len(decisions)
                noise_seed = decision_noise_seed(
                    config.base_seed, trial, decision_index
                )
                compare_reference = (
                    config.compare_reference_first_decision and decision_index == 0
                )
                element[REQUEST_KEY] = {
                    "decision_id": decision_index,
                    "noise_seed": noise_seed,
                    "compare_reference": compare_reference,
                }
                inference_started = time.monotonic()
                response = client.infer(element)
                client_inference_ms = (time.monotonic() - inference_started) * 1000
                arrays = validate_policy_response(
                    response,
                    decision_index=decision_index,
                    noise_seed=noise_seed,
                    compare_reference=compare_reference,
                )
                for offset, action in enumerate(
                    arrays["actions"][: config.replan_steps]
                ):
                    action_plan.append((decision_index, offset, action.copy()))
                decisions.append(
                    {
                        "environment_step": environment_step,
                        "image": image.copy(),
                        "wrist_image": wrist_image.copy(),
                        "state": np.asarray(element["observation/state"]).copy(),
                        "noise_seed": noise_seed,
                        "sampling_noise": arrays["sampling_noise"].copy(),
                        "raw_action_chunk": arrays["raw_actions"].copy(),
                        "action_chunk": arrays["actions"].copy(),
                        "pre_velocity": arrays["pre_velocity"].copy(),
                        "client_inference_ms": client_inference_ms,
                        "policy_timing": response.get("policy_timing"),
                        "server_timing": response.get("server_timing"),
                        "reference_comparison": response.get("reference_comparison"),
                    }
                )
                if heartbeat is not None:
                    heartbeat(
                        {
                            "state": "running",
                            "task_id": trial.task_id,
                            "episode_index": trial.episode_index,
                            "environment_step": environment_step,
                            "model_decisions": len(decisions),
                        }
                    )

            decision_index, chunk_offset, action = action_plan.popleft()
            source_action = decisions[decision_index]["action_chunk"][chunk_offset]
            if not np.array_equal(action, source_action):
                raise ValueError("queued action no longer matches its recorded chunk")
            simulator_started = time.monotonic()
            observation, reward, done, _ = env.step(action.tolist())
            simulator_ms = (time.monotonic() - simulator_started) * 1000
            rows.append(
                {
                    "environment_step": environment_step,
                    "decision_index": decision_index,
                    "chunk_offset": chunk_offset,
                    **{
                        f"action_{index}": float(value)
                        for index, value in enumerate(action)
                    },
                    "reward": float(reward),
                    "done": bool(done),
                    "simulator_ms": simulator_ms,
                }
            )
            if done:
                success = True
                break
    finally:
        env.close()

    if not rows or not decisions:
        raise ValueError("rollout produced no policy-controlled environment steps")
    for row in rows:
        expected = decisions[row["decision_index"]]["action_chunk"][row["chunk_offset"]]
        recorded = np.asarray(
            [row[f"action_{index}"] for index in range(LIBERO_ACTION_DIMENSION)]
        )
        if not np.array_equal(recorded, expected):
            raise ValueError(
                "environment action does not match its recorded policy chunk"
            )

    rollout_seconds = time.monotonic() - rollout_started
    artifact_started = time.monotonic()
    stem = safe_stem(trial, success)
    csv_path = config.output_dir / f"{stem}.csv"
    pickle_path = config.output_dir / f"{stem}.pkl"
    video_path = config.output_dir / f"{stem}.mp4"
    write_csv_atomic(csv_path, rows)
    decision_arrays = {
        "environment_steps": np.asarray(
            [item["environment_step"] for item in decisions], dtype=np.int32
        ),
        "images": np.stack([item["image"] for item in decisions]),
        "wrist_images": np.stack([item["wrist_image"] for item in decisions]),
        "states": np.stack([item["state"] for item in decisions]),
        "noise_seeds": np.asarray(
            [item["noise_seed"] for item in decisions], dtype=np.uint32
        ),
        "sampling_noise": np.stack([item["sampling_noise"] for item in decisions]),
        "raw_action_chunks": np.stack([item["raw_action_chunk"] for item in decisions]),
        "action_chunks": np.stack([item["action_chunk"] for item in decisions]),
        "pre_velocity": np.stack([item["pre_velocity"] for item in decisions]),
        "client_inference_ms": np.asarray(
            [item["client_inference_ms"] for item in decisions], dtype=np.float64
        ),
        "timing": [
            {
                "policy": item["policy_timing"],
                "server": item["server_timing"],
            }
            for item in decisions
        ],
        "reference_comparisons": [item["reference_comparison"] for item in decisions],
    }
    step_arrays = {
        "environment_steps": np.asarray(
            [row["environment_step"] for row in rows], dtype=np.int32
        ),
        "decision_indices": np.asarray(
            [row["decision_index"] for row in rows], dtype=np.int32
        ),
        "chunk_offsets": np.asarray(
            [row["chunk_offset"] for row in rows], dtype=np.int16
        ),
        "actions": np.asarray(
            [
                [row[f"action_{index}"] for index in range(LIBERO_ACTION_DIMENSION)]
                for row in rows
            ]
        ),
        "rewards": np.asarray([row["reward"] for row in rows]),
        "done": np.asarray([row["done"] for row in rows]),
        "numeric_observations": stack_observations(observation_history),
    }
    write_pickle_atomic(
        pickle_path,
        {
            "schema_version": 1,
            "model": "pi0.5",
            "condition": "clean",
            "task_suite_name": config.task_suite,
            "task_id": trial.task_id,
            "task_description": task_description,
            "episode_idx": trial.episode_index,
            "episode_success": success,
            "initial_state_sha256": array_sha256(initial_state),
            "replan_steps": config.replan_steps,
            "decisions": decision_arrays,
            "environment": step_arrays,
            "execution": execution,
        },
    )
    if config.save_video:
        _write_video(video_path, replay_images)

    artifact_paths = [csv_path, pickle_path]
    if config.save_video:
        artifact_paths.append(video_path)
    artifact_manifest = [artifact_record(path) for path in artifact_paths]
    artifact_seconds = time.monotonic() - artifact_started
    result = {
        "schema_version": 1,
        "status": "complete",
        "condition": "clean",
        "model": "pi0.5",
        "task_suite_name": config.task_suite,
        "task_id": trial.task_id,
        "task_description": task_description,
        "episode_index": trial.episode_index,
        "initial_state_sha256": array_sha256(initial_state),
        "success": success,
        "environment_steps": len(rows),
        "maximum_environment_steps": MAX_STEPS[config.task_suite],
        "model_decisions": len(decisions),
        "action_horizon": ACTION_HORIZON,
        "replan_steps": config.replan_steps,
        "rollout_seconds": rollout_seconds,
        "inference_seconds": sum(item["client_inference_ms"] for item in decisions)
        / 1000,
        "simulator_seconds": sum(row["simulator_ms"] for row in rows) / 1000,
        "artifact_seconds": artifact_seconds,
        "execution": execution,
        "files": {
            "csv": csv_path.name,
            "pickle": pickle_path.name,
            "video": video_path.name if config.save_video else None,
        },
        "artifact_manifest": artifact_manifest,
    }
    write_json_atomic(completion_path(config.output_dir, trial), result)
    if heartbeat is not None:
        heartbeat(
            {
                "state": "complete",
                "task_id": trial.task_id,
                "episode_index": trial.episode_index,
                "environment_steps": len(rows),
                "model_decisions": len(decisions),
                "success": success,
            }
        )
    return result
