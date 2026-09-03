from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    safe_stem,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
    write_pickle_atomic,
)
from embodied_silent_failures.language_fault import (
    LanguageBlockInjector,
    LanguageInferenceTrace,
)
from embodied_silent_failures.language_policy import PolicyDecision, policy_decision
from embodied_silent_failures.language_trajectory_archive import TrajectoryRecorder
from embodied_silent_failures.openvla_rollout import MAX_STEPS
from embodied_silent_failures.openvla_runtime import Runtime, array_sha256
from embodied_silent_failures.plan import Trial


class ContextTerminatedEarly(RuntimeError):
    pass


@dataclass(frozen=True)
class CapturedContext:
    observation: dict[str, Any]
    simulator_state: Any
    simulator_state_sha256: str
    prefix_commands: tuple[Any, ...]
    prefix_hidden_states: tuple[Any, ...]
    prefix_rows: tuple[dict[str, Any], ...]
    source_trace: LanguageInferenceTrace | None
    source_decision: PolicyDecision | None = None


def write_captured_context_archive(
    path: Path,
    runtime: Runtime,
    captured: CapturedContext,
) -> dict[str, Any]:
    # language_context.py::capture_context copies env.get_sim_state() and the
    # policy observation immediately before intervention. Preserve those raw
    # arrays without assigning model- or task-specific semantic groups.
    arrays = {"simulator_state": runtime.np.asarray(captured.simulator_state)}
    observations = []
    for index, name in enumerate(sorted(captured.observation)):
        archive_key = f"observation_{index:04d}"
        value = runtime.np.asarray(captured.observation[name])
        if value.dtype.hasobject:
            raise TypeError(f"cannot archive object-valued observation {name!r}")
        arrays[archive_key] = value
        observations.append(
            {
                "name": str(name),
                "archive_key": archive_key,
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": array_sha256(runtime, value),
            }
        )
    write_npz_atomic(path, runtime.np, arrays)
    state = runtime.np.asarray(captured.simulator_state)
    return {
        "schema_version": 1,
        "format": "compressed NumPy archive readable with allow_pickle=False",
        "artifact": artifact_record(path),
        "simulator_state": {
            "archive_key": "simulator_state",
            "shape": list(state.shape),
            "dtype": state.dtype.str,
            "sha256": captured.simulator_state_sha256,
        },
        "observations": observations,
    }


def _action_row(policy_step: int, command: Any, decision: PolicyDecision) -> dict[str, Any]:
    row = {
        "action/timestep": policy_step,
        "action/dx": command[0],
        "action/dy": command[1],
        "action/dz": command[2],
        "action/droll": command[3],
        "action/dpitch": command[4],
        "action/dyaw": command[5],
        "action/dgripper": command[6],
        "timing/inference_seconds": decision.inference_seconds,
    }
    for index, value in enumerate(decision.raw_action):
        row[f"policy/raw_action_{index}"] = value
    for index, value in enumerate(decision.action_tokens):
        row[f"policy/action_token_{index}"] = value
    return row


def _start_episode(runtime: Runtime, env: Any, initial_state: Any, wait_steps: int) -> Any:
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        observation, _, _, _ = env.step(runtime.get_libero_dummy_action("openvla"))
    return observation


def action_row(
    policy_step: int, command: Any, decision: PolicyDecision
) -> dict[str, Any]:
    return _action_row(policy_step, command, decision)


def start_episode(runtime: Runtime, env: Any, initial_state: Any, wait_steps: int) -> Any:
    return _start_episode(runtime, env, initial_state, wait_steps)


def capture_context(
    runtime: Runtime,
    policy_config: Any,
    model: Any,
    processor: Any,
    injector: LanguageBlockInjector,
    env: Any,
    task_description: str,
    initial_state: Any,
    context: dict[str, Any],
    *,
    wait_steps: int,
    executed_prefix: tuple[Any, ...] | None = None,
    capture_internal_state: bool = False,
    capture_context_state: bool = False,
) -> CapturedContext:
    observation = _start_episode(runtime, env, initial_state, wait_steps)
    prefix_commands = []
    prefix_hidden_states = []
    prefix_rows = []
    source_trace = None
    source_decision = None
    source_step = int(context["source_policy_step"])
    token_position = int(context["action_token_position"])
    policy_steps = int(context["policy_step"])
    if executed_prefix is not None and len(executed_prefix) != policy_steps:
        raise ValueError("archived executed-command prefix has the wrong length")
    for policy_step in range(policy_steps):
        trace_source = policy_step == source_step
        decision = policy_decision(
            runtime,
            policy_config,
            model,
            processor,
            observation,
            task_description,
            injector=injector if trace_source else None,
            action_token_position=token_position if trace_source else None,
            capture_internal_state=capture_internal_state if trace_source else False,
            capture_context_state=capture_context_state if trace_source else False,
        )
        if trace_source:
            source_trace = decision.trace
            source_decision = decision
        command = runtime.np.asarray(
            decision.command
            if executed_prefix is None
            else executed_prefix[policy_step]
        ).copy()
        row = _action_row(policy_step, command, decision)
        observation, reward, done, _ = env.step(command.tolist())
        row["environment/reward"] = reward
        row["environment/done"] = bool(done)
        prefix_commands.append(command)
        prefix_hidden_states.append(decision.hidden_states)
        prefix_rows.append(row)
        if done:
            raise ContextTerminatedEarly(
                f"clean rerun succeeded at step {policy_step} before context "
                f"{context['context_id']}"
            )
    if source_trace is None or source_decision is None:
        raise RuntimeError("previous-decision language-block values were not captured")
    simulator_state = runtime.np.asarray(env.get_sim_state()).copy()
    return CapturedContext(
        observation={key: runtime.np.asarray(value).copy() for key, value in observation.items()},
        simulator_state=simulator_state,
        simulator_state_sha256=array_sha256(runtime, simulator_state),
        prefix_commands=tuple(prefix_commands),
        prefix_hidden_states=tuple(prefix_hidden_states),
        prefix_rows=tuple(prefix_rows),
        source_decision=source_decision,
        source_trace=source_trace,
    )


def _observation_drift(np: Any, reference: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(reference) & set(value))
    numeric_maximum = 0.0
    image_changed = 0
    image_maximum = 0.0
    compared_numeric = 0
    compared_images = 0
    for key in keys:
        left = np.asarray(reference[key])
        right = np.asarray(value[key])
        if left.shape != right.shape or left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf":
            continue
        difference = np.abs(left.astype(float) - right.astype(float))
        maximum = float(np.max(difference)) if difference.size else 0.0
        if "image" in key.lower():
            compared_images += 1
            image_changed += int(np.count_nonzero(difference))
            image_maximum = max(image_maximum, maximum)
        else:
            compared_numeric += 1
            numeric_maximum = max(numeric_maximum, maximum)
    return {
        "numeric_keys": compared_numeric,
        "image_keys": compared_images,
        "maximum_numeric_error": numeric_maximum,
        "changed_image_channels": image_changed,
        "maximum_image_channel_error": image_maximum,
    }


def replay_context(
    runtime: Runtime,
    env: Any,
    initial_state: Any,
    captured: CapturedContext,
    *,
    wait_steps: int,
    trial_seed: int,
) -> tuple[Any, dict[str, Any]]:
    # LIBERO@8f1084e, libero/envs/env_wrapper.py::get_sim_state, preserves only
    # MuJoCo's flattened state. Replaying through robosuite@1.4.1
    # environments/base.py::step also reconstructs the OSC controller cache and
    # GripperModel.current_action that a reset clears but the flattened state omits.
    # Reapply the seed used before the original reset so LIBERO's reset-time
    # randomization cannot change that hidden initial controller state.
    runtime.set_seed_everywhere(trial_seed)
    observation = _start_episode(runtime, env, initial_state, wait_steps)
    for replay_step, command in enumerate(captured.prefix_commands):
        observation, _, done, _ = env.step(command.tolist())
        if done:
            raise ContextTerminatedEarly(
                f"replayed prefix terminated at step {replay_step} before the "
                "captured intervention context"
            )
    replay_state = runtime.np.asarray(env.get_sim_state()).copy()
    state_difference = replay_state.astype(float) - captured.simulator_state.astype(float)
    return observation, {
        "method": "reset initial state and replay the captured executed-command prefix",
        "trial_seed_reapplied": trial_seed,
        "simulator_state_sha256": array_sha256(runtime, replay_state),
        "simulator_state_exact_equal": bool(
            runtime.np.array_equal(replay_state, captured.simulator_state)
        ),
        "simulator_state_l2": float(runtime.np.linalg.norm(state_difference)),
        "simulator_state_linf": float(runtime.np.max(runtime.np.abs(state_difference))),
        "observation": _observation_drift(runtime.np, captured.observation, observation),
    }


def restore_context(
    runtime: Runtime,
    env: Any,
    captured: CapturedContext,
) -> tuple[Any, dict[str, Any]]:
    # LIBERO@8f1084e, libero/envs/env_wrapper.py::regenerate_obs_from_state
    # restores a flattened MuJoCo state, forwards simulation, and regenerates
    # observations. reset clears episode bookkeeping left by the prior branch;
    # the captured MuJoCo state is then restored before any action is executed.
    env.reset()
    observation = env.regenerate_obs_from_state(captured.simulator_state)
    restored_state = runtime.np.asarray(env.get_sim_state()).copy()
    difference = restored_state.astype(float) - captured.simulator_state.astype(float)
    return observation, {
        "method": "restore the captured flattened MuJoCo state directly",
        "simulator_state_sha256": array_sha256(runtime, restored_state),
        "simulator_state_exact_equal": bool(
            runtime.np.array_equal(restored_state, captured.simulator_state)
        ),
        "simulator_state_l2": float(runtime.np.linalg.norm(difference)),
        "simulator_state_linf": float(runtime.np.max(runtime.np.abs(difference))),
        "observation": _observation_drift(
            runtime.np, captured.observation, observation
        ),
    }


def run_terminal_branch(
    runtime: Runtime,
    policy_config: Any,
    model: Any,
    processor: Any,
    env: Any,
    task_description: str,
    initial_state: Any,
    context: dict[str, Any],
    captured: CapturedContext,
    target_decision: PolicyDecision,
    *,
    wait_steps: int,
    restore_directly: bool = False,
) -> dict[str, Any]:
    if restore_directly:
        observation, replay = restore_context(runtime, env, captured)
    else:
        observation, replay = replay_context(
            runtime,
            env,
            initial_state,
            captured,
            wait_steps=wait_steps,
            trial_seed=int(context["trial_seed"]),
        )
    rows = [dict(row) for row in captured.prefix_rows]
    hidden_states = list(captured.prefix_hidden_states)
    trajectory = TrajectoryRecorder(runtime)
    policy_step = int(context["policy_step"])
    trajectory.append(
        policy_step=policy_step,
        stage="before_action",
        observation=observation,
        simulator_state=env.get_sim_state(),
    )
    command = runtime.np.asarray(target_decision.command).copy()
    trajectory.append_decision(
        policy_step=policy_step,
        decision=target_decision,
        executed_command=command,
    )
    row = _action_row(policy_step, command, target_decision)
    observation, reward, done, _ = env.step(command.tolist())
    row["environment/reward"] = reward
    row["environment/done"] = bool(done)
    rows.append(row)
    hidden_states.append(target_decision.hidden_states)
    success = bool(done)

    for next_step in range(policy_step + 1, MAX_STEPS[policy_config.task_suite_name]):
        if success:
            break
        trajectory.append(
            policy_step=next_step,
            stage="before_action",
            observation=observation,
            simulator_state=env.get_sim_state(),
        )
        decision = policy_decision(
            runtime,
            policy_config,
            model,
            processor,
            observation,
            task_description,
        )
        command = runtime.np.asarray(decision.command).copy()
        trajectory.append_decision(
            policy_step=next_step,
            decision=decision,
            executed_command=command,
        )
        row = _action_row(next_step, command, decision)
        observation, reward, done, _ = env.step(command.tolist())
        row["environment/reward"] = reward
        row["environment/done"] = bool(done)
        rows.append(row)
        hidden_states.append(decision.hidden_states)
        success = bool(done)
    trajectory.append(
        policy_step=(int(rows[-1]["action/timestep"]) + 1),
        stage="after_final_action",
        observation=observation,
        simulator_state=env.get_sim_state(),
    )
    return {
        "success": success,
        "rows": rows,
        "hidden_states": runtime.torch.stack(hidden_states, dim=0),
        "trajectory": trajectory,
        "replay": replay,
    }


def write_terminal_branch(
    output_dir: Path,
    runtime: Runtime,
    context: dict[str, Any],
    task_description: str,
    branch: dict[str, Any],
    fault: dict[str, Any],
    execution: dict[str, Any],
    *,
    condition: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    trial = Trial(int(context["task_id"]), int(context["episode_index"]))
    stem = safe_stem(trial, bool(branch["success"]))
    csv_path = output_dir / f"{stem}.csv"
    pickle_path = output_dir / f"{stem}.pkl"
    trajectory_path = output_dir / f"{stem}.trajectory.npz"
    write_csv_atomic(csv_path, branch["rows"])
    write_pickle_atomic(
        pickle_path,
        {
            "hidden_states": branch["hidden_states"],
            "condition": condition,
            "fault": fault,
            "task_suite_name": "libero_10",
            "task_id": trial.task_id,
            "task_description": task_description,
            "episode_idx": trial.episode_index,
            "episode_success": bool(branch["success"]),
            "trial_seed": int(context["trial_seed"]),
            "mp4_path": None,
        },
    )
    trajectory_archive = None
    trajectory_error = None
    try:
        trajectory_archive = branch["trajectory"].write(trajectory_path)
    except Exception as error:
        trajectory_error = f"{type(error).__name__}: {error}"
        write_json_atomic(
            output_dir / f"{stem}.trajectory.error.json",
            {
                "schema_version": 1,
                "status": "error",
                "error": trajectory_error,
            },
        )
    write_json_atomic(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "condition": condition,
            "context": context,
            "fault": fault,
            "execution": execution,
        },
    )
    result = {
        "schema_version": 2,
        "status": "complete",
        "condition": condition,
        "task_suite_name": "libero_10",
        "task_id": trial.task_id,
        "task_description": task_description,
        "episode_index": trial.episode_index,
        "trial_seed": int(context["trial_seed"]),
        "initial_state_sha256": context["initial_state_sha256"],
        "success": bool(branch["success"]),
        "policy_steps": len(branch["rows"]),
        "maximum_policy_steps": MAX_STEPS["libero_10"],
        "rollout_seconds": elapsed_seconds,
        "fault": fault,
        "context_replay": branch["replay"],
        "execution": execution,
        "trajectory_archive": trajectory_archive,
        "trajectory_archive_error": trajectory_error,
        "files": {
            "csv": csv_path.name,
            "pickle": pickle_path.name,
            "video": None,
            "trajectory": trajectory_path.name if trajectory_archive else None,
        },
        "artifact_manifest": [
            artifact_record(csv_path),
            artifact_record(pickle_path),
            *([artifact_record(trajectory_path)] if trajectory_archive else []),
        ],
    }
    write_json_atomic(
        output_dir / f"task{trial.task_id}--ep{trial.episode_index}.complete.json",
        result,
    )
    return result
