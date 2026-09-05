from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_silent_failures.atlas_mechanism_extraction import vector_difference
from embodied_silent_failures.provenance import load_json


HORIZONS = (1, 5, 10, 25)
ALIGNMENT_WINDOWS = (5, 10, 25)
STATE_STREAMS = ("simulator_state", "object-state", "robot0_proprio-state")


def is_reference_control(physical: dict[str, Any]) -> bool:
    return str(physical["run"]) == str(physical["control_run"])


def one_completion(attempt_dir: Path) -> tuple[dict[str, Any], Path]:
    completions = list(attempt_dir.glob("*.complete.json"))
    if len(completions) != 1:
        raise ValueError(
            f"expected one completion in {attempt_dir}, found {len(completions)}"
        )
    result = load_json(completions[0])
    trajectory_name = result.get("files", {}).get("trajectory")
    if not trajectory_name:
        raise ValueError(f"trajectory archive is absent from {completions[0]}")
    return result, attempt_dir / str(trajectory_name)


def load_state_trajectory(
    np: Any, result: dict[str, Any], path: Path
) -> dict[str, Any]:
    observation_keys = {
        str(record["name"]): str(record["archive_key"])
        for record in result["trajectory_archive"]["observations"]
    }
    with np.load(path, allow_pickle=False) as archive:
        streams = {"simulator_state": np.asarray(archive["simulator_state"])}
        for name in STATE_STREAMS[1:]:
            if name in observation_keys:
                streams[name] = np.asarray(archive[observation_keys[name]])
        return {
            "policy_steps": np.asarray(archive["policy_step"], dtype=np.int64),
            "snapshot_stages": np.asarray(
                archive["snapshot_stage"], dtype=np.uint8
            ),
            "streams": streams,
        }


def _step_index(trajectory: dict[str, Any]) -> dict[int, int]:
    steps = trajectory["policy_steps"]
    if len(set(int(value) for value in steps)) != len(steps):
        raise ValueError("trajectory contains repeated policy-step snapshots")
    return {int(value): index for index, value in enumerate(steps)}


def align_state_stream(
    np: Any,
    faulted_state: Any,
    control_states: Any,
    control_steps: Any,
    *,
    target_step: int,
) -> dict[str, Any]:
    faulted_state = np.asarray(faulted_state, dtype=np.float64)
    control_states = np.asarray(control_states, dtype=np.float64)
    if control_states.ndim != faulted_state.ndim + 1:
        raise ValueError("control path and faulted state ranks do not agree")
    if control_states.shape[1:] != faulted_state.shape:
        raise ValueError("control path and faulted state shapes do not agree")
    if len(control_states) != len(control_steps) or not len(control_states):
        raise ValueError("control path is empty or has inconsistent steps")

    differences = control_states - faulted_state
    flat_differences = differences.reshape(len(differences), -1)
    flat_control = control_states.reshape(len(control_states), -1)
    flat_faulted = faulted_state.reshape(-1)
    difference_l2 = np.linalg.norm(flat_differences, axis=1)
    scale = 0.5 * (
        np.linalg.norm(flat_control, axis=1) + np.linalg.norm(flat_faulted)
    )
    normalized = difference_l2 / np.maximum(scale, np.finfo(np.float64).eps)
    offsets = np.asarray(control_steps, dtype=np.int64) - int(target_step)
    selected = min(
        range(len(control_steps)),
        key=lambda index: (
            float(normalized[index]),
            abs(int(offsets[index])),
            int(offsets[index]),
        ),
    )
    comparison = vector_difference(
        np, faulted_state, control_states[selected]
    )
    return {
        "nearest_control_policy_step": int(control_steps[selected]),
        "signed_step_offset": int(offsets[selected]),
        **comparison,
    }


def align_at_horizon(
    np: Any,
    faulted: dict[str, Any],
    control: dict[str, Any],
    *,
    fault_step: int,
    horizon: int,
    window: int,
) -> dict[str, Any] | None:
    target_step = fault_step + horizon
    faulted_index = _step_index(faulted)
    control_index = _step_index(control)
    if target_step not in faulted_index or target_step not in control_index:
        return None
    faulted_target = faulted_index[target_step]
    control_target = control_index[target_step]
    if (
        int(faulted["snapshot_stages"][faulted_target]) != 0
        or int(control["snapshot_stages"][control_target]) != 0
    ):
        return None

    # The recorder starts each branch at the intervention state. Restricting the
    # path to that point and later prevents an identical pre-fault history from
    # being counted as recovery. The symmetric window otherwise allows the data
    # to show whether a branch is behind or ahead of the successful control.
    candidate_steps = np.asarray(
        sorted(
            step
            for step in control_index
            if max(fault_step, target_step - window)
            <= step
            <= target_step + window
            and int(control["snapshot_stages"][control_index[step]]) == 0
        ),
        dtype=np.int64,
    )
    if not len(candidate_steps):
        return None
    candidate_indices = [control_index[int(step)] for step in candidate_steps]

    streams = {}
    for name in STATE_STREAMS:
        if name not in faulted["streams"] or name not in control["streams"]:
            continue
        faulted_value = faulted["streams"][name][faulted_target]
        control_values = control["streams"][name]
        same_clock = vector_difference(
            np,
            faulted_value,
            control_values[control_target],
        )
        nearest_path = align_state_stream(
            np,
            faulted_value,
            control_values[candidate_indices],
            candidate_steps,
            target_step=target_step,
        )
        current = float(same_clock["symmetric_normalized_difference_l2"])
        nearest = float(nearest_path["symmetric_normalized_difference_l2"])
        streams[name] = {
            "same_clock": same_clock,
            "nearest_path": nearest_path,
            "alignment_gain": current - nearest,
            "unexplained_fraction": nearest / current if current else 0.0,
        }
    return {
        "faulted_policy_step": target_step,
        "candidate_control_step_minimum": int(candidate_steps[0]),
        "candidate_control_step_maximum": int(candidate_steps[-1]),
        "candidate_control_step_count": len(candidate_steps),
        "streams": streams,
    }


def extract_pair_alignments(
    np: Any,
    physical: dict[str, Any],
    context: dict[str, Any],
    faulted: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    fault_step = int(context["policy_step"])
    alignments = {}
    for horizon in HORIZONS:
        windows = {}
        for window in ALIGNMENT_WINDOWS:
            aligned = align_at_horizon(
                np,
                faulted,
                control,
                fault_step=fault_step,
                horizon=horizon,
                window=window,
            )
            if aligned is not None:
                windows[str(window)] = aligned
        if windows:
            alignments[str(horizon)] = windows
    return {
        "run": str(physical["run"]),
        "context_id": str(context["context_id"]),
        "analysis_split": str(context["analysis_split"]),
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "phase": str(context["phase"]),
        "phase_fraction": float(context["phase_fraction"]),
        "fault_step": fault_step,
        "faulted_success": bool(physical["faulted_success"]),
        "faulted_length": int(physical["faulted_length"]),
        "control_success": bool(physical["control_success"]),
        "control_length": int(physical["control_length"]),
        "alignments": alignments,
    }
