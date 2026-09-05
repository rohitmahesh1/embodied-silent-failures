from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import load_json


COMPARISON_HORIZONS = (0, 1, 5, 10, 25, 50, 100)
STATE_OBSERVATIONS = ("object-state", "robot0_proprio-state")


def vector_difference(np: Any, left: Any, right: Any) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"cannot compare shapes {left.shape} and {right.shape}")
    difference = left - right
    difference_l2 = float(np.linalg.norm(difference))
    scale = 0.5 * (float(np.linalg.norm(left)) + float(np.linalg.norm(right)))
    normalized = difference_l2 / max(scale, np.finfo(np.float64).eps)
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "difference_l2": difference_l2,
        "symmetric_normalized_difference_l2": normalized,
        "maximum_absolute_difference": (
            float(np.max(np.abs(difference))) if difference.size else 0.0
        ),
    }


def _one_completion(attempt_dir: Path) -> tuple[dict[str, Any], Path]:
    completions = list(attempt_dir.glob("*.complete.json"))
    if len(completions) != 1:
        raise ValueError(
            f"expected one completion in {attempt_dir}, found {len(completions)}"
        )
    result = load_json(completions[0])
    trajectory_name = result["files"].get("trajectory")
    if not trajectory_name:
        raise ValueError(f"trajectory archive is absent from {completions[0]}")
    return result, attempt_dir / trajectory_name


def _observation_keys(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(record["name"]): str(record["archive_key"])
        for record in result["trajectory_archive"]["observations"]
    }


def _index(values: Any) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(values.tolist())}


def _comparison_at_horizon(
    np: Any,
    faulted: Any,
    control: Any,
    faulted_result: dict[str, Any],
    control_result: dict[str, Any],
    *,
    policy_step: int,
    horizon: int,
) -> dict[str, Any] | None:
    step = policy_step + horizon
    faulted_snapshots = _index(faulted["policy_step"])
    control_snapshots = _index(control["policy_step"])
    if step not in faulted_snapshots or step not in control_snapshots:
        return None
    faulted_index = faulted_snapshots[step]
    control_index = control_snapshots[step]
    result = {
        "policy_step": step,
        "simulator_state": vector_difference(
            np,
            faulted["simulator_state"][faulted_index],
            control["simulator_state"][control_index],
        ),
    }
    faulted_observations = _observation_keys(faulted_result)
    control_observations = _observation_keys(control_result)
    for name in STATE_OBSERVATIONS:
        if name not in faulted_observations or name not in control_observations:
            continue
        result[name] = vector_difference(
            np,
            faulted[faulted_observations[name]][faulted_index],
            control[control_observations[name]][control_index],
        )

    faulted_decisions = _index(faulted["decision_policy_step"])
    control_decisions = _index(control["decision_policy_step"])
    if step in faulted_decisions and step in control_decisions:
        faulted_decision = faulted_decisions[step]
        control_decision = control_decisions[step]
        result["executed_command"] = vector_difference(
            np,
            faulted["executed_command"][faulted_decision],
            control["executed_command"][control_decision],
        )
        faulted_tokens = faulted["action_tokens"][faulted_decision]
        control_tokens = control["action_tokens"][control_decision]
        result["changed_action_token_fraction"] = float(
            np.mean(faulted_tokens != control_tokens)
        )
        entropy_difference = (
            faulted["action_entropy"][faulted_decision]
            - control["action_entropy"][control_decision]
        )
        result["mean_absolute_action_entropy_difference"] = float(
            np.mean(np.abs(entropy_difference))
        )
    return result


def extract_context_state(
    campaign_dir: Path,
    context: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    context_dir = campaign_dir / "contexts" / str(context["context_id"])
    local = load_json(context_dir / "local.json")
    archive = np.load(context_dir / "captured_context.npz", allow_pickle=False)
    observation_keys = {
        str(record["name"]): str(record["archive_key"])
        for record in local["captured_context_archive"]["observations"]
    }
    values = {
        "simulator_state": archive["simulator_state"].astype(float).tolist()
    }
    for name in STATE_OBSERVATIONS:
        if name in observation_keys:
            values[name] = archive[observation_keys[name]].astype(float).tolist()
    if any(
        not math.isfinite(float(item))
        for vector in values.values()
        for item in vector
    ):
        raise ValueError(f"context {context['context_id']} has non-finite state")
    return {
        "context_id": str(context["context_id"]),
        "analysis_split": str(context["analysis_split"]),
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "phase": str(context["phase"]),
        "phase_fraction": float(context["phase_fraction"]),
        "policy_step": int(context["policy_step"]),
        "state": values,
    }


def extract_physical_pair(
    campaign_dir: Path,
    physical: dict[str, Any],
    context: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    run = str(physical["run"])
    control_run = f"{context['context_id']}-control"
    faulted_result, faulted_path = _one_completion(
        campaign_dir / "attempts" / run
    )
    control_result, control_path = _one_completion(
        campaign_dir / "attempts" / control_run
    )
    with np.load(faulted_path, allow_pickle=False) as faulted, np.load(
        control_path, allow_pickle=False
    ) as control:
        comparisons = {
            str(horizon): value
            for horizon in COMPARISON_HORIZONS
            if (
                value := _comparison_at_horizon(
                    np,
                    faulted,
                    control,
                    faulted_result,
                    control_result,
                    policy_step=int(context["policy_step"]),
                    horizon=horizon,
                )
            )
            is not None
        }
    return {
        "run": run,
        "control_run": control_run,
        "context_id": str(context["context_id"]),
        "analysis_split": str(context["analysis_split"]),
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "phase": str(context["phase"]),
        "policy_step": int(context["policy_step"]),
        "faulted_success": bool(physical["success"]),
        "faulted_length": int(physical["length"]),
        "control_success": bool(control_result["success"]),
        "control_length": int(control_result["policy_steps"]),
        "comparisons": comparisons,
    }
