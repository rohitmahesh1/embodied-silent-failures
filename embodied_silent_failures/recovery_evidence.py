from __future__ import annotations

import math
from typing import Any


HORIZONS = (1, 5, 10, 25)
POLICY_CHECKPOINTS = (0, 1, 5, 10, 25)
PHYSICAL_CHECKPOINTS = (1, 5, 10, 25)
MODEL_FAMILIES = (
    "context",
    "policy",
    "policy_physical_current",
    "policy_physical_path",
    "policy_safe",
    "all",
)


def _distance(comparison: dict[str, Any], name: str) -> float:
    value = comparison[name]
    if isinstance(value, dict):
        return float(value["symmetric_normalized_difference_l2"])
    return float(value)


def _at_or_before(checkpoints: tuple[int, ...], horizon: int) -> tuple[int, ...]:
    return tuple(value for value in checkpoints if value <= horizon)


def _context_features(row: dict[str, Any]) -> dict[str, float]:
    return {
        f"task={row['task_id']}": 1.0,
        "phase_fraction": float(row["phase_fraction"]),
    }


def _policy_features(row: dict[str, Any], horizon: int) -> dict[str, float]:
    result = {}
    comparisons = row["comparisons"]
    for checkpoint in _at_or_before(POLICY_CHECKPOINTS, horizon):
        comparison = comparisons.get(str(checkpoint))
        if comparison is None:
            result[f"policy:h{checkpoint}:missing"] = 1.0
            continue
        measurements = (
            ("command_distance", "executed_command", True),
            ("changed_token_fraction", "changed_action_token_fraction", False),
            (
                "entropy_difference",
                "mean_absolute_action_entropy_difference",
                True,
            ),
        )
        for label, name, log_transform in measurements:
            key = f"policy:h{checkpoint}:{label}"
            if name not in comparison:
                result[f"{key}:missing"] = 1.0
                continue
            value = _distance(comparison, name)
            result[key] = math.log1p(value) if log_transform else value
    return result


def _physical_at(
    row: dict[str, Any], checkpoint: int, *, prefix: str
) -> dict[str, float]:
    comparison = row["comparisons"].get(str(checkpoint))
    if comparison is None:
        return {f"{prefix}:h{checkpoint}:missing": 1.0}
    return {
        f"{prefix}:h{checkpoint}:{name}": math.log1p(_distance(comparison, name))
        for name in ("object-state", "robot0_proprio-state", "simulator_state")
    }


def _physical_current_features(
    row: dict[str, Any], horizon: int
) -> dict[str, float]:
    return _physical_at(row, horizon, prefix="physical_current")


def _physical_path_features(
    row: dict[str, Any], horizon: int
) -> dict[str, float]:
    result = {}
    for checkpoint in _at_or_before(PHYSICAL_CHECKPOINTS, horizon):
        result.update(_physical_at(row, checkpoint, prefix="physical_path"))
    return result


def monitor_window_features(
    arrays: dict[str, Any], index: int, horizon: int
) -> dict[str, float]:
    import numpy as np

    response = np.asarray(arrays["monitor_increment_delta"][index, :horizon], dtype=float)
    absolute_response = np.asarray(
        arrays["absolute_monitor_increment_delta"][index, :horizon], dtype=float
    )
    displacement = np.asarray(
        arrays["selected_feature_normalized_l2"][index, :horizon], dtype=float
    )
    signed_sum = float(response.sum())
    absolute_sum = float(absolute_response.sum())
    return {
        "safe:feature_displacement_rms": math.log1p(
            float(math.sqrt(float(np.mean(displacement * displacement))))
        ),
        "safe:feature_displacement_last": math.log1p(float(displacement[-1])),
        # SAFE-MLP at commit b6036ab cumulatively sums one scalar contribution
        # per policy step. These terms expose its net response and what that sum
        # removes, without changing or retraining the published monitor.
        "safe:response_signed_sum": signed_sum,
        "safe:response_absolute_sum": math.log1p(absolute_sum),
        "safe:response_cancellation_fraction": (
            1.0 - abs(signed_sum) / absolute_sum if absolute_sum else 0.0
        ),
    }


def feature_dict(
    row: dict[str, Any],
    *,
    horizon: int,
    family: str,
) -> dict[str, float]:
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family: {family}")

    result = _context_features(row)
    if family == "context":
        return result

    result.update(_policy_features(row, horizon))
    if family == "policy_physical_current":
        result.update(_physical_current_features(row, horizon))
    if family in {"policy_physical_path", "all"}:
        result.update(_physical_path_features(row, horizon))
    if family in {"policy_safe", "all"}:
        result.update(row["monitor_features"][str(horizon)])
    return result


def direct_measurements(row: dict[str, Any], horizon: int) -> dict[str, float]:
    comparison = row["comparisons"][str(horizon)]
    result = {
        "policy_command_distance": _distance(comparison, "executed_command"),
        "policy_changed_token_fraction": _distance(
            comparison, "changed_action_token_fraction"
        ),
    }
    for name in ("object-state", "robot0_proprio-state", "simulator_state"):
        label = name.replace("0_", "_").replace("-state", "")
        result[f"physical_{label}_distance"] = _distance(comparison, name)

    previous = max(
        (value for value in PHYSICAL_CHECKPOINTS if value < horizon),
        default=None,
    )
    if previous is not None:
        previous_comparison = row["comparisons"][str(previous)]
        for name in ("object-state", "robot0_proprio-state", "simulator_state"):
            label = name.replace("0_", "_").replace("-state", "")
            result[f"physical_{label}_log_growth"] = (
                math.log1p(_distance(comparison, name))
                - math.log1p(_distance(previous_comparison, name))
            )

    monitor = row["monitor_features"][str(horizon)]
    result.update(
        {
            "safe_feature_displacement_rms": monitor[
                "safe:feature_displacement_rms"
            ],
            "safe_response_signed_sum": monitor["safe:response_signed_sum"],
            "safe_response_absolute_sum": monitor["safe:response_absolute_sum"],
            "safe_response_cancellation_fraction": monitor[
                "safe:response_cancellation_fraction"
            ],
        }
    )
    return result
