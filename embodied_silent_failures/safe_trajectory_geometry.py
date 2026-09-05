from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from embodied_silent_failures.safe_directions import monitor_direction_batch


WINDOW_STEPS = 25


def physical_population(
    site_documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse eligible graph sites to distinct non-control physical branches."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monitor = None
    for document in site_documents:
        current_monitor = document["monitor"]
        monitor = current_monitor if monitor is None else monitor
        if current_monitor != monitor:
            raise ValueError("site documents used different SAFE monitors")
        split = str(document["analysis_split"])
        for record in document["records"]:
            if not record.get("primary_eligible"):
                continue
            context_id = str(record["context_id"])
            run = str(record["physical_run"])
            if run == f"{context_id}-control":
                continue
            grouped[run].append({**record, "analysis_split": split})

    primary_alpha = format(float(monitor["primary_alpha"]), "g")
    population = []
    stable_fields = ("context_id", "analysis_split", "policy_failure")
    for run, members in sorted(grouped.items()):
        representative = next(
            (
                record
                for record in members
                if record["site_id"] == record["representative_site_id"]
            ),
            members[0],
        )
        for member in members[1:]:
            disagreements = [
                field
                for field in stable_fields
                if member[field] != representative[field]
            ]
            if disagreements:
                raise ValueError(f"physical run {run} disagrees on {disagreements}")

        context = representative["context"]
        failed = bool(representative["policy_failure"])
        alarm_record = representative["safe_faulted_evidence"]["alarms"][
            primary_alpha
        ]
        alarm = bool(alarm_record["post_fault_any"]["triggered"])
        population.append(
            {
                "physical_run": run,
                "context_id": str(representative["context_id"]),
                "analysis_split": str(representative["analysis_split"]),
                "task_id": int(context["task_id"]),
                "episode_index": int(context["episode_index"]),
                "phase": str(context["phase"]),
                "fault_step": int(context["policy_step"]),
                "policy_failure": failed,
                "safe_alarm_post_fault_any": alarm,
                "safe_alarm_within_25_steps": bool(
                    alarm_record["within_25_steps"]["triggered"]
                ),
                "safe_first_alarm_step": alarm_record["post_fault_any"][
                    "first_step"
                ],
                "outcome_group": (
                    "detected_failure"
                    if failed and alarm
                    else "silent_failure"
                    if failed
                    else "successful_continuation"
                ),
                "member_site_count": len(members),
                "representative_site_id": str(representative["site_id"]),
            }
        )
    return population, monitor


def trajectory_window_geometry(
    model: Any,
    control_features: Any,
    faulted_features: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure a paired feature trajectory in SAFE's learned local geometry."""
    if control_features.ndim != 2 or control_features.shape != faulted_features.shape:
        raise ValueError("paired SAFE trajectories must have equal [step, feature] shape")
    if not len(control_features):
        raise ValueError("paired SAFE trajectory is empty")

    measurements = monitor_direction_batch(
        model, control_features, faulted_features, torch
    )
    array_fields = (
        "selected_feature_l2",
        "selected_feature_normalized_l2",
        "clean_monitor_increment",
        "faulted_monitor_increment",
        "monitor_increment_delta",
        "absolute_monitor_increment_delta",
        "clean_gradient_l2",
        "clean_gradient_dot_delta",
        "clean_gradient_cosine",
        "clean_linearization_error",
        "relu_gate_flip_fraction",
    )
    arrays = {
        field: torch.tensor(
            [
                float(record[field])
                if record[field] is not None
                else float("nan")
                for record in measurements
            ],
            dtype=torch.float32,
        )
        .cpu()
        .numpy()
        for field in array_fields
    }

    return summarize_trajectory_arrays(arrays), arrays


def summarize_trajectory_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-step geometry without discarding the signed response."""
    import numpy as np

    displacement = arrays["selected_feature_l2"]
    normalized_displacement = arrays["selected_feature_normalized_l2"]
    response = arrays["monitor_increment_delta"]
    gradient_dot = arrays["clean_gradient_dot_delta"]
    gradient_norm = arrays["clean_gradient_l2"]
    cosine = arrays["clean_gradient_cosine"]
    gate_flips = arrays["relu_gate_flip_fraction"]
    linearization_error = arrays["clean_linearization_error"]
    absolute_response = arrays["absolute_monitor_increment_delta"]

    response_absolute_sum = float(absolute_response.sum())
    response_signed_sum = float(response.sum())
    possible_projection = float((gradient_norm * displacement).sum())
    projection_absolute_sum = float(abs(gradient_dot).sum())
    linearization_absolute_sum = float(abs(linearization_error).sum())
    return {
        "window_steps": len(response),
        "feature_displacement_l2_energy": float(
            math.sqrt(float((displacement * displacement).sum()))
        ),
        "normalized_feature_displacement_l2_energy": float(
            math.sqrt(float((normalized_displacement * normalized_displacement).sum()))
        ),
        "median_feature_displacement_l2": float(np.median(displacement)),
        "maximum_feature_displacement_l2": float(displacement.max()),
        "safe_response_signed_sum": response_signed_sum,
        "safe_response_absolute_sum": response_absolute_sum,
        "safe_response_cancellation_fraction": (
            1.0 - abs(response_signed_sum) / response_absolute_sum
            if response_absolute_sum > 0
            else 0.0
        ),
        "gradient_projection_signed_sum": float(gradient_dot.sum()),
        "gradient_projection_absolute_sum": projection_absolute_sum,
        "gradient_alignment_fraction": (
            projection_absolute_sum / possible_projection
            if possible_projection > 0
            else None
        ),
        "mean_absolute_gradient_cosine": float(abs(cosine).mean()),
        "linearization_error_absolute_sum": linearization_absolute_sum,
        "linearization_error_fraction": (
            linearization_absolute_sum / response_absolute_sum
            if response_absolute_sum > 0
            else None
        ),
        "mean_relu_gate_flip_fraction": float(gate_flips.mean()),
        "maximum_relu_gate_flip_fraction": float(gate_flips.max()),
    }
