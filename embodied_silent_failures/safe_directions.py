from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


DIRECTION_FIELDS = (
    "selected_feature_l2",
    "selected_feature_normalized_l2",
    "absolute_monitor_increment_delta",
    "monitor_secant_sensitivity",
    "absolute_clean_gradient_cosine",
    "relu_gate_flip_fraction",
    "threshold_margin_after_fault",
)


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def monitor_direction_batch(
    model: Any,
    clean_features: Any,
    faulted_features: Any,
    torch: Any,
) -> list[dict[str, Any]]:
    if clean_features.ndim != 2 or clean_features.shape != faulted_features.shape:
        raise ValueError(
            "SAFE direction batches must have equal [batch, feature] shape"
        )
    clean = clean_features.detach().float().requires_grad_(True)
    faulted = faulted_features.detach().float()
    clean_increments = model.projector(clean).reshape(-1)
    clean_gradients = torch.autograd.grad(clean_increments.sum(), clean)[0]
    with torch.no_grad():
        faulted_increments = model.projector(faulted).reshape(-1)
        clean_preactivation = model.projector[0](clean.detach())
        faulted_preactivation = model.projector[0](faulted)

    deltas = faulted - clean.detach()
    delta_norms = torch.linalg.vector_norm(deltas, dim=-1)
    clean_norms = torch.linalg.vector_norm(clean.detach(), dim=-1)
    gradient_norms = torch.linalg.vector_norm(clean_gradients, dim=-1)
    gradient_dots = (clean_gradients * deltas).sum(dim=-1)
    increment_deltas = faulted_increments - clean_increments.detach()
    gate_flips = (clean_preactivation > 0) != (faulted_preactivation > 0)

    records = []
    for index in range(len(clean)):
        delta_norm = float(delta_norms[index].item())
        clean_norm = float(clean_norms[index].item())
        gradient_norm = float(gradient_norms[index].item())
        gradient_dot = float(gradient_dots[index].item())
        increment_delta = float(increment_deltas[index].item())
        denominator = gradient_norm * delta_norm
        records.append(
            {
                "selected_feature_l2": delta_norm,
                "selected_feature_normalized_l2": _optional_ratio(
                    delta_norm, clean_norm
                ),
                "clean_monitor_increment": float(clean_increments[index].item()),
                "faulted_monitor_increment": float(faulted_increments[index].item()),
                "monitor_increment_delta": increment_delta,
                "absolute_monitor_increment_delta": abs(increment_delta),
                "monitor_secant_sensitivity": _optional_ratio(
                    abs(increment_delta), delta_norm
                ),
                "clean_gradient_l2": gradient_norm,
                "clean_gradient_dot_delta": gradient_dot,
                "clean_gradient_cosine": (
                    gradient_dot / denominator if denominator > 0 else None
                ),
                "absolute_clean_gradient_cosine": (
                    abs(gradient_dot / denominator) if denominator > 0 else None
                ),
                "clean_linearization_error": increment_delta - gradient_dot,
                "relu_gate_flip_count": int(gate_flips[index].sum().item()),
                "relu_gate_flip_fraction": float(
                    gate_flips[index].float().mean().item()
                ),
            }
        )
    return records


def _median(values: list[float]) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    middle = len(finite) // 2
    if len(finite) % 2:
        return finite[middle]
    return (finite[middle - 1] + finite[middle]) / 2


def direction_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"interventions": len(records)}
    for field in DIRECTION_FIELDS:
        values = [
            float(record[field])
            for record in records
            if record.get(field) is not None
        ]
        result[f"median_{field}"] = _median(values)
    result["alarm_at_fault_interventions"] = sum(
        bool(record.get("safe_alarm_at_fault")) for record in records
    )
    return result


def collapse_physical_failures(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("outcome_group") not in {
            "detected_failure",
            "silent_failure",
        }:
            continue
        physical_run = record.get("physical_run")
        if not physical_run:
            raise ValueError("failed intervention has no physical run")
        grouped[str(physical_run)].append(record)

    result = []
    stable_fields = (
        "outcome_group",
        "analysis_split",
        "context_id",
        "task_id",
        "episode_index",
        "phase",
        "policy_step",
        "action_token_position",
        "safe_alarm_at_fault",
        "safe_alarm_post_fault_any",
    )
    for physical_run, members in sorted(grouped.items()):
        first = members[0]
        for member in members[1:]:
            disagreements = [
                field
                for field in stable_fields
                if member.get(field) != first.get(field)
            ]
            if disagreements:
                raise ValueError(
                    f"physical failure {physical_run} disagrees on {disagreements}"
                )
        branch = {
            "physical_run": physical_run,
            "member_interventions": len(members),
            **{field: first.get(field) for field in stable_fields},
        }
        for field in DIRECTION_FIELDS:
            values = [
                float(member[field])
                for member in members
                if member.get(field) is not None
            ]
            branch[field] = _median(values)
        result.append(branch)
    return result
