from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


OUTCOME = "operational_silent_failure"
MODEL_SPECS = {
    "depth_only": (("normalized_depth", "identity"),),
    "output_effect": (("command_normalized_l2", "log1p"),),
    "compositional": (
        ("injection_normalized_l2", "log1p"),
        ("final_propagation_normalized_l2", "log1p"),
        ("safe_feature_normalized_l2", "log1p"),
        ("command_normalized_l2", "log1p"),
        ("score_change_from_control_at_fault", "signed_log1p"),
    ),
}


def eligible_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in analysis["records"] if row.get("eligible_causal_outcome")
    ]


def transform(value: float, kind: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("risk feature is not finite")
    if kind == "identity":
        return number
    if kind == "log1p":
        if number < 0:
            raise ValueError("log1p risk feature cannot be negative")
        return math.log1p(number)
    if kind == "signed_log1p":
        return math.copysign(math.log1p(abs(number)), number)
    raise ValueError(f"unknown risk feature transform: {kind}")


def feature_row(
    row: dict[str, Any], specification: list[dict[str, str]] | tuple[tuple[str, str], ...]
) -> list[float]:
    result = []
    for item in specification:
        if isinstance(item, dict):
            name, kind = item["name"], item["transform"]
        else:
            name, kind = item
        value = row.get(name)
        if value is None:
            raise ValueError(f"eligible record has no {name}: {row.get('record_id')}")
        result.append(transform(float(value), kind))
    return result


def predict_probability(model: dict[str, Any], row: dict[str, Any]) -> float:
    values = feature_row(row, model["features"])
    means = model["standardization"]["mean"]
    scales = model["standardization"]["scale"]
    coefficients = model["logistic_regression"]["coefficients"]
    if not (len(values) == len(means) == len(scales) == len(coefficients)):
        raise ValueError("risk model arrays have inconsistent lengths")
    logit = float(model["logistic_regression"]["intercept"])
    for value, mean, scale, coefficient in zip(
        values, means, scales, coefficients, strict=True
    ):
        if float(scale) <= 0:
            raise ValueError("risk model feature scale must be positive")
        logit += float(coefficient) * (value - float(mean)) / float(scale)
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1 + exponential)


def top_risk_group(
    probabilities: list[float], fraction: float = 0.2
) -> tuple[list[int], float]:
    if not probabilities:
        raise ValueError("cannot rank an empty risk population")
    if not 0 < fraction <= 1:
        raise ValueError("risk fraction must be in (0, 1]")
    target = max(1, math.ceil(len(probabilities) * fraction))
    threshold = sorted(probabilities, reverse=True)[target - 1]
    return [
        index for index, probability in enumerate(probabilities) if probability >= threshold
    ], threshold


def trajectory_groups(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = f"task{int(row['task_id'])}:episode{int(row['episode_index'])}"
        groups[key].append(index)
    return dict(sorted(groups.items()))


def outcome_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    positive = [row for row in rows if row[OUTCOME]]
    return {
        "interventions": len(rows),
        "residual_interventions": len(positive),
        "trajectory_clusters": len(trajectory_groups(rows)),
        "residual_trajectory_clusters": len(
            {(row["task_id"], row["episode_index"]) for row in positive}
        ),
        "distinct_physical_runs": len({row["physical_run"] for row in rows}),
        "residual_physical_runs": len({row["physical_run"] for row in positive}),
    }
