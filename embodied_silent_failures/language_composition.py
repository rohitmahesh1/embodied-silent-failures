from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from embodied_silent_failures.language_gates import (
    COMMAND_COMPONENTS,
    physical_command_branches,
)
from embodied_silent_failures.language_risk import (
    feature_row,
    predict_probability,
    top_risk_group,
)


# These descriptions are fixed before inspecting model results. They form a ladder:
# command magnitude, signed command geometry, then the coarse context retained by the
# campaign. "Coarse context" is not a claim that task/phase/clean command encode the
# full MuJoCo state.
CONSEQUENCE_BASE_SPECS = {
    "magnitude": (("command_l2", "log1p"),),
    "signed_command": tuple(
        (f"delta_{name}", "identity") for name in COMMAND_COMPONENTS
    ),
}
MONITOR_SPECS = {
    "fault_margin": (("fault_margin_ratio", "identity"),),
    "margin_and_response": (
        ("control_margin_ratio", "identity"),
        ("score_shift_ratio", "identity"),
    ),
}


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("cannot average absent or non-finite measurements")
    return sum(values) / len(values)


def physical_consequence_rows(
    analysis_rows: list[dict[str, Any]],
    score_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse exact-command aliases for the task-consequence gate."""
    branches = physical_command_branches(analysis_rows, score_index)
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        if not row.get("eligible_causal_outcome") or not row.get("command_changed"):
            continue
        members[str(row["physical_run"])].append(score_index[str(row["record_id"])])

    result = []
    for branch in branches:
        scores = members[str(branch["physical_run"])]
        thresholds = [float(score["threshold_at_fault"]) for score in scores]
        score_values = [float(score["score_at_fault"]) for score in scores]
        control_values = [float(score["control_score_at_fault"]) for score in scores]
        phase_values = [float(score["context"]["phase_fraction"]) for score in scores]
        row = dict(branch)
        row.update(
            {
                "phase_fraction": _mean(phase_values),
                "member_score_spread": max(score_values) - min(score_values),
                "member_threshold_spread": max(thresholds) - min(thresholds),
                "member_control_score_spread": max(control_values)
                - min(control_values),
            }
        )
        result.append(row)
    return result


def intervention_composition_rows(
    analysis_rows: list[dict[str, Any]],
    score_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retain one row per changed layer intervention and its own SAFE evidence."""
    result = []
    for analysis in analysis_rows:
        if not analysis.get("eligible_causal_outcome") or not analysis.get(
            "command_changed"
        ):
            continue
        score = score_index[str(analysis["record_id"])]
        local = score["local_measurements"]
        clean = tuple(float(value) for value in local["clean_executed_command"])
        faulted = tuple(float(value) for value in local["faulted_executed_command"])
        if len(clean) != len(COMMAND_COMPONENTS) or len(faulted) != len(
            COMMAND_COMPONENTS
        ):
            raise ValueError("changed intervention has a non-seven-dimensional command")
        threshold = float(score["threshold_at_fault"])
        fault_score = float(score["score_at_fault"])
        control_score = float(score["control_score_at_fault"])
        if threshold <= 0 or not all(
            math.isfinite(value) for value in (threshold, fault_score, control_score)
        ):
            raise ValueError("SAFE injection-time measurements must be finite")
        row = dict(analysis)
        row["phase_fraction"] = float(score["context"]["phase_fraction"])
        row["threshold_at_fault"] = threshold
        row["fault_score_at_fault"] = fault_score
        row["control_score_at_fault"] = control_score
        row["fault_margin_ratio"] = (threshold - fault_score) / threshold
        row["control_margin_ratio"] = (threshold - control_score) / threshold
        row["score_shift_ratio"] = (fault_score - control_score) / threshold
        row["monitor_missed"] = bool(analysis["operational_silent_failure"])
        for name, clean_value, faulted_value in zip(
            COMMAND_COMPONENTS, clean, faulted, strict=True
        ):
            row[f"clean_{name}"] = clean_value
            row[f"faulted_{name}"] = faulted_value
            row[f"delta_{name}"] = faulted_value - clean_value
        result.append(row)
    return result


def add_context_indicators(
    rows: list[dict[str, Any]], task_ids: tuple[int, ...]
) -> None:
    for row in rows:
        for task_id in task_ids:
            row[f"task_{task_id}"] = float(int(row["task_id"]) == task_id)
        for phase in ("early", "middle", "late"):
            row[f"phase_{phase}"] = float(str(row["phase"]) == phase)


def model_specifications(task_ids: tuple[int, ...]) -> dict[str, Any]:
    signed = CONSEQUENCE_BASE_SPECS["signed_command"]
    context = (
        signed
        + tuple((f"clean_{name}", "identity") for name in COMMAND_COMPONENTS)
        + tuple((f"task_{task_id}", "identity") for task_id in task_ids)
        + tuple((f"phase_{phase}", "identity") for phase in ("early", "middle", "late"))
    )
    return {
        "consequence": {
            **CONSEQUENCE_BASE_SPECS,
            "coarse_context": context,
        },
        "monitor": MONITOR_SPECS,
        "direct": {
            "command_magnitude": CONSEQUENCE_BASE_SPECS["magnitude"],
            "compact": (
                ("command_l2", "log1p"),
                ("phase_fraction", "identity"),
                ("control_margin_ratio", "identity"),
                ("score_shift_ratio", "identity"),
            ),
            "same_inputs": context + MONITOR_SPECS["margin_and_response"],
        },
        "compositions": {
            "magnitude_margin": ("magnitude", "fault_margin"),
            "signed_response": ("signed_command", "margin_and_response"),
            "coarse_context_response": ("coarse_context", "margin_and_response"),
        },
    }


def fit_logistic_model(
    rows: list[dict[str, Any]],
    outcome: str,
    specification: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([int(row[outcome]) for row in rows], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{outcome} requires both classes")
    matrix = np.asarray(
        [feature_row(row, specification) for row in rows], dtype=np.float64
    )
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    estimator = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=2_000,
        penalty="l2",
        solver="lbfgs",
    ).fit(standardized, labels)
    probabilities = estimator.predict_proba(standardized)[:, 1]
    return {
        "outcome": outcome,
        "features": [
            {"name": name, "transform": transform}
            for name, transform in specification
        ],
        "standardization": {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
        },
        "logistic_regression": {
            "C": 1.0,
            "class_weight": None,
            "coefficients": estimator.coef_[0].astype(float).tolist(),
            "intercept": float(estimator.intercept_[0]),
            "penalty": "l2",
            "solver": "lbfgs",
        },
        "training": {
            "rows": len(rows),
            "positive_outcomes": int(labels.sum()),
            "resubstitution_roc_auc": float(roc_auc_score(labels, probabilities)),
            "resubstitution_average_precision": float(
                average_precision_score(labels, probabilities)
            ),
            "resubstitution_brier_score": float(
                brier_score_loss(labels, probabilities)
            ),
        },
    }


def model_probabilities(
    model: dict[str, Any], rows: list[dict[str, Any]]
) -> list[float]:
    return [predict_probability(model, row) for row in rows]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def binary_metrics(
    rows: list[dict[str, Any]], outcome: str, probabilities: list[float]
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    if len(rows) != len(probabilities) or not rows:
        raise ValueError("metric rows and probabilities must have equal nonzero length")
    labels = np.asarray([int(row[outcome]) for row in rows], dtype=int)
    values = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{outcome} metrics require both classes")
    top_indices, threshold = top_risk_group(values.astype(float).tolist())
    base_rate = float(labels.mean())
    top_rate = float(labels[top_indices].mean())
    return {
        "rows": len(rows),
        "positive_outcomes": int(labels.sum()),
        "base_rate": base_rate,
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "brier_score": float(brier_score_loss(labels, values)),
        "top_fifth": {
            "target_fraction": 0.2,
            "tie_policy": "include every intervention tied at the boundary",
            "threshold": float(threshold),
            "rows": len(top_indices),
            "positive_outcomes": int(labels[top_indices].sum()),
            "rate": top_rate,
            "enrichment_over_uniform": top_rate / base_rate,
        },
    }


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    outcome: str,
    predictions: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
    baseline: str | None = None,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(int(row["task_id"]), int(row["episode_index"]))].append(index)
    clusters = list(groups.values())
    labels = np.asarray([int(row[outcome]) for row in rows], dtype=int)
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in predictions.items()
    }
    rng = random.Random(seed)
    estimates = {name: [] for name in arrays}
    differences = {
        f"{name}_minus_{baseline}": []
        for name in arrays
        if baseline is not None and name != baseline
    }
    if baseline is not None and baseline not in arrays:
        raise ValueError(f"bootstrap baseline is absent: {baseline}")
    for _ in range(samples):
        selected_clusters = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        indices = np.asarray(
            [index for cluster in selected_clusters for index in cluster], dtype=int
        )
        selected_labels = labels[indices]
        if len(np.unique(selected_labels)) != 2:
            continue
        aucs = {
            name: float(roc_auc_score(selected_labels, values[indices]))
            for name, values in arrays.items()
        }
        for name, value in aucs.items():
            estimates[name].append(value)
        baseline_auc = aucs.get(baseline) if baseline is not None else None
        if baseline_auc is not None:
            for name in arrays:
                if name != baseline:
                    differences[f"{name}_minus_{baseline}"].append(
                        aucs[name] - baseline_auc
                    )

    def interval(values: list[float]) -> dict[str, Any]:
        return {
            "valid_samples": len(values),
            "95": (
                [_percentile(values, 0.025), _percentile(values, 0.975)]
                if values
                else []
            ),
        }

    return {
        "requested_samples": samples,
        "roc_auc": {name: interval(values) for name, values in estimates.items()},
        "roc_auc_difference": {
            name: interval(values) for name, values in differences.items()
        },
    }
