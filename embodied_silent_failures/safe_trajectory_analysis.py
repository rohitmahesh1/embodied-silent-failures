from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable


METRICS = (
    "feature_displacement_l2_energy",
    "normalized_feature_displacement_l2_energy",
    "safe_response_signed_sum",
    "safe_response_absolute_sum",
    "safe_response_per_feature_displacement",
    "safe_response_cancellation_fraction",
    "gradient_projection_absolute_sum",
    "gradient_alignment_fraction",
    "mean_absolute_gradient_cosine",
    "linearization_error_absolute_sum",
    "linearization_error_fraction",
    "mean_relu_gate_flip_fraction",
)

TASK_BREAKDOWN_METRICS = (
    "feature_displacement_l2_energy",
    "safe_response_signed_sum",
    "safe_response_absolute_sum",
    "safe_response_cancellation_fraction",
    "gradient_alignment_fraction",
    "mean_relu_gate_flip_fraction",
)

TEMPORAL_FIELDS = (
    "feature_displacement_at_fault",
    "safe_response_at_fault",
    "absolute_safe_response_at_fault",
    "later_safe_response_signed_sum",
    "later_safe_response_absolute_sum",
    "later_gradient_projection_absolute_sum",
)


def derived_geometry(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    displacement = float(record["feature_displacement_l2_energy"])
    response = float(record["safe_response_absolute_sum"])
    result["safe_response_per_feature_displacement"] = (
        response / displacement if displacement > 0 else None
    )
    result["safe_response_net_fraction"] = (
        abs(float(record["safe_response_signed_sum"])) / response
        if response > 0
        else None
    )
    return result


def attach_temporal_geometry(
    record: dict[str, Any], arrays: dict[str, Any], index: int
) -> dict[str, Any]:
    response = arrays["monitor_increment_delta"][index]
    displacement = arrays["selected_feature_l2"][index]
    projection = arrays["clean_gradient_dot_delta"][index]
    return {
        **record,
        "feature_displacement_at_fault": float(displacement[0]),
        "safe_response_at_fault": float(response[0]),
        "absolute_safe_response_at_fault": float(abs(response[0])),
        "later_safe_response_signed_sum": float(response[1:].sum()),
        "later_safe_response_absolute_sum": float(abs(response[1:]).sum()),
        "later_gradient_projection_absolute_sum": float(abs(projection[1:]).sum()),
    }


def distribution(values: list[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray([value for value in values if math.isfinite(value)])
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array),
        "minimum": float(array.min()),
        "quantiles": {
            "0.25": float(np.quantile(array, 0.25)),
            "0.50": float(np.quantile(array, 0.50)),
            "0.75": float(np.quantile(array, 0.75)),
        },
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _selected_values(
    rows: list[dict[str, Any]],
    metric: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if predicate(row)
        and row.get(metric) is not None
        and math.isfinite(float(row[metric]))
    ]


def binary_metric_summary(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    positive: Callable[[dict[str, Any]], bool],
    negative: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    positive_values = _selected_values(rows, metric, positive)
    negative_values = _selected_values(rows, metric, negative)
    if not positive_values or not negative_values:
        return {
            "positive": distribution(positive_values),
            "negative": distribution(negative_values),
            "roc_auc_for_larger_value": None,
            "median_difference_positive_minus_negative": None,
        }
    labels = np.asarray([1] * len(positive_values) + [0] * len(negative_values))
    values = np.asarray(positive_values + negative_values)
    return {
        "positive": distribution(positive_values),
        "negative": distribution(negative_values),
        "roc_auc_for_larger_value": float(roc_auc_score(labels, values)),
        "median_difference_positive_minus_negative": float(
            np.median(positive_values) - np.median(negative_values)
        ),
    }


def _trajectory_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["episode_index"]))].append(row)
    return list(grouped.values())


def trajectory_bootstrap_auc(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    positive: Callable[[dict[str, Any]], bool],
    negative: Callable[[dict[str, Any]], bool],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    eligible = [
        row
        for row in rows
        if (positive(row) or negative(row))
        and row.get(metric) is not None
        and math.isfinite(float(row[metric]))
    ]
    groups = _trajectory_groups(eligible)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled = [
            member
            for _group in groups
            for member in groups[rng.randrange(len(groups))]
        ]
        labels = np.asarray([int(positive(row)) for row in sampled])
        if len(np.unique(labels)) != 2:
            continue
        estimates.append(
            float(
                roc_auc_score(
                    labels, [float(row[metric]) for row in sampled]
                )
            )
        )
    return {
        "resampling_unit": "task and clean-rollout episode trajectory",
        "requested_resamples": samples,
        "successful_resamples": len(estimates),
        "roc_auc_interval_95": (
            [
                float(np.quantile(estimates, 0.025)),
                float(np.quantile(estimates, 0.975)),
            ]
            if estimates
            else None
        ),
    }


def outcome_comparisons(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    failed = lambda row: bool(row["policy_failure"])
    succeeded = lambda row: not bool(row["policy_failure"])
    silent = lambda row: row["outcome_group"] == "silent_failure"
    detected = lambda row: row["outcome_group"] == "detected_failure"
    result = {}
    for offset, metric in enumerate(METRICS):
        result[metric] = {
            "failure_vs_success": binary_metric_summary(
                rows,
                metric=metric,
                positive=failed,
                negative=succeeded,
            ),
            "silent_vs_detected_failure": binary_metric_summary(
                rows,
                metric=metric,
                positive=silent,
                negative=detected,
            ),
            "failure_vs_success_trajectory_bootstrap": trajectory_bootstrap_auc(
                rows,
                metric=metric,
                positive=failed,
                negative=succeeded,
                samples=bootstrap_samples,
                seed=seed + offset,
            ),
        }
    return result


def task_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = lambda row: bool(row["policy_failure"])
    succeeded = lambda row: not bool(row["policy_failure"])
    output = {}
    for task in sorted({int(row["task_id"]) for row in rows}):
        selected = [row for row in rows if int(row["task_id"]) == task]
        output[str(task)] = {
            "physical_continuations": len(selected),
            "policy_failures": sum(row["policy_failure"] for row in selected),
            "failure_vs_success": {
                metric: binary_metric_summary(
                    selected,
                    metric=metric,
                    positive=failed,
                    negative=succeeded,
                )
                for metric in TASK_BREAKDOWN_METRICS
            },
        }
    return output


def rank_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    pairs = (
        (
            "feature_displacement_to_absolute_safe_response",
            "feature_displacement_l2_energy",
            "safe_response_absolute_sum",
        ),
        (
            "gradient_projection_to_absolute_safe_response",
            "gradient_projection_absolute_sum",
            "safe_response_absolute_sum",
        ),
        (
            "relu_gate_changes_to_linearization_error",
            "mean_relu_gate_flip_fraction",
            "linearization_error_absolute_sum",
        ),
    )
    output = {}
    for name, left, right in pairs:
        selected = [
            row
            for row in rows
            if row.get(left) is not None
            and row.get(right) is not None
            and math.isfinite(float(row[left]))
            and math.isfinite(float(row[right]))
        ]
        statistic = spearmanr(
            [float(row[left]) for row in selected],
            [float(row[right]) for row in selected],
        )
        output[name] = {
            "pairs": len(selected),
            "spearman_rho": float(statistic.statistic),
            "p_value_unclustered_descriptive_only": float(statistic.pvalue),
        }
    return output


def quiet_failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    failures = [row for row in rows if row["policy_failure"]]
    if not failures:
        return {"failures": 0}
    net = np.asarray([abs(float(row["safe_response_signed_sum"])) for row in failures])
    cutoff = float(np.quantile(net, 0.25))
    quiet = [
        row
        for row in failures
        if abs(float(row["safe_response_signed_sum"])) <= cutoff
    ]
    other = [row for row in failures if row not in quiet]
    fields = (
        "feature_displacement_l2_energy",
        "safe_response_absolute_sum",
        "safe_response_cancellation_fraction",
        "gradient_alignment_fraction",
        "linearization_error_fraction",
        "mean_relu_gate_flip_fraction",
    )
    return {
        "definition": (
            "failures in the lowest split-specific quarter of absolute net "
            "25-step SAFE response; this is a descriptive rank group"
        ),
        "absolute_net_response_cutoff": cutoff,
        "failures": len(failures),
        "quiet_failures": len(quiet),
        "metrics": {
            field: {
                "quiet": distribution(_selected_values(quiet, field, lambda _row: True)),
                "other_failures": distribution(
                    _selected_values(other, field, lambda _row: True)
                ),
            }
            for field in fields
        },
    }


def split_geometry_summary(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    rows = [derived_geometry(record) for record in records]
    result = {}
    for offset, split in enumerate(("development", "holdout")):
        selected = [row for row in rows if row["analysis_split"] == split]
        result[split] = {
            "physical_continuations": len(selected),
            "outcomes": {
                name: sum(row["outcome_group"] == name for row in selected)
                for name in (
                    "successful_continuation",
                    "detected_failure",
                    "silent_failure",
                )
            },
            "alarms_within_25_steps": sum(
                row["safe_alarm_within_25_steps"] for row in selected
            ),
            "comparisons": outcome_comparisons(
                selected,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100 * offset,
            ),
            "relationships": rank_correlations(selected),
            "quiet_failures": quiet_failure_summary(selected),
            "temporal_comparisons": {
                metric: binary_metric_summary(
                    selected,
                    metric=metric,
                    positive=lambda row: bool(row["policy_failure"]),
                    negative=lambda row: not bool(row["policy_failure"]),
                )
                for metric in TEMPORAL_FIELDS
            },
            "by_task": task_breakdown(selected),
        }
    return result
