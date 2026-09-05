from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


ACTION_METRIC = "same_feature_action_js_at_fault"
MONITOR_METRIC = "absolute_safe_response_at_fault"
DISTANCE_METRIC = "feature_displacement_at_fault"


def attach_safe_arrays(
    record: dict[str, Any], safe_arrays: dict[str, Any], index: int
) -> dict[str, Any]:
    response = safe_arrays["monitor_increment_delta"][index]
    displacement = safe_arrays["selected_feature_l2"][index]
    return {
        **record,
        "safe_response_at_fault": float(response[0]),
        "absolute_safe_response_at_fault": float(abs(response[0])),
        "later_safe_response_absolute_sum": float(abs(response[1:]).sum()),
        "feature_displacement_at_fault": float(displacement[0]),
    }


def rank_cdf(reference: list[float], values: list[float]) -> list[float]:
    import numpy as np

    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(ordered):
        raise ValueError("rank reference cannot be empty")
    return (
        np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right")
        / len(ordered)
    ).tolist()


def rank_mismatch_diagnostic(
    development: list[dict[str, Any]], holdout: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    action_reference = [float(row[ACTION_METRIC]) for row in development]
    monitor_reference = [float(row[MONITOR_METRIC]) for row in development]

    def scores(rows: list[dict[str, Any]]) -> Any:
        action_rank = np.asarray(
            rank_cdf(action_reference, [float(row[ACTION_METRIC]) for row in rows])
        )
        monitor_rank = np.asarray(
            rank_cdf(monitor_reference, [float(row[MONITOR_METRIC]) for row in rows])
        )
        return action_rank - monitor_rank

    development_scores = scores(development)
    holdout_scores = scores(holdout)
    cutoff = float(np.quantile(development_scores, 0.8))
    selected = holdout_scores >= cutoff
    labels = np.asarray([int(row["policy_failure"]) for row in holdout])
    overall_rate = float(labels.mean()) if len(labels) else None
    selected_rate = float(labels[selected].mean()) if selected.any() else None
    return {
        "definition": (
            "development empirical rank of same-feature action divergence minus "
            "development empirical rank of absolute SAFE response at the fault step"
        ),
        "development_top_fifth_cutoff": cutoff,
        "holdout_roc_auc": (
            float(roc_auc_score(labels, holdout_scores))
            if len(set(labels.tolist())) == 2
            else None
        ),
        "holdout_top_fifth": {
            "continuations": int(selected.sum()),
            "failures": int(labels[selected].sum()),
            "failure_rate": selected_rate,
            "overall_failure_rate": overall_rate,
            "enrichment": (
                selected_rate / overall_rate
                if selected_rate is not None and overall_rate
                else None
            ),
        },
    }


def _numeric_value(row: dict[str, Any], name: str) -> float:
    value = float(row[name])
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and nonnegative")
    return math.log1p(value)


def _fit_design_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    names = (DISTANCE_METRIC, ACTION_METRIC, MONITOR_METRIC)
    values = np.asarray(
        [[_numeric_value(row, name) for name in names] for row in rows]
    )
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return {
        "task_levels": sorted({int(row["task_id"]) for row in rows}),
        "phase_levels": sorted({str(row["phase"]) for row in rows}),
        "numeric_names": list(names),
        "numeric_mean": values.mean(axis=0).tolist(),
        "numeric_scale": scale.tolist(),
    }


def _design_matrix(
    rows: list[dict[str, Any]], state: dict[str, Any], model_name: str
) -> tuple[Any, list[str]]:
    import numpy as np

    task_levels = state["task_levels"]
    phase_levels = state["phase_levels"]
    numeric_names = state["numeric_names"]
    unknown_tasks = sorted({int(row["task_id"]) for row in rows} - set(task_levels))
    unknown_phases = sorted({str(row["phase"]) for row in rows} - set(phase_levels))
    if unknown_tasks or unknown_phases:
        raise ValueError(
            f"holdout contains unknown categories: tasks={unknown_tasks}, phases={unknown_phases}"
        )

    columns = []
    names = []
    for level in task_levels[1:]:
        columns.append([float(int(row["task_id"]) == level) for row in rows])
        names.append(f"task_{level}")
    for level in phase_levels[1:]:
        columns.append([float(str(row["phase"]) == level) for row in rows])
        names.append(f"phase_{level}")

    raw = np.asarray(
        [[_numeric_value(row, name) for name in numeric_names] for row in rows]
    )
    standardized = (
        raw - np.asarray(state["numeric_mean"])
    ) / np.asarray(state["numeric_scale"])
    included = {
        "baseline": (0,),
        "action": (0, 1),
        "joint": (0, 1, 2),
        "interaction": (0, 1, 2),
    }[model_name]
    for index in included:
        columns.append(standardized[:, index].tolist())
        names.append(numeric_names[index])
    if model_name == "interaction":
        columns.append((standardized[:, 1] * standardized[:, 2]).tolist())
        names.append("action_x_monitor")
    return np.asarray(columns, dtype=np.float64).T, names


def _prediction_metrics(labels: Any, probabilities: Any) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    if len(set(labels.tolist())) != 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "log_loss": None,
        }
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
    }


def _clustered_auc_deltas(
    rows: list[dict[str, Any]],
    labels: Any,
    predictions: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(int(row["task_id"]), int(row["episode_index"]))].append(index)
    groups = list(grouped.values())
    comparisons = {
        "action_minus_baseline": ("action", "baseline"),
        "joint_minus_action": ("joint", "action"),
        "interaction_minus_joint": ("interaction", "joint"),
    }
    estimates: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        indices = [
            index
            for _group in groups
            for index in groups[rng.randrange(len(groups))]
        ]
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        for name, (left, right) in comparisons.items():
            estimates[name].append(
                float(
                    roc_auc_score(sampled_labels, predictions[left][indices])
                    - roc_auc_score(sampled_labels, predictions[right][indices])
                )
            )
    return {
        name: {
            "successful_resamples": len(values),
            "mean_delta": float(np.mean(values)) if values else None,
            "interval_95": (
                [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                if values
                else None
            ),
        }
        for name, values in estimates.items()
    }


def nested_holdout_models(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    state = _fit_design_state(development)
    development_labels = np.asarray(
        [int(row["policy_failure"]) for row in development]
    )
    holdout_labels = np.asarray([int(row["policy_failure"]) for row in holdout])
    predictions = {}
    output = {}
    for model_name in ("baseline", "action", "joint", "interaction"):
        train, feature_names = _design_matrix(development, state, model_name)
        test, _ = _design_matrix(holdout, state, model_name)
        model = LogisticRegression(C=1.0, max_iter=2_000, solver="lbfgs")
        model.fit(train, development_labels)
        probabilities = model.predict_proba(test)[:, 1]
        predictions[model_name] = probabilities
        output[model_name] = {
            "features": feature_names,
            "coefficients": {
                name: float(value)
                for name, value in zip(feature_names, model.coef_[0], strict=True)
            },
            "intercept": float(model.intercept_[0]),
            "holdout": _prediction_metrics(holdout_labels, probabilities),
        }
    return {
        "contract": {
            "fit": "development trajectories only; no hyperparameter search",
            "evaluation": (
                "previously opened holdout trajectories; exploratory replication, "
                "not confirmation"
            ),
            "baseline": "task, phase, and fault-step SAFE-input displacement",
            "action": "baseline plus same-feature action-distribution divergence",
            "joint": "action model plus absolute fault-step SAFE response",
            "interaction": "joint model plus action-by-monitor interaction",
            "numeric_transform": "log1p followed by development-set standardization",
            "regularization": "fixed L2 logistic regression with C=1",
        },
        "design_state": state,
        "models": output,
        "holdout_clustered_auc_differences": _clustered_auc_deltas(
            holdout,
            holdout_labels,
            predictions,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def coupling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    output = {}
    groups = {
        "all": rows,
        "policy_failure": [row for row in rows if row["policy_failure"]],
        "successful_continuation": [
            row for row in rows if not row["policy_failure"]
        ],
    }
    for name, selected in groups.items():
        statistic = spearmanr(
            [float(row[ACTION_METRIC]) for row in selected],
            [float(row[MONITOR_METRIC]) for row in selected],
        )
        output[name] = {
            "continuations": len(selected),
            "spearman_rho": float(statistic.statistic),
            "p_value_unclustered_descriptive_only": float(statistic.pvalue),
        }
    return output
