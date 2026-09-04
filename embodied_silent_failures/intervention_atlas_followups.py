from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable

from embodied_silent_failures.fit_intervention_atlas_risk import _metrics
from embodied_silent_failures.intervention_atlas_risk import (
    ACTION_FEATURES,
    CONTEXT_FEATURES,
    GRAPH_FEATURES,
    MODEL_SPECS,
    MONITOR_FEATURES,
    model_features,
    trajectory_groups,
)

POSTHOC_MODEL_SPECS = {
    "miss_phase": CONTEXT_FEATURES,
    "miss_graph_context": GRAPH_FEATURES + CONTEXT_FEATURES,
    "miss_action_context": CONTEXT_FEATURES + ACTION_FEATURES,
    "miss_full": (
        GRAPH_FEATURES + CONTEXT_FEATURES + ACTION_FEATURES + MONITOR_FEATURES
    ),
    "silent_action_context": CONTEXT_FEATURES + ACTION_FEATURES,
    "silent_action_monitor_context": (
        CONTEXT_FEATURES + ACTION_FEATURES + MONITOR_FEATURES
    ),
}

STABILITY_MODEL_NAMES = (
    "task_only",
    "phase_only",
    "site_only",
    "context_only",
    "site_plus_context",
    "site_context_interactions",
)


def _pipeline():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True, sort=True)),
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def classification_metrics(labels, probabilities) -> dict[str, Any]:
    from sklearn import metrics

    result = _metrics(labels, probabilities, metrics)
    result.update(
        {
            "mean_prediction": float(probabilities.mean()),
            "mean_prediction_minus_observed_rate": float(
                probabilities.mean() - labels.mean()
            ),
            "brier_score": float(metrics.brier_score_loss(labels, probabilities)),
            "log_loss": float(metrics.log_loss(labels, probabilities, labels=[0, 1])),
        }
    )
    return result


def grouped_cross_validated_probabilities(
    rows: list[dict[str, Any]],
    features: list[dict[str, float]],
    outcome: str,
    *,
    folds: int,
):
    import numpy as np
    from sklearn.model_selection import GroupKFold, cross_val_predict

    groups = np.asarray(
        [f"task{row['task_id']}:episode{row['episode_index']}" for row in rows]
    )
    if len(set(groups)) < folds:
        raise ValueError("fewer trajectory groups than requested folds")
    labels = np.asarray([int(row[outcome]) for row in rows], dtype=int)
    probabilities = cross_val_predict(
        _pipeline(),
        features,
        labels,
        groups=groups,
        cv=GroupKFold(folds),
        method="predict_proba",
    )[:, 1]
    return labels, probabilities


def fit_posthoc_model(
    development_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    features: tuple[str, ...],
    outcome: str,
    select: Callable[[dict[str, Any]], bool] = lambda row: True,
    folds: int = 5,
) -> tuple[dict[str, Any], Any, Any]:
    import numpy as np

    development = [row for row in development_rows if select(row)]
    holdout = [row for row in holdout_rows if select(row)]
    development_features = [model_features(row, features) for row in development]
    holdout_features = [model_features(row, features) for row in holdout]
    development_labels, development_probabilities = (
        grouped_cross_validated_probabilities(
            development,
            development_features,
            outcome,
            folds=folds,
        )
    )
    model = _pipeline()
    model.fit(development_features, development_labels)
    holdout_labels = np.asarray([int(row[outcome]) for row in holdout], dtype=int)
    holdout_probabilities = model.predict_proba(holdout_features)[:, 1]
    return (
        {
            "development_grouped_cross_validation": classification_metrics(
                development_labels, development_probabilities
            ),
            "holdout": classification_metrics(
                holdout_labels, holdout_probabilities
            ),
            "development_rows": len(development),
            "holdout_rows": len(holdout),
            "features": features,
        },
        model,
        holdout_probabilities,
    )


def probability_summary(values) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "minimum": float(values.min()),
        "quantiles": {
            "0.10": float(np.quantile(values, 0.1)),
            "0.50": float(np.quantile(values, 0.5)),
            "0.90": float(np.quantile(values, 0.9)),
        },
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
    }


def paired_metric_bootstrap(
    rows: list[dict[str, Any]],
    labels,
    candidate,
    baseline,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn import metrics

    groups = list(trajectory_groups(rows).values())
    labels = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    point = {
        "roc_auc_candidate_minus_baseline": float(
            metrics.roc_auc_score(labels, candidate)
            - metrics.roc_auc_score(labels, baseline)
        ),
        "average_precision_candidate_minus_baseline": float(
            metrics.average_precision_score(labels, candidate)
            - metrics.average_precision_score(labels, baseline)
        ),
        "brier_improvement_candidate_over_baseline": float(
            metrics.brier_score_loss(labels, baseline)
            - metrics.brier_score_loss(labels, candidate)
        ),
    }
    estimates = {name: [] for name in point}
    rng = random.Random(seed)
    for _ in range(samples):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        indices = np.asarray([index for group in chosen for index in group], dtype=int)
        selected_labels = labels[indices]
        if len(np.unique(selected_labels)) != 2:
            continue
        estimates["roc_auc_candidate_minus_baseline"].append(
            metrics.roc_auc_score(selected_labels, candidate[indices])
            - metrics.roc_auc_score(selected_labels, baseline[indices])
        )
        estimates["average_precision_candidate_minus_baseline"].append(
            metrics.average_precision_score(selected_labels, candidate[indices])
            - metrics.average_precision_score(selected_labels, baseline[indices])
        )
        estimates["brier_improvement_candidate_over_baseline"].append(
            metrics.brier_score_loss(selected_labels, baseline[indices])
            - metrics.brier_score_loss(selected_labels, candidate[indices])
        )
    return {
        name: {
            "estimate": point[name],
            "valid_samples": len(values),
            "interval_95": np.quantile(values, [0.025, 0.975]).tolist()
            if values
            else [],
            "probability_positive": sum(value > 0 for value in values) / len(values)
            if values
            else None,
        }
        for name, values in estimates.items()
    }


def stability_features(row: dict[str, Any], name: str) -> dict[str, float]:
    site = str(row["site_id"])
    task = str(row["task_id"])
    phase = str(row["phase"])
    features = {}
    if name in {"site_only", "site_plus_context", "site_context_interactions"}:
        features[f"site={site}"] = 1.0
    context_models = {
        "context_only",
        "site_plus_context",
        "site_context_interactions",
    }
    if name == "task_only" or name in context_models:
        features[f"task={task}"] = 1.0
    if name == "phase_only" or name in context_models:
        features[f"phase={phase}"] = 1.0
    if name == "site_context_interactions":
        features[f"site_task={site}|{task}"] = 1.0
        features[f"site_phase={site}|{phase}"] = 1.0
    if not features:
        raise ValueError(f"unknown stability model: {name}")
    return features


def fit_stability_model(
    development_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    name: str,
    outcome: str,
    select: Callable[[dict[str, Any]], bool] = lambda row: True,
    folds: int = 5,
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    development = [row for row in development_rows if select(row)]
    holdout = [row for row in holdout_rows if select(row)]
    development_features = [stability_features(row, name) for row in development]
    holdout_features = [stability_features(row, name) for row in holdout]
    development_labels, development_probabilities = (
        grouped_cross_validated_probabilities(
            development,
            development_features,
            outcome,
            folds=folds,
        )
    )
    model = _pipeline()
    model.fit(development_features, development_labels)
    holdout_labels = np.asarray([int(row[outcome]) for row in holdout], dtype=int)
    holdout_probabilities = model.predict_proba(holdout_features)[:, 1]
    return (
        {
            "development_grouped_cross_validation": classification_metrics(
                development_labels, development_probabilities
            ),
            "holdout": classification_metrics(
                holdout_labels, holdout_probabilities
            ),
            "development_rows": len(development),
            "holdout_rows": len(holdout),
        },
        holdout_probabilities,
    )


def site_rates(
    rows: list[dict[str, Any]],
    outcome: str,
    *,
    select: Callable[[dict[str, Any]], bool] = lambda row: True,
) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    positives: dict[str, int] = defaultdict(int)
    for row in rows:
        if not select(row):
            continue
        site = str(row["site_id"])
        totals[site] += 1
        positives[site] += int(row[outcome])
    return {
        site: positives[site] / total
        for site, total in totals.items()
        if total
    }


def rank_stability(
    development_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    outcome: str,
    *,
    select: Callable[[dict[str, Any]], bool] = lambda row: True,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import spearmanr

    def compare(left: list[dict[str, Any]], right: list[dict[str, Any]]):
        left_rates = site_rates(left, outcome, select=select)
        right_rates = site_rates(right, outcome, select=select)
        sites = sorted(set(left_rates) & set(right_rates))
        if len(sites) < 2:
            return None, 0, None
        left_values = [left_rates[site] for site in sites]
        right_values = [right_rates[site] for site in sites]
        correlation = spearmanr(left_values, right_values).statistic
        count = max(1, math.ceil(len(sites) / 5))
        left_top = set(
            sorted(sites, key=lambda site: (-left_rates[site], site))[:count]
        )
        right_top = set(
            sorted(sites, key=lambda site: (-right_rates[site], site))[:count]
        )
        overlap = len(left_top & right_top) / len(left_top | right_top)
        return (
            None if math.isnan(float(correlation)) else float(correlation),
            len(sites),
            overlap,
        )

    point, sites, overlap = compare(development_rows, holdout_rows)
    development_rates = site_rates(development_rows, outcome, select=select)
    holdout_rates = site_rates(holdout_rows, outcome, select=select)
    development_groups = list(trajectory_groups(development_rows).values())
    holdout_groups = list(trajectory_groups(holdout_rows).values())
    rng = random.Random(seed)
    correlations = []
    for _ in range(samples):
        development = [
            development_rows[index]
            for _ in development_groups
            for index in development_groups[rng.randrange(len(development_groups))]
        ]
        holdout = [
            holdout_rows[index]
            for _ in holdout_groups
            for index in holdout_groups[rng.randrange(len(holdout_groups))]
        ]
        correlation, _, _ = compare(development, holdout)
        if correlation is not None:
            correlations.append(correlation)
    return {
        "spearman_development_vs_holdout": point,
        "sites_with_defined_rates_in_both_splits": sites,
        "highest_fifth_jaccard": overlap,
        "highest_fifth_tie_break": "site ID in ascending order",
        "development_site_rates": dict(sorted(development_rates.items())),
        "holdout_site_rates": dict(sorted(holdout_rates.items())),
        "trajectory_cluster_bootstrap": {
            "valid_samples": len(correlations),
            "interval_95": np.quantile(correlations, [0.025, 0.975]).tolist()
            if correlations
            else [],
        },
    }


def phase_rank_stability(
    rows: list[dict[str, Any]],
    outcome: str,
    *,
    select: Callable[[dict[str, Any]], bool] = lambda row: True,
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    rates = {
        phase: site_rates(
            [row for row in rows if row["phase"] == phase],
            outcome,
            select=select,
        )
        for phase in ("early", "middle", "late")
    }
    output = {}
    for left, right in (("early", "middle"), ("early", "late"), ("middle", "late")):
        sites = sorted(set(rates[left]) & set(rates[right]))
        statistic = spearmanr(
            [rates[left][site] for site in sites],
            [rates[right][site] for site in sites],
        ).statistic
        output[f"{left}_vs_{right}"] = {
            "sites": len(sites),
            "spearman": None
            if math.isnan(float(statistic))
            else float(statistic),
        }
    return output


def physical_equivalence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_site: dict[str, dict[tuple[int, int, str], str]] = defaultdict(dict)
    by_context: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for row in rows:
        context = (row["task_id"], row["episode_index"], row["phase"])
        by_site[str(row["site_id"])][context] = str(row["physical_run"])
        by_context[context].add(str(row["physical_run"]))
    expected_contexts = set(by_context)
    incomplete = [
        site for site, values in by_site.items() if set(values) != expected_contexts
    ]
    if incomplete:
        raise ValueError(f"sites missing contexts in equivalence audit: {incomplete}")

    classes: dict[tuple[tuple[tuple[int, int, str], str], ...], list[str]] = (
        defaultdict(list)
    )
    for site, values in by_site.items():
        classes[tuple(sorted(values.items()))].append(site)
    members = sorted((sorted(sites) for sites in classes.values()), key=lambda x: x[0])
    branch_counts = [len(values) for values in by_context.values()]
    return {
        "graph_sites": len(by_site),
        "behavioral_equivalence_classes": len(classes),
        "class_sizes_descending": sorted(
            (len(sites) for sites in classes.values()), reverse=True
        ),
        "classes_with_multiple_sites": [
            sites for sites in members if len(sites) > 1
        ],
        "distinct_physical_continuations_per_context": {
            "minimum": min(branch_counts),
            "median": statistics.median(branch_counts),
            "maximum": max(branch_counts),
        },
        "definition": (
            "sites share a class only when they map to the same recorded physical "
            "continuation in every eligible context"
        ),
    }


def validate_frozen_models(artifact: dict[str, Any]) -> None:
    if artifact.get("specifications") != MODEL_SPECS:
        raise ValueError("frozen model specifications differ from analysis code")
