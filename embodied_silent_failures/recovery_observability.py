from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from embodied_silent_failures.intervention_atlas_followups import (
    classification_metrics,
    paired_metric_bootstrap,
)
from embodied_silent_failures.recovery_evidence import (
    HORIZONS,
    MODEL_FAMILIES,
    direct_measurements,
    feature_dict,
)


def trajectory_key(row: dict[str, Any]) -> str:
    return f"task{row['task_id']}:episode{row['episode_index']}"


def trajectory_fold_assignments(
    rows: list[dict[str, Any]], folds: int
) -> dict[str, int]:
    import numpy as np
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([trajectory_key(row) for row in rows])
    assignments = {}
    for fold, (_train, test) in enumerate(
        GroupKFold(folds).split(np.zeros(len(rows)), groups=groups)
    ):
        for index in test:
            assignments[groups[index]] = fold
    return assignments


def _pipeline():
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True, sort=True)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "logistic_regression",
                LogisticRegression(C=1.0, max_iter=5_000, solver="lbfgs"),
            ),
        ]
    )


def fit_model(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    horizon: int,
    family: str,
    fold_assignments: dict[str, int],
) -> tuple[dict[str, Any], Any, Any]:
    import numpy as np
    from sklearn.model_selection import PredefinedSplit, cross_val_predict

    development_features = [
        feature_dict(row, horizon=horizon, family=family) for row in development
    ]
    holdout_features = [
        feature_dict(row, horizon=horizon, family=family) for row in holdout
    ]
    development_labels = np.asarray(
        [int(row["policy_failure"]) for row in development], dtype=int
    )
    holdout_labels = np.asarray(
        [int(row["policy_failure"]) for row in holdout], dtype=int
    )
    test_folds = np.asarray(
        [fold_assignments[trajectory_key(row)] for row in development], dtype=int
    )
    cross_validated = cross_val_predict(
        _pipeline(),
        development_features,
        development_labels,
        cv=PredefinedSplit(test_folds),
        method="predict_proba",
    )[:, 1]
    model = _pipeline()
    model.fit(development_features, development_labels)
    holdout_probabilities = model.predict_proba(holdout_features)[:, 1]
    return (
        {
            "feature_count": len(model.named_steps["vectorizer"].feature_names_),
            "development_grouped_cross_validation": classification_metrics(
                development_labels, cross_validated
            ),
            "holdout": classification_metrics(
                holdout_labels, holdout_probabilities
            ),
        },
        cross_validated,
        holdout_probabilities,
    )


def within_context_concordance(
    rows: list[dict[str, Any]],
    values: Any,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Rank failures within a saved state and cluster uncertainty by trajectory."""
    import numpy as np

    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[str(row["context_id"])].append((row, float(value)))

    context_contributions: dict[
        tuple[int, int], list[tuple[int, float]]
    ] = defaultdict(list)
    for members in grouped.values():
        failures = [value for row, value in members if row["policy_failure"]]
        successes = [value for row, value in members if not row["policy_failure"]]
        if not failures or not successes:
            continue
        wins = sum(left > right for left in failures for right in successes)
        ties = sum(left == right for left in failures for right in successes)
        pairs = len(failures) * len(successes)
        row = members[0][0]
        trajectory = (int(row["task_id"]), int(row["episode_index"]))
        context_contributions[trajectory].append((pairs, wins + 0.5 * ties))

    def estimate(trajectories: list[tuple[int, int]]) -> float:
        contributions = [
            contribution
            for trajectory in trajectories
            for contribution in context_contributions[trajectory]
        ]
        return sum(wins for _pairs, wins in contributions) / sum(
            pairs for pairs, _wins in contributions
        )

    trajectories = list(context_contributions)
    if not trajectories:
        return {
            "comparable_contexts": 0,
            "failure_success_pairs": 0,
            "pair_weighted_concordance": None,
            "trajectory_bootstrap_interval_95": None,
        }
    point = estimate(trajectories)
    rng = random.Random(seed)
    samples = [
        estimate([trajectories[rng.randrange(len(trajectories))] for _ in trajectories])
        for _ in range(bootstrap_samples)
    ]
    return {
        "comparable_contexts": sum(
            len(values) for values in context_contributions.values()
        ),
        "trajectories_with_comparable_contexts": len(trajectories),
        "failure_success_pairs": sum(
            pairs
            for values in context_contributions.values()
            for pairs, _wins in values
        ),
        "pair_weighted_concordance": point,
        "trajectory_bootstrap_interval_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def direct_signal_summary(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    measurements = [direct_measurements(row, horizon) for row in rows]
    names = tuple(sorted(measurements[0]))
    labels = np.asarray([int(row["policy_failure"]) for row in rows], dtype=int)
    return {
        name: {
            "larger_value_failure_roc_auc": float(
                roc_auc_score(labels, [values[name] for values in measurements])
            ),
            "within_context": within_context_concordance(
                rows,
                [values[name] for values in measurements],
                bootstrap_samples=bootstrap_samples,
                seed=seed + index,
            ),
        }
        for index, name in enumerate(names)
    }


def analyze_horizon(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    horizon: int,
    fold_assignments: dict[str, int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    labels = np.asarray([int(row["policy_failure"]) for row in holdout], dtype=int)
    models = {}
    development_probabilities = {}
    holdout_probabilities = {}
    for index, family in enumerate(MODEL_FAMILIES):
        (
            models[family],
            development_probabilities[family],
            holdout_probabilities[family],
        ) = fit_model(
            development,
            holdout,
            horizon=horizon,
            family=family,
            fold_assignments=fold_assignments,
        )
        models[family]["holdout_within_context"] = within_context_concordance(
            holdout,
            holdout_probabilities[family],
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        models[family]["development_grouped_cv_within_context"] = (
            within_context_concordance(
                development,
                development_probabilities[family],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 50 + index,
            )
        )

    comparisons = {
        "policy_over_context": ("policy", "context"),
        "physical_current_over_policy": ("policy_physical_current", "policy"),
        "physical_path_over_current": (
            "policy_physical_path",
            "policy_physical_current",
        ),
        "safe_over_policy": ("policy_safe", "policy"),
        "safe_over_physical_path": ("all", "policy_physical_path"),
        "physical_path_over_safe": ("all", "policy_safe"),
    }
    differences = {
        name: paired_metric_bootstrap(
            holdout,
            labels,
            holdout_probabilities[candidate],
            holdout_probabilities[baseline],
            samples=bootstrap_samples,
            seed=seed + 100 + index,
        )
        for index, (name, (candidate, baseline)) in enumerate(comparisons.items())
    }
    return {
        "horizon_steps": horizon,
        "population": {
            "development": len(development),
            "development_failures": int(
                sum(row["policy_failure"] for row in development)
            ),
            "holdout": len(holdout),
            "holdout_failures": int(sum(row["policy_failure"] for row in holdout)),
        },
        "models": models,
        "direct_signal_diagnostics": {
            "development": direct_signal_summary(
                development,
                horizon=horizon,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 500,
            ),
            "holdout": direct_signal_summary(
                holdout,
                horizon=horizon,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 600,
            ),
        },
        "holdout_trajectory_bootstrap_differences": differences,
    }


def analyze_recovery_observability(
    rows: list[dict[str, Any]],
    *,
    folds: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    def observed_before_outcome(row: dict[str, Any], horizon: int) -> bool:
        boundary = int(row["fault_step"]) + horizon
        return (
            int(row["faulted_length"]) > boundary
            and int(row["control_length"]) > boundary
        )

    development = [row for row in rows if row["analysis_split"] == "development"]
    holdout = [row for row in rows if row["analysis_split"] == "holdout"]
    fold_assignments = trajectory_fold_assignments(development, folds)
    return {
        "population": {
            "physical_continuations": len(rows),
            "development": len(development),
            "development_failures": sum(row["policy_failure"] for row in development),
            "holdout": len(holdout),
            "holdout_failures": sum(row["policy_failure"] for row in holdout),
        },
        "horizons": {
            str(horizon): analyze_horizon(
                [row for row in development if observed_before_outcome(row, horizon)],
                [row for row in holdout if observed_before_outcome(row, horizon)],
                horizon=horizon,
                fold_assignments=fold_assignments,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1_000 * horizon,
            )
            for horizon in HORIZONS
        },
    }
