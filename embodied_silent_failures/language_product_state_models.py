from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any


MODEL_FAMILIES = ("linear", "extra_trees", "tuned_forest")
FOREST_CANDIDATES = (
    {"max_depth": None, "max_features": "sqrt", "min_samples_leaf": 2},
    {"max_depth": None, "max_features": "sqrt", "min_samples_leaf": 5},
    {"max_depth": None, "max_features": 0.25, "min_samples_leaf": 10},
    {"max_depth": 12, "max_features": 0.25, "min_samples_leaf": 5},
)
LINEAR_CANDIDATES = tuple({"C": value} for value in (0.001, 0.01, 0.1, 1.0))


def _estimator(family: str, seed: int, parameters: dict[str, Any] | None = None) -> Any:
    if family == "linear":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float((parameters or {"C": 1.0})["C"]),
                class_weight=None,
                max_iter=3_000,
                solver="lbfgs",
            ),
        )
    if family == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )
    if family == "tuned_forest":
        from sklearn.ensemble import RandomForestClassifier

        if parameters is None:
            raise ValueError("tuned forest requires selected parameters")
        return RandomForestClassifier(
            n_estimators=250,
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
            **parameters,
        )
    raise ValueError(f"unknown model family: {family}")


def _grouped_log_loss_selection(
    np: Any,
    family: str,
    candidates: tuple[dict[str, Any], ...],
    matrix: Any,
    labels: Any,
    groups: list[str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold

    unique_groups = sorted(set(groups))
    folds = min(4, len(unique_groups))
    if folds < 2:
        raise ValueError(f"{family} selection requires two trajectories")
    splitter = GroupKFold(n_splits=folds)
    records = []
    for index, parameters in enumerate(candidates):
        losses = []
        for fold, (train, validation) in enumerate(
            splitter.split(matrix, labels, groups), start=1
        ):
            estimator = _estimator(family, seed + index * 10 + fold, parameters)
            estimator.fit(matrix[train], labels[train])
            probabilities = estimator.predict_proba(matrix[validation])[:, 1]
            losses.append(
                float(log_loss(labels[validation], probabilities, labels=[0, 1]))
            )
        records.append(
            {
                "parameters": parameters,
                "fold_log_loss": losses,
                "mean_log_loss": float(np.mean(losses)),
            }
        )
    selected_index = min(
        range(len(records)),
        key=lambda index: (records[index]["mean_log_loss"], index),
    )
    return candidates[selected_index], {
        "selection_metric": "mean grouped validation log loss",
        "group": "task and episode trajectory",
        "folds": folds,
        "candidates": records,
        "selected_index": selected_index,
    }


def fit_predict(
    np: Any,
    family: str,
    development_matrix: Any,
    development_labels: Any,
    development_groups: list[str],
    holdout_matrix: Any,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    parameters = None
    selection = None
    if family == "linear":
        parameters, selection = _grouped_log_loss_selection(
            np,
            "linear",
            LINEAR_CANDIDATES,
            development_matrix,
            development_labels,
            development_groups,
            seed,
        )
    elif family == "tuned_forest":
        parameters, selection = _grouped_log_loss_selection(
            np,
            "tuned_forest",
            FOREST_CANDIDATES,
            development_matrix,
            development_labels,
            development_groups,
            seed,
        )
    estimator = _estimator(family, seed, parameters)
    estimator.fit(development_matrix, development_labels)
    probabilities = estimator.predict_proba(holdout_matrix)[:, 1]
    return probabilities, {
        "family": family,
        "parameters": parameters,
        "development_selection": selection,
    }


def binary_metrics(
    np: Any, labels: Any, probabilities: Any, sample_weight: Any | None = None
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("metrics require equal nonempty labels and probabilities")
    if len(np.unique(labels)) != 2:
        raise ValueError("metrics require both outcome classes")
    return {
        "rows": len(labels),
        "positive_outcomes": int(np.sum(labels)),
        "base_rate": float(np.average(labels, weights=sample_weight)),
        "roc_auc": float(
            roc_auc_score(labels, probabilities, sample_weight=sample_weight)
        ),
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=sample_weight)
        ),
        "brier_score": float(
            brier_score_loss(labels, probabilities, sample_weight=sample_weight)
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                sample_weight=sample_weight,
                labels=[0, 1],
            )
        ),
    }


def alias_weights(np: Any, rows: list[dict[str, Any]]) -> Any:
    counts = Counter(str(row["physical_run"]) for row in rows)
    return np.asarray(
        [1.0 / counts[str(row["physical_run"])] for row in rows], dtype=np.float64
    )


def alias_audit(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["physical_run"])].append(row)
    aliases = [members for members in groups.values() if len(members) > 1]
    mixed = [
        members
        for members in aliases
        if len({bool(member[outcome]) for member in members}) > 1
    ]
    return {
        "physical_branches": len(groups),
        "branches_with_multiple_source_layers": len(aliases),
        "largest_source_layer_alias_group": max(map(len, aliases), default=1),
        "alias_groups_with_mixed_outcome": len(mixed),
        "rows_in_mixed_alias_groups": sum(map(len, mixed)),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_differences(
    np: Any,
    rows: list[dict[str, Any]],
    labels: Any,
    baseline: Any,
    candidate: Any,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(int(row["task_id"]), int(row["episode_index"]))].append(index)
    clusters = list(grouped.values())
    rng = random.Random(seed)
    auc_differences = []
    brier_improvements = []
    for _ in range(samples):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        indices = np.asarray([index for cluster in selected for index in cluster])
        selected_labels = labels[indices]
        if len(np.unique(selected_labels)) != 2:
            continue
        auc_differences.append(
            float(
                roc_auc_score(selected_labels, candidate[indices])
                - roc_auc_score(selected_labels, baseline[indices])
            )
        )
        brier_improvements.append(
            float(
                brier_score_loss(selected_labels, baseline[indices])
                - brier_score_loss(selected_labels, candidate[indices])
            )
        )

    def summary(values: list[float]) -> dict[str, Any]:
        return {
            "valid_samples": len(values),
            "median": float(_percentile(values, 0.5)) if values else None,
            "95_interval": (
                [_percentile(values, 0.025), _percentile(values, 0.975)]
                if values
                else []
            ),
        }

    return {
        "requested_samples": samples,
        "trajectory_clusters": len(clusters),
        "candidate_minus_product_roc_auc": summary(auc_differences),
        "product_minus_candidate_brier_score": summary(brier_improvements),
    }
