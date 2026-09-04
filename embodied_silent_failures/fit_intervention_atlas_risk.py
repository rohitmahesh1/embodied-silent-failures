from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic, write_pickle_atomic
from embodied_silent_failures.intervention_atlas_risk import (
    MODEL_SPECS,
    attach_trajectory_weights,
    load_analysis_rows,
    model_features,
    rate_table,
)
from embodied_silent_failures.provenance import file_sha256, git_dirty, git_revision


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit frozen graph-atlas risk rankings on development trajectories."
    )
    parser.add_argument("--analysis", action="append", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def _metrics(labels, probabilities, sklearn_metrics) -> dict:
    order = probabilities.argsort()[::-1]
    count = max(1, (len(order) + 4) // 5)
    selected = order[:count]
    base_rate = float(labels.mean())
    selected_rate = float(labels[selected].mean())
    return {
        "roc_auc": float(sklearn_metrics.roc_auc_score(labels, probabilities)),
        "average_precision": float(
            sklearn_metrics.average_precision_score(labels, probabilities)
        ),
        "base_rate": base_rate,
        "highest_ranked_fifth": {
            "interventions": len(selected),
            "rate": selected_rate,
            "enrichment_over_uniform": (
                selected_rate / base_rate if base_rate else None
            ),
        },
    }


def main() -> None:
    args = _arguments()
    if args.folds < 2:
        raise ValueError("at least two trajectory folds are required")
    rows, source = load_analysis_rows(args.analysis, analysis_split="development")
    manifest = attach_trajectory_weights(rows, args.manifest)
    if not rows:
        raise ValueError("development atlas has no eligible interventions")

    import numpy as np
    import sklearn
    from sklearn import metrics as sklearn_metrics
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    groups = np.asarray(
        [f"task{row['task_id']}:episode{row['episode_index']}" for row in rows]
    )
    if len(set(groups)) < args.folds:
        raise ValueError("development atlas has fewer trajectories than folds")
    fitted = {}
    development_metrics = {}
    for name, spec in MODEL_SPECS.items():
        labels = np.asarray([int(row[spec["outcome"]]) for row in rows], dtype=int)
        if len(np.unique(labels)) != 2:
            raise ValueError(f"development outcome has fewer than two classes: {name}")
        features = [model_features(row, spec["features"]) for row in rows]
        model = Pipeline(
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
        probabilities = cross_val_predict(
            model,
            features,
            labels,
            groups=groups,
            cv=GroupKFold(args.folds),
            method="predict_proba",
        )[:, 1]
        development_metrics[name] = _metrics(
            labels, probabilities, sklearn_metrics
        )
        model.fit(features, labels)
        fitted[name] = model

    model_artifact = {
        "schema_version": 1,
        "models": fitted,
        "specifications": MODEL_SPECS,
    }
    write_pickle_atomic(args.models, model_artifact)
    output = {
        "schema_version": 1,
        "analysis": "development-only fit of graph-atlas risk rankings",
        "development_only": True,
        "analysis_code": {
            "revision": git_revision(Path(__file__).resolve().parents[1]),
            "dirty": git_dirty(Path(__file__).resolve().parents[1]),
            "fit_sha256": file_sha256(Path(__file__)),
        },
        "source": source,
        "manifest": manifest,
        "population": {
            "interventions": len(rows),
            "trajectory_clusters": len(set(groups)),
            "sites": len({row["site_id"] for row in rows}),
            "physical_runs": len({row["physical_run"] for row in rows}),
        },
        "rates": rate_table(rows),
        "modeling_contract": {
            "cross_validation": (
                f"{args.folds}-fold GroupKFold; every early/middle/late context from "
                "one trajectory remains in one fold"
            ),
            "family": "scikit-learn logistic regression with C=1",
            "class_weight": None,
            "tuning": "none",
            "numeric_transforms": (
                "log1p for nonnegative change magnitudes; raw phase fraction, "
                "changed-token fraction, and signed SAFE contribution change"
            ),
            "feature_basis": (
                "graph factors are the five fields declared before execution in the "
                "sampling stratum; local measurements are observed before terminal outcome"
            ),
        },
        "software": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": {
            "artifact": {
                "path": str(args.models.resolve()),
                "sha256": file_sha256(args.models),
            },
            "specifications": MODEL_SPECS,
            "development_cross_validation": development_metrics,
        },
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
