from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_risk import (
    MODEL_SPECS,
    OUTCOME,
    eligible_rows,
    feature_row,
    outcome_counts,
)
from embodied_silent_failures.provenance import file_sha256, git_dirty, git_revision, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze simple residual-risk rankings on the development split."
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    analysis = load_json(args.analysis)
    if analysis.get("analysis_split") != "development":
        raise ValueError("risk rankings must be fitted on the development split")
    rows = eligible_rows(analysis)
    if not rows:
        raise ValueError("development analysis has no eligible causal outcomes")

    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([int(row[OUTCOME]) for row in rows], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("development outcomes must contain both classes")
    models = {}
    for name, specification in MODEL_SPECS.items():
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
        models[name] = {
            "features": [
                {"name": feature, "transform": transform}
                for feature, transform in specification
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
            "development_resubstitution": {
                "average_precision": float(average_precision_score(labels, probabilities)),
                "brier_score": float(brier_score_loss(labels, probabilities)),
                "roc_auc": float(roc_auc_score(labels, probabilities)),
            },
        }

    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "analysis": "frozen OpenVLA language-block residual-risk rankings",
        "estimand": (
            "Uniform choice of one language block at one sampled context, conditional "
            "on a successful fresh control."
        ),
        "outcome": OUTCOME,
        "development_only": True,
        "selection_policy": (
            "Freeze and carry all three predeclared descriptions to holdout; development "
            "fit metrics are descriptive and do not select a winner."
        ),
        "training_population": outcome_counts(rows),
        "source_analysis": {
            "path": str(args.analysis.resolve()),
            "sha256": file_sha256(args.analysis),
        },
        "implementation": {
            "experiment_code_revision": git_revision(project_root),
            "experiment_code_dirty": git_dirty(project_root),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        },
        "models": models,
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
