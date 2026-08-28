from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_risk import (
    OUTCOME,
    eligible_rows,
    outcome_counts,
    predict_probability,
    top_risk_group,
    trajectory_groups,
)
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen residual-risk rankings on untouched contexts."
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def _metrics(
    labels: Any,
    probabilities: Any,
    np: Any,
    sklearn_metrics: Any,
    *,
    include_calibration: bool,
    calibration_curve: Any = None,
) -> dict[str, Any]:
    indices, threshold = top_risk_group(probabilities.astype(float).tolist())
    base_rate = float(labels.mean())
    top_rate = float(labels[indices].mean())
    result = {
        "base_rate": base_rate,
        "roc_auc": float(sklearn_metrics.roc_auc_score(labels, probabilities)),
        "average_precision": float(
            sklearn_metrics.average_precision_score(labels, probabilities)
        ),
        "brier_score": float(sklearn_metrics.brier_score_loss(labels, probabilities)),
        "top_risk_group": {
            "target_fraction": 0.2,
            "tie_policy": "include every intervention tied at the boundary",
            "threshold": float(threshold),
            "interventions": len(indices),
            "fraction": len(indices) / len(labels),
            "residual_interventions": int(labels[indices].sum()),
            "rate": top_rate,
            "enrichment_over_uniform": top_rate / base_rate if base_rate else None,
        },
    }
    if include_calibration:
        if calibration_curve is None:
            raise ValueError("calibration function is required")
        observed, predicted = calibration_curve(
            labels, probabilities, n_bins=5, strategy="quantile"
        )
        result["calibration_by_predicted_risk_fifth"] = [
            {
                "mean_predicted_probability": float(mean),
                "observed_rate": float(rate),
            }
            for mean, rate in zip(predicted, observed, strict=True)
        ]
    return result


def _bootstrap(
    rows: list[dict[str, Any]],
    probabilities: Any,
    *,
    samples: int,
    seed: int,
    np: Any,
    sklearn_metrics: Any,
) -> dict[str, list[float] | int]:
    groups = list(trajectory_groups(rows).values())
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "brier_score": [],
        "top_risk_enrichment": [],
    }
    labels = np.asarray([int(row[OUTCOME]) for row in rows], dtype=int)
    for _ in range(samples):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        indices = np.asarray([index for group in sampled for index in group], dtype=int)
        selected_labels = labels[indices]
        selected_probabilities = probabilities[indices]
        if len(np.unique(selected_labels)) == 2:
            result = _metrics(
                selected_labels,
                selected_probabilities,
                np,
                sklearn_metrics,
                include_calibration=False,
            )
            values["roc_auc"].append(result["roc_auc"])
            values["average_precision"].append(result["average_precision"])
            values["top_risk_enrichment"].append(
                result["top_risk_group"]["enrichment_over_uniform"]
            )
        values["brier_score"].append(
            float(
                sklearn_metrics.brier_score_loss(
                    selected_labels, selected_probabilities
                )
            )
        )

    output: dict[str, list[float] | int] = {"requested_samples": samples}
    for name, estimates in values.items():
        output[f"{name}_valid_samples"] = len(estimates)
        output[f"{name}_95"] = (
            np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()
            if estimates
            else []
        )
    return output


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    analysis = load_json(args.analysis)
    frozen = load_json(args.models)
    if analysis.get("analysis_split") != "holdout":
        raise ValueError("frozen rankings must be evaluated on the holdout split")
    if frozen.get("development_only") is not True or frozen.get("outcome") != OUTCOME:
        raise ValueError("risk model artifact is not a frozen development fit")
    rows = eligible_rows(analysis)
    if not rows:
        raise ValueError("holdout analysis has no eligible causal outcomes")

    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn import metrics as sklearn_metrics

    labels = np.asarray([int(row[OUTCOME]) for row in rows], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("holdout outcomes must contain both classes")
    model_results = {}
    record_rows = []
    for model_index, (name, model) in enumerate(sorted(frozen["models"].items())):
        probabilities = np.asarray(
            [predict_probability(model, row) for row in rows], dtype=np.float64
        )
        result = _metrics(
            labels,
            probabilities,
            np,
            sklearn_metrics,
            include_calibration=True,
            calibration_curve=calibration_curve,
        )
        result["trajectory_cluster_bootstrap"] = _bootstrap(
            rows,
            probabilities,
            samples=args.bootstrap_samples,
            seed=args.seed + model_index,
            np=np,
            sklearn_metrics=sklearn_metrics,
        )
        model_results[name] = result
        for row, probability in zip(rows, probabilities, strict=True):
            record_rows.append(
                {
                    "model": name,
                    "record_id": row["record_id"],
                    "task_id": row["task_id"],
                    "episode_index": row["episode_index"],
                    "physical_run": row["physical_run"],
                    "observed_residual_failure": row[OUTCOME],
                    "predicted_probability": float(probability),
                }
            )

    output = {
        "schema_version": 1,
        "analysis": "held-out OpenVLA language-block residual-risk ranking",
        "estimand": frozen["estimand"],
        "outcome": OUTCOME,
        "population": outcome_counts(rows),
        "artifacts": {
            "holdout_analysis": {
                "path": str(args.analysis.resolve()),
                "sha256": file_sha256(args.analysis),
            },
            "frozen_models": {
                "path": str(args.models.resolve()),
                "sha256": file_sha256(args.models),
            },
        },
        "models": model_results,
    }
    write_json_atomic(args.output, output)
    write_csv_atomic(args.records_csv, record_rows)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
