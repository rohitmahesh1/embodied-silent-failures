from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.fit_intervention_atlas_risk import _metrics
from embodied_silent_failures.intervention_atlas_risk import (
    MODEL_SPECS,
    attach_trajectory_weights,
    load_analysis_rows,
    model_features,
    rate_table,
    trajectory_groups,
)
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen graph-atlas risk rankings on holdout trajectories."
    )
    parser.add_argument("--analysis", action="append", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def _bootstrap_metrics(
    rows, labels, probabilities, *, samples, seed, np, sklearn_metrics
) -> dict:
    groups = list(trajectory_groups(rows).values())
    rng = random.Random(seed)
    values = {"roc_auc": [], "average_precision": [], "top_fifth_enrichment": []}
    for _ in range(samples):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        indices = np.asarray([index for group in chosen for index in group], dtype=int)
        selected_labels = labels[indices]
        if len(np.unique(selected_labels)) != 2:
            continue
        result = _metrics(
            selected_labels, probabilities[indices], sklearn_metrics
        )
        values["roc_auc"].append(result["roc_auc"])
        values["average_precision"].append(result["average_precision"])
        values["top_fifth_enrichment"].append(
            result["highest_ranked_fifth"]["enrichment_over_uniform"]
        )
    return {
        name: {
            "valid_samples": len(estimates),
            "interval_95": np.quantile(estimates, [0.025, 0.975]).tolist()
            if estimates
            else [],
        }
        for name, estimates in values.items()
    }


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    frozen = load_json(args.frozen)
    if frozen.get("development_only") is not True:
        raise ValueError("risk fit is not a frozen development-only artifact")
    if file_sha256(args.models) != frozen["models"]["artifact"]["sha256"]:
        raise ValueError("frozen model artifact hash does not match its record")

    import numpy as np
    from scipy.stats import spearmanr
    from sklearn import metrics as sklearn_metrics

    with args.models.open("rb") as file:
        model_artifact = __import__("pickle").load(file)
    if model_artifact["specifications"] != MODEL_SPECS:
        raise ValueError("frozen model specifications differ from analysis code")
    rows, source = load_analysis_rows(args.analysis, analysis_split="holdout")
    manifest = attach_trajectory_weights(rows, args.manifest)
    if source["manifest_sha256"] != frozen["source"]["manifest_sha256"]:
        raise ValueError("development and holdout use different intervention manifests")
    if source["monitor"] != frozen["source"]["monitor"]:
        raise ValueError("development and holdout use different SAFE monitors")

    results = {}
    predictions = {}
    for offset, (name, spec) in enumerate(MODEL_SPECS.items()):
        model = model_artifact["models"][name]
        features = [model_features(row, spec["features"]) for row in rows]
        probabilities = model.predict_proba(features)[:, 1]
        labels = np.asarray([int(row[spec["outcome"]]) for row in rows], dtype=int)
        result = _metrics(labels, probabilities, sklearn_metrics)
        result["trajectory_cluster_bootstrap"] = _bootstrap_metrics(
            rows,
            labels,
            probabilities,
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
            np=np,
            sklearn_metrics=sklearn_metrics,
        )
        results[name] = result
        predictions[name] = probabilities

    conventional = predictions["conventional_vulnerability"]
    residual = predictions["local_residual"]
    count = max(1, (len(rows) + 4) // 5)
    conventional_top = set(conventional.argsort()[::-1][:count].tolist())
    residual_top = set(residual.argsort()[::-1][:count].tolist())
    silent_labels = np.asarray([int(row["silent_failure"]) for row in rows], dtype=int)
    results["conventional_vulnerability"]["silent_failure_ranking"] = _metrics(
        silent_labels, conventional, sklearn_metrics
    )
    ranking_comparison = {
        "spearman_probability_correlation": float(spearmanr(conventional, residual).statistic),
        "highest_fifth_overlap": {
            "intersection": len(conventional_top & residual_top),
            "union": len(conventional_top | residual_top),
            "jaccard": len(conventional_top & residual_top)
            / len(conventional_top | residual_top),
        },
        "interpretation_contract": (
            "a ranking difference is supported only if the frozen residual ranking "
            "both differs from and enriches silent failures better than the conventional ranking"
        ),
    }
    output = {
        "schema_version": 1,
        "analysis": "held-out evaluation of graph-atlas residual-risk rankings",
        "source": source,
        "manifest": manifest,
        "frozen_development_fit": {
            "path": str(args.frozen.resolve()),
            "sha256": file_sha256(args.frozen),
        },
        "population": {
            "interventions": len(rows),
            "trajectory_clusters": len(trajectory_groups(rows)),
            "sites": len({row["site_id"] for row in rows}),
            "physical_runs": len({row["physical_run"] for row in rows}),
        },
        "rates": rate_table(rows),
        "models": results,
        "ranking_comparison": ranking_comparison,
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
