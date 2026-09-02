from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_layer_rankings import (
    OUTCOMES,
    PROTECTION_BUDGETS,
    eligible_rows,
    expected_protection,
    layer_rates,
    protection_weights,
    ranking_scores,
    spearman,
)
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen language-layer rankings on untouched holdout data."
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _metric(
    rows: list[dict[str, Any]], outcome: str, scores: dict[int, float]
) -> dict[str, float]:
    import numpy as np
    from sklearn import metrics

    labels = np.asarray([int(row[outcome]) for row in rows], dtype=int)
    probabilities = np.asarray(
        [scores[int(row["layer_index"])] for row in rows], dtype=float
    )
    if len(np.unique(labels)) != 2:
        raise ValueError(f"holdout {outcome} does not contain both classes")
    return {
        "roc_auc": float(metrics.roc_auc_score(labels, probabilities)),
        "average_precision": float(
            metrics.average_precision_score(labels, probabilities)
        ),
        "brier_score": float(metrics.brier_score_loss(labels, probabilities)),
    }


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = sum(bool(row["task_failure"]) for row in rows)
    residuals = sum(bool(row["operational_silent_failure"]) for row in rows)
    if residuals > failures:
        raise ValueError("residual failures cannot exceed task failures")
    return {
        "interventions": len(rows),
        "task_failures": failures,
        "residual_failures": residuals,
        "susceptibility": failures / len(rows),
        "eventual_miss_given_failure": residuals / failures if failures else None,
        "residual_risk": residuals / len(rows),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _intervals(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        name: {
            "valid_samples": len(estimates),
            "bootstrap_95": [
                _percentile(estimates, 0.025),
                _percentile(estimates, 0.975),
            ]
            if estimates
            else None,
        }
        for name, estimates in values.items()
    }


def _bootstrap(
    rows: list[dict[str, Any]],
    vulnerability_scores: dict[int, float],
    residual_scores: dict[int, float],
    vulnerability_weights: dict[int, float],
    residual_weights: dict[int, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["episode_index"]))].append(row)
    clusters = list(grouped.values())
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "susceptibility": [],
        "eventual_miss_given_failure": [],
        "residual_risk": [],
        "residual_auc_advantage_monitor_aware_minus_conventional": [],
        "residual_capture_advantage_at_8_layers": [],
    }
    for _ in range(samples):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled = [row for cluster in selected for row in cluster]
        rates = _rates(sampled)
        values["susceptibility"].append(rates["susceptibility"])
        values["residual_risk"].append(rates["residual_risk"])
        if rates["eventual_miss_given_failure"] is not None:
            values["eventual_miss_given_failure"].append(
                rates["eventual_miss_given_failure"]
            )
        labels = [int(row["operational_silent_failure"]) for row in sampled]
        if len(set(labels)) == 2:
            vulnerability = [
                vulnerability_scores[int(row["layer_index"])] for row in sampled
            ]
            residual = [residual_scores[int(row["layer_index"])] for row in sampled]
            values[
                "residual_auc_advantage_monitor_aware_minus_conventional"
            ].append(
                roc_auc_score(labels, residual)
                - roc_auc_score(labels, vulnerability)
            )
        if any(labels):
            vulnerability_capture = expected_protection(
                sampled,
                "operational_silent_failure",
                vulnerability_weights,
            )["expected_capture_fraction"]
            residual_capture = expected_protection(
                sampled, "operational_silent_failure", residual_weights
            )["expected_capture_fraction"]
            values["residual_capture_advantage_at_8_layers"].append(
                residual_capture - vulnerability_capture
            )
    return {
        "requested_samples": samples,
        "trajectory_clusters": len(clusters),
        **_intervals(values),
    }


def main() -> None:
    args = _arguments()
    analysis = load_json(args.analysis)
    frozen = load_json(args.rankings)
    if analysis.get("analysis_split") != "holdout":
        raise ValueError("ranking evaluation requires the untouched holdout analysis")
    if frozen.get("development_only") is not True:
        raise ValueError("rankings were not frozen from development only")
    rows = eligible_rows(analysis)
    rankings = frozen["rankings"]
    if set(rankings) != set(OUTCOMES):
        raise ValueError("frozen artifact has an unexpected ranking set")
    scores = {name: ranking_scores(value) for name, value in rankings.items()}
    evaluations = {}
    for name, ranking in rankings.items():
        evaluations[name] = {
            "metrics": {
                outcome: _metric(rows, outcome, scores[name])
                for outcome in OUTCOMES.values()
            },
            "equal_cost_protection": {
                str(budget): {
                    outcome: expected_protection(
                        rows, outcome, protection_weights(ranking, budget)
                    )
                    for outcome in OUTCOMES.values()
                }
                for budget in PROTECTION_BUDGETS
            },
        }

    primary_budget = int(frozen["evaluation_contract"]["primary_budget_layers"])
    vulnerability_weights = protection_weights(
        rankings["conventional_vulnerability"], primary_budget
    )
    residual_weights = protection_weights(
        rankings["monitor_aware_residual"], primary_budget
    )
    observed_layer_rates = {
        name: layer_rates(rows, outcome) for name, outcome in OUTCOMES.items()
    }
    development = frozen["development_population"]
    development_count = int(development["eligible_interventions"])
    development_failures = int(development["events"]["task_failure"])
    development_residuals = int(
        development["events"]["operational_silent_failure"]
    )
    output = {
        "schema_version": 1,
        "analysis": "untouched holdout evaluation of OpenVLA layer-risk rankings",
        "development_rates": {
            "interventions": development_count,
            "task_failures": development_failures,
            "residual_failures": development_residuals,
            "susceptibility": development_failures / development_count,
            "eventual_miss_given_failure": (
                development_residuals / development_failures
            ),
            "residual_risk": development_residuals / development_count,
        },
        "holdout_rates": _rates(rows),
        "holdout_coverage": analysis["coverage"],
        "rankings": evaluations,
        "ranking_comparison": {
            "development_score_spearman": frozen[
                "development_ranking_comparison"
            ]["spearman_with_average_ties"],
            "holdout_rate_stability": {
                name: spearman(
                    scores[name],
                    {
                        layer: float(value["rate"])
                        for layer, value in observed_layer_rates[name].items()
                    },
                )
                for name in OUTCOMES
            },
            "primary_8_layer_residual_capture": {
                "conventional_vulnerability": expected_protection(
                    rows,
                    "operational_silent_failure",
                    vulnerability_weights,
                ),
                "monitor_aware_residual": expected_protection(
                    rows, "operational_silent_failure", residual_weights
                ),
            },
        },
        "trajectory_cluster_bootstrap": _bootstrap(
            rows,
            scores["conventional_vulnerability"],
            scores["monitor_aware_residual"],
            vulnerability_weights,
            residual_weights,
            samples=int(frozen["evaluation_contract"]["bootstrap_samples"]),
            seed=int(frozen["evaluation_contract"]["bootstrap_seed"]),
        ),
        "artifacts": {
            "holdout_analysis": {
                "path": str(args.analysis.resolve()),
                "sha256": file_sha256(args.analysis),
            },
            "frozen_rankings": {
                "path": str(args.rankings.resolve()),
                "sha256": file_sha256(args.rankings),
            },
        },
        "interpretation_boundary": frozen["evaluation_contract"]["interpretation"],
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
