from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


LANGUAGE_BLOCK_COUNT = 32
OUTCOMES = {
    "conventional_vulnerability": "task_failure",
    "monitor_aware_residual": "operational_silent_failure",
}
PROTECTION_BUDGETS = (4, 8, 16)


def eligible_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row for row in analysis["records"] if row.get("eligible_causal_outcome")
    ]
    if not rows:
        raise ValueError("analysis contains no eligible causal outcomes")
    expected_layers = set(range(LANGUAGE_BLOCK_COUNT))
    observed_layers = {int(row["layer_index"]) for row in rows}
    if observed_layers != expected_layers:
        raise ValueError(
            "eligible rows do not cover the complete 32-block language population"
        )
    for row in rows:
        for outcome in OUTCOMES.values():
            if not isinstance(row.get(outcome), bool):
                raise ValueError(
                    f"eligible row {row.get('record_id')} has no Boolean {outcome}"
                )
    return rows


def layer_rates(rows: list[dict[str, Any]], outcome: str) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer_index"])].append(row)
    if set(grouped) != set(range(LANGUAGE_BLOCK_COUNT)):
        raise ValueError("layer-rate population is incomplete")
    result = {}
    for layer, values in sorted(grouped.items()):
        events = sum(bool(row[outcome]) for row in values)
        result[layer] = {
            "events": events,
            "interventions": len(values),
            "rate": events / len(values),
        }
    return result


def average_ranks(scores: dict[int, float]) -> dict[int, float]:
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    result: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2
        for layer, _score in ordered[start:end]:
            result[layer] = rank
        start = end
    return result


def spearman(scores_a: dict[int, float], scores_b: dict[int, float]) -> float | None:
    if set(scores_a) != set(scores_b):
        raise ValueError("rankings cover different layer populations")
    ranks_a = average_ranks(scores_a)
    ranks_b = average_ranks(scores_b)
    layers = sorted(scores_a)
    values_a = [ranks_a[layer] for layer in layers]
    values_b = [ranks_b[layer] for layer in layers]
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    covariance = sum(
        (left - mean_a) * (right - mean_b)
        for left, right in zip(values_a, values_b, strict=True)
    )
    variance_a = sum((value - mean_a) ** 2 for value in values_a)
    variance_b = sum((value - mean_b) ** 2 for value in values_b)
    if variance_a == 0 or variance_b == 0:
        return None
    return covariance / math.sqrt(variance_a * variance_b)


def tie_aware_protection_weights(
    scores: dict[int, float], budget: int
) -> dict[int, float]:
    if set(scores) != set(range(LANGUAGE_BLOCK_COUNT)):
        raise ValueError("protection ranking must cover all 32 language blocks")
    if not 0 <= budget <= LANGUAGE_BLOCK_COUNT:
        raise ValueError("protection budget must be between 0 and 32 layers")
    weights = {layer: 0.0 for layer in scores}
    remaining = budget
    for score in sorted(set(scores.values()), reverse=True):
        tied = sorted(layer for layer, value in scores.items() if value == score)
        if remaining <= 0:
            break
        if remaining >= len(tied):
            for layer in tied:
                weights[layer] = 1.0
            remaining -= len(tied)
        else:
            probability = remaining / len(tied)
            for layer in tied:
                weights[layer] = probability
            remaining = 0
    if not math.isclose(sum(weights.values()), budget, abs_tol=1e-12):
        raise RuntimeError("tie-aware policy did not spend its complete layer budget")
    return weights


def freeze_rankings(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("analysis_split") != "development":
        raise ValueError("layer rankings must be frozen from development data")
    rows = eligible_rows(analysis)
    rankings = {}
    for name, outcome in OUTCOMES.items():
        rates = layer_rates(rows, outcome)
        scores = {layer: float(value["rate"]) for layer, value in rates.items()}
        rankings[name] = {
            "outcome": outcome,
            "score": "raw development event rate within language block",
            "layers": {str(layer): value for layer, value in rates.items()},
            "protection_policies": {
                str(budget): {
                    "layer_budget": budget,
                    "weights": {
                        str(layer): weight
                        for layer, weight in tie_aware_protection_weights(
                            scores, budget
                        ).items()
                    },
                }
                for budget in PROTECTION_BUDGETS
            },
        }
    vulnerability_scores = ranking_scores(rankings["conventional_vulnerability"])
    residual_scores = ranking_scores(rankings["monitor_aware_residual"])
    vulnerability_ranks = average_ranks(vulnerability_scores)
    residual_ranks = average_ranks(residual_scores)
    return {
        "development_population": {
            "eligible_interventions": len(rows),
            "trajectory_clusters": len(
                {(int(row["task_id"]), int(row["episode_index"])) for row in rows}
            ),
            "events": {
                outcome: sum(bool(row[outcome]) for row in rows)
                for outcome in OUTCOMES.values()
            },
        },
        "rankings": rankings,
        "development_ranking_comparison": {
            "spearman_with_average_ties": spearman(
                vulnerability_scores, residual_scores
            ),
            "layers_with_different_rates": sum(
                vulnerability_scores[layer] != residual_scores[layer]
                for layer in vulnerability_scores
            ),
            "layers_with_different_average_rank": sum(
                vulnerability_ranks[layer] != residual_ranks[layer]
                for layer in vulnerability_ranks
            ),
            "equal_budget_policy_overlap": {
                str(budget): _policy_overlap(
                    protection_weights(
                        rankings["conventional_vulnerability"], budget
                    ),
                    protection_weights(rankings["monitor_aware_residual"], budget),
                    budget,
                )
                for budget in PROTECTION_BUDGETS
            },
        },
    }


def _policy_overlap(
    left: dict[int, float], right: dict[int, float], budget: int
) -> dict[str, float | int]:
    overlap = sum(min(left[layer], right[layer]) for layer in left)
    return {
        "layer_budget": budget,
        "expected_layer_overlap": overlap,
        "layer_equivalents_reallocated": budget - overlap,
    }


def ranking_scores(ranking: dict[str, Any]) -> dict[int, float]:
    result = {
        int(layer): float(value["rate"])
        for layer, value in ranking["layers"].items()
    }
    if set(result) != set(range(LANGUAGE_BLOCK_COUNT)):
        raise ValueError("frozen ranking does not cover all 32 language blocks")
    return result


def protection_weights(ranking: dict[str, Any], budget: int) -> dict[int, float]:
    policy = ranking["protection_policies"].get(str(budget))
    if not isinstance(policy, dict) or int(policy.get("layer_budget", -1)) != budget:
        raise ValueError(f"frozen ranking has no valid {budget}-layer policy")
    weights = {int(layer): float(value) for layer, value in policy["weights"].items()}
    if set(weights) != set(range(LANGUAGE_BLOCK_COUNT)):
        raise ValueError("frozen protection policy does not cover all language blocks")
    if any(not 0 <= value <= 1 for value in weights.values()):
        raise ValueError("frozen protection weights must be probabilities")
    if not math.isclose(sum(weights.values()), budget, abs_tol=1e-12):
        raise ValueError("frozen protection policy spends the wrong layer budget")
    return weights


def expected_protection(
    rows: list[dict[str, Any]], outcome: str, weights: dict[int, float]
) -> dict[str, Any]:
    events = [row for row in rows if row[outcome]]
    captured = sum(weights[int(row["layer_index"])] for row in events)
    return {
        "events": len(events),
        "expected_events_captured": captured,
        "expected_capture_fraction": captured / len(events) if events else None,
        "expected_events_remaining": len(events) - captured,
    }
