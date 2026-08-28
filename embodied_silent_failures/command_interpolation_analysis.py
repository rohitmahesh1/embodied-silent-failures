from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def branch_boundary_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["physical_run"])].append(record)
    branches = []
    for physical_run, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda record: float(record["interpolation"]))
        outcomes = [bool(record["success"]) for record in ordered]
        lambdas = [float(record["interpolation"]) for record in ordered]
        first_failure = next(
            (
                value
                for value, success in zip(lambdas, outcomes, strict=True)
                if not success
            ),
            None,
        )
        monotone = not any(
            not outcomes[left] and outcomes[right]
            for left in range(len(outcomes))
            for right in range(left + 1, len(outcomes))
        )
        branches.append(
            {
                "physical_run": physical_run,
                "context_id": ordered[0]["context_id"],
                "worker_shard": ordered[0]["worker_shard"],
                "analysis_split": ordered[0]["analysis_split"],
                "lambdas": lambdas,
                "successes": outcomes,
                "first_observed_failure_lambda": first_failure,
                "monotone_success_to_failure": monotone,
                "endpoint_contract_holds": (
                    0.0 in lambdas
                    and 1.0 in lambdas
                    and outcomes[lambdas.index(0.0)]
                    and not outcomes[lambdas.index(1.0)]
                ),
            }
        )
    patterns = Counter(
        "".join("S" if success else "F" for success in branch["successes"])
        for branch in branches
    )
    return {
        "branches": len(branches),
        "monotone_branches": sum(
            branch["monotone_success_to_failure"] for branch in branches
        ),
        "endpoint_contract_branches": sum(
            branch["endpoint_contract_holds"] for branch in branches
        ),
        "outcome_patterns": dict(sorted(patterns.items())),
        "records": branches,
    }
