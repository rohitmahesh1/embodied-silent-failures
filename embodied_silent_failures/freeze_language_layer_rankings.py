from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_layer_rankings import freeze_rankings
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze language-layer vulnerability and residual-risk rankings."
    )
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    analysis = load_json(args.analysis)
    frozen = freeze_rankings(analysis)
    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "analysis": "frozen OpenVLA language-layer risk rankings",
        "development_only": True,
        "estimand": (
            "Uniform choice of one of OpenVLA's 32 language blocks at one sampled "
            "context, conditional on a successful current control."
        ),
        "detection_rule": (
            "A residual event is a causal task failure with no SAFE alpha=0.1 alarm "
            "before the fault or at any later point in the complete physical trace."
        ),
        "ranking_rule": (
            "Score each layer by its raw development event rate. Equal denominators "
            "make smoothing unnecessary; ROC metrics retain ties."
        ),
        "protection_rule": (
            "At budgets of 4, 8, and 16 equal-cost layers, protect every higher-score "
            "layer and uniformly randomize within a score tie at the budget boundary. "
            "Reported capture is the expectation over that randomization."
        ),
        "evaluation_contract": {
            "primary_budget_layers": 8,
            "metrics": [
                "holdout susceptibility, conditional eventual miss, and residual risk",
                "ROC-AUC and average precision against task failure and "
                "residual events",
                "expected event capture at equal layer budgets",
                "paired trajectory-cluster bootstrap differences between rankings",
            ],
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 20260902,
            "interpretation": (
                "Equal-cost perfect protection is a screening abstraction. It does not "
                "claim equal physical exposure, equal implementation cost, or hardware "
                "fault prevalence."
            ),
        },
        "source_analysis": {
            "path": str(args.analysis.resolve()),
            "sha256": file_sha256(args.analysis),
        },
        "implementation": {
            "experiment_code_revision": git_revision(project_root),
            "experiment_code_dirty": git_dirty(project_root),
        },
        **frozen,
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
