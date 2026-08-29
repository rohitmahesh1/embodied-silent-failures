from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_composition import (
    add_context_indicators,
    binary_metrics,
    clustered_bootstrap,
    fit_logistic_model,
    intervention_composition_rows,
    model_probabilities,
    model_specifications,
    physical_consequence_rows,
)
from embodied_silent_failures.language_gates import (
    COMMAND_SIGNALS,
    command_signal_auc,
    scored_record_index,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a frozen three-gate description on exact-state branches."
    )
    parser.add_argument(
        "--analysis", action="append", dest="analysis_paths", required=True, type=Path
    )
    parser.add_argument(
        "--scores", action="append", dest="score_paths", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def _validate_sources(
    analyses: list[dict[str, Any]],
    analysis_paths: list[Path],
    scores: list[dict[str, Any]],
    score_paths: list[Path],
) -> None:
    if sorted(str(document.get("analysis_split")) for document in analyses) != [
        "development",
        "holdout",
    ]:
        raise ValueError("composition analysis requires development and holdout")
    shards = sorted(
        int(document["source_campaign"]["worker_shard"]) for document in scores
    )
    if shards != [0, 1]:
        raise ValueError("composition analysis requires worker shards 0 and 1")
    score_hashes = {file_sha256(path) for path in score_paths}
    for document, path in zip(analyses, analysis_paths, strict=True):
        if {artifact["sha256"] for artifact in document["artifacts"]} != score_hashes:
            raise ValueError(f"analysis does not cite the supplied scores: {path}")


def _fit_models(
    development_branches: list[dict[str, Any]],
    development_interventions: list[dict[str, Any]],
    specifications: dict[str, Any],
) -> dict[str, Any]:
    consequence = {
        name: fit_logistic_model(
            development_branches, "task_failure", specification
        )
        for name, specification in specifications["consequence"].items()
    }
    failed = [row for row in development_interventions if row["task_failure"]]
    monitor = {
        name: fit_logistic_model(failed, "monitor_missed", specification)
        for name, specification in specifications["monitor"].items()
    }
    direct = {
        name: fit_logistic_model(
            development_interventions,
            "operational_silent_failure",
            specification,
        )
        for name, specification in specifications["direct"].items()
    }
    return {"consequence": consequence, "monitor": monitor, "direct": direct}


def _evaluate_models(
    models: dict[str, Any],
    specifications: dict[str, Any],
    holdout_branches: list[dict[str, Any]],
    holdout_interventions: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    holdout_failures = [
        row for row in holdout_interventions if row["task_failure"]
    ]
    consequence_predictions = {
        name: model_probabilities(model, holdout_branches)
        for name, model in models["consequence"].items()
    }
    consequence_predictions_on_interventions = {
        name: model_probabilities(model, holdout_interventions)
        for name, model in models["consequence"].items()
    }
    monitor_predictions_on_failures = {
        name: model_probabilities(model, holdout_failures)
        for name, model in models["monitor"].items()
    }
    monitor_predictions = {
        name: model_probabilities(model, holdout_interventions)
        for name, model in models["monitor"].items()
    }
    direct_predictions = {
        name: model_probabilities(model, holdout_interventions)
        for name, model in models["direct"].items()
    }
    composition_predictions = {}
    for name, (consequence_name, monitor_name) in specifications[
        "compositions"
    ].items():
        composition_predictions[name] = [
            consequence * miss
            for consequence, miss in zip(
                consequence_predictions_on_interventions[consequence_name],
                monitor_predictions[monitor_name],
                strict=True,
            )
        ]

    consequence_results = {
        name: binary_metrics(holdout_branches, "task_failure", values)
        for name, values in consequence_predictions.items()
    }
    monitor_results = {
        name: binary_metrics(holdout_failures, "monitor_missed", values)
        for name, values in monitor_predictions_on_failures.items()
    }
    residual_predictions = {
        "command_magnitude": direct_predictions["command_magnitude"],
        "direct_compact": direct_predictions["compact"],
        "direct_same_inputs": direct_predictions["same_inputs"],
        **{
            f"composed_{name}": values
            for name, values in composition_predictions.items()
        },
    }
    residual_results = {
        name: binary_metrics(
            holdout_interventions, "operational_silent_failure", values
        )
        for name, values in residual_predictions.items()
    }
    bootstrap = {
        "consequence_gate": clustered_bootstrap(
            holdout_branches,
            "task_failure",
            consequence_predictions,
            samples=bootstrap_samples,
            seed=seed,
            baseline="magnitude",
        ),
        "monitor_gate": clustered_bootstrap(
            holdout_failures,
            "monitor_missed",
            monitor_predictions_on_failures,
            samples=bootstrap_samples,
            seed=seed + 1,
            baseline="fault_margin",
        ),
        "residual_risk": clustered_bootstrap(
            holdout_interventions,
            "operational_silent_failure",
            residual_predictions,
            samples=bootstrap_samples,
            seed=seed + 2,
            baseline="command_magnitude",
        ),
    }

    record_rows = []
    for index, row in enumerate(holdout_interventions):
        record = {
            "record_id": row["record_id"],
            "layer_index": row["layer_index"],
            "physical_run": row["physical_run"],
            "context_id": row["context_id"],
            "task_id": row["task_id"],
            "episode_index": row["episode_index"],
            "task_failure": row["task_failure"],
            "monitor_missed": row["monitor_missed"],
            "operational_silent_failure": row["operational_silent_failure"],
        }
        for name, values in consequence_predictions_on_interventions.items():
            record[f"consequence_{name}"] = values[index]
        for name, values in monitor_predictions.items():
            record[f"monitor_{name}"] = values[index]
        for name, values in residual_predictions.items():
            record[f"residual_{name}"] = values[index]
        record_rows.append(record)
    return (
        {
            "consequence_gate": consequence_results,
            "monitor_gate": monitor_results,
            "residual_risk": residual_results,
            "trajectory_cluster_bootstrap": bootstrap,
        },
        record_rows,
    )


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    analyses = [load_json(path) for path in args.analysis_paths]
    scores = [load_json(path) for path in args.score_paths]
    _validate_sources(analyses, args.analysis_paths, scores, args.score_paths)
    by_split = {
        str(document["analysis_split"]): document["records"] for document in analyses
    }
    score_index = scored_record_index(scores)
    branches = {
        split: physical_consequence_rows(rows, score_index)
        for split, rows in by_split.items()
    }
    interventions = {
        split: intervention_composition_rows(rows, score_index)
        for split, rows in by_split.items()
    }
    task_ids = tuple(sorted({int(row["task_id"]) for row in branches["development"]}))
    holdout_task_ids = tuple(
        sorted({int(row["task_id"]) for row in branches["holdout"]})
    )
    for rows in branches.values():
        add_context_indicators(rows, task_ids)
    for rows in interventions.values():
        add_context_indicators(rows, task_ids)
    specifications = model_specifications(task_ids)
    models = _fit_models(
        branches["development"], interventions["development"], specifications
    )
    evaluation, record_rows = _evaluate_models(
        models,
        specifications,
        branches["holdout"],
        interventions["holdout"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    command_screen = {
        split: {
            "eligible_interventions": len(
                [row for row in rows if row.get("eligible_causal_outcome")]
            ),
            "changed_command_interventions": sum(
                bool(row.get("command_changed"))
                for row in rows
                if row.get("eligible_causal_outcome")
            ),
            "silent_failures_with_unchanged_command": sum(
                bool(row.get("operational_silent_failure"))
                for row in rows
                if row.get("eligible_causal_outcome")
                and not row.get("command_changed")
            ),
            "raw_signal_auc": {
                signal: command_signal_auc(
                    rows,
                    signal,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + signal_index,
                )
                for signal_index, signal in enumerate(COMMAND_SIGNALS)
            },
        }
        for split, rows in by_split.items()
    }

    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "analysis": "retrospective test of a three-gate residual-risk description",
        "status": (
            "hypothesis-generating: aggregate outcomes from the named holdout were "
            "inspected before this predictor ladder was frozen"
        ),
        "estimands": {
            "command_screen": (
                "Uniform language-block intervention at a successful-control context; "
                "exactly unchanged commands are resolved by the local forward."
            ),
            "physical_branch_ranking": (
                "The task-consequence gate uses one distinct changed command at one "
                "directly restored simulator state. Monitor visibility and final "
                "residual risk use a uniform changed language-block intervention "
                "because distinct sites can produce the same command but different "
                "SAFE evidence."
            ),
        },
        "interpretation_boundary": (
            "Task identity, rollout phase, and the clean command are coarse retained "
            "context. They do not reconstruct or represent the full MuJoCo state."
        ),
        "composition_rule": (
            "P(silent failure | changed command) = P(task failure | command, retained "
            "context) * P(SAFE miss | task failure, injection-time monitor evidence)."
        ),
        "selection_policy": (
            "All predeclared consequence, monitor, direct, and composed descriptions "
            "are reported; this analysis does not choose a model from holdout "
            "performance."
        ),
        "artifacts": {
            "analyses": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.analysis_paths
            ],
            "scores": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.score_paths
            ],
        },
        "implementation": {
            "experiment_code_revision": git_revision(project_root),
            "experiment_code_dirty": git_dirty(project_root),
            "logistic_regression": (
                "scikit-learn L2 logistic regression, C=1, no class weighting or tuning"
            ),
            "task_ids_fitted_from_development": list(task_ids),
            "holdout_task_ids_absent_from_development": sorted(
                set(holdout_task_ids) - set(task_ids)
            ),
        },
        "populations": {
            split: {
                "physical_changed_command_branches": len(rows),
                "physical_task_failures": sum(
                    bool(row["task_failure"]) for row in rows
                ),
                "physical_silent_failures": sum(
                    bool(row["operational_silent_failure"]) for row in rows
                ),
                "changed_layer_interventions": len(interventions[split]),
                "layer_task_failures": sum(
                    bool(row["task_failure"]) for row in interventions[split]
                ),
                "layer_silent_failures": sum(
                    bool(row["operational_silent_failure"])
                    for row in interventions[split]
                ),
                "maximum_within_command_score_spread": max(
                    float(row["member_score_spread"]) for row in rows
                ),
            }
            for split, rows in branches.items()
        },
        "command_screen": command_screen,
        "model_specifications": specifications,
        "frozen_development_models": models,
        "retrospective_holdout_evaluation": evaluation,
    }
    write_json_atomic(args.output, output)
    write_csv_atomic(args.records_csv, record_rows)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
