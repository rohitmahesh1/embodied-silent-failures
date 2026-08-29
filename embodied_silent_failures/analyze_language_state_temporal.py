from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_composition import (
    binary_metrics,
    clustered_bootstrap,
    fit_logistic_model,
    model_probabilities,
    physical_consequence_rows,
)
from embodied_silent_failures.language_gates import scored_record_index
from embodied_silent_failures.language_state_temporal import (
    DISTANCE_MODES,
    NEIGHBOR_COUNTS,
    TEMPORAL_HORIZONS,
    attach_interfaces,
    load_context_states,
    load_physical_score_traces,
    nearest_context_probabilities,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test raw simulator state and early SAFE evidence as residual-risk interfaces."
        )
    )
    parser.add_argument(
        "--campaign-dir", action="append", dest="campaign_dirs", required=True, type=Path
    )
    parser.add_argument(
        "--analysis", action="append", dest="analysis_paths", required=True, type=Path
    )
    parser.add_argument("--previous-composition", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def _source_rows(
    campaign_dirs: list[Path], analysis_paths: list[Path]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    analyses = [load_json(path) for path in analysis_paths]
    by_split = {
        str(document["analysis_split"]): document["records"] for document in analyses
    }
    if set(by_split) != {"development", "holdout"}:
        raise ValueError("state-temporal analysis requires development and holdout")
    score_paths = [campaign / "scoring" / "language-safe.json" for campaign in campaign_dirs]
    scores = [load_json(path) for path in score_paths]
    if sorted(int(value["source_campaign"]["worker_shard"]) for value in scores) != [0, 1]:
        raise ValueError("state-temporal analysis requires worker shards 0 and 1")
    expected_hashes = {file_sha256(path) for path in score_paths}
    for document, path in zip(analyses, analysis_paths, strict=True):
        if {item["sha256"] for item in document["artifacts"]} != expected_hashes:
            raise ValueError(f"language analysis cites different SAFE scores: {path}")
    return by_split, scores


def _state_gate(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    development_predictions = {}
    holdout_predictions = {}
    for mode in DISTANCE_MODES:
        for neighbors in NEIGHBOR_COUNTS:
            name = f"{mode}_k{neighbors}"
            development_predictions[name] = nearest_context_probabilities(
                development,
                development,
                mode=mode,
                neighbors=neighbors,
                leave_target_trajectory_out=True,
            )
            holdout_predictions[name] = nearest_context_probabilities(
                development,
                holdout,
                mode=mode,
                neighbors=neighbors,
            )
    magnitude_model = fit_logistic_model(
        development, "task_failure", (("command_l2", "log1p"),)
    )
    development_predictions["previous_command_magnitude"] = model_probabilities(
        magnitude_model, development
    )
    holdout_predictions["previous_command_magnitude"] = model_probabilities(
        magnitude_model, holdout
    )

    def ablation_difference(
        predictions: dict[str, list[float]], neighbors: int
    ) -> dict[str, Any]:
        command = predictions[f"executed_command_k{neighbors}"]
        state_command = predictions[f"state_and_executed_command_k{neighbors}"]
        differences = [
            abs(left - right)
            for left, right in zip(command, state_command, strict=True)
        ]
        return {
            "rows": len(differences),
            "changed_predictions": sum(value > 0 for value in differences),
            "maximum_absolute_change": max(differences),
        }

    return (
        {
            "development_leave_trajectory_out": {
                name: binary_metrics(development, "task_failure", values)
                for name, values in development_predictions.items()
                if name != "previous_command_magnitude"
            },
            "holdout": {
                name: binary_metrics(holdout, "task_failure", values)
                for name, values in holdout_predictions.items()
            },
            "raw_state_ablation": {
                "development_leave_trajectory_out": {
                    f"k{neighbors}": ablation_difference(
                        development_predictions, neighbors
                    )
                    for neighbors in NEIGHBOR_COUNTS
                },
                "holdout": {
                    f"k{neighbors}": ablation_difference(
                        holdout_predictions, neighbors
                    )
                    for neighbors in NEIGHBOR_COUNTS
                },
            },
            "holdout_trajectory_cluster_bootstrap": clustered_bootstrap(
                holdout,
                "task_failure",
                holdout_predictions,
                samples=bootstrap_samples,
                seed=seed,
                baseline="previous_command_magnitude",
            ),
            "previous_command_magnitude_model": magnitude_model,
        },
        holdout_predictions,
    )


def _monitor_gate(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    development_failures = [row for row in development if row["task_failure"]]
    holdout_failures = [row for row in holdout if row["task_failure"]]
    models = {}
    holdout_failure_predictions = {}
    holdout_all_predictions = {}
    for horizon in TEMPORAL_HORIZONS:
        name = f"margin_{horizon}"
        field = f"monitor_margin_{horizon}"
        model = fit_logistic_model(
            development_failures,
            "operational_silent_failure",
            ((field, "identity"),),
        )
        models[name] = model
        holdout_failure_predictions[name] = model_probabilities(model, holdout_failures)
        holdout_all_predictions[name] = model_probabilities(model, holdout)
    return (
        {
            "development_failed_physical_branches": len(development_failures),
            "holdout_failed_physical_branches": len(holdout_failures),
            "holdout": {
                name: binary_metrics(
                    holdout_failures, "operational_silent_failure", values
                )
                for name, values in holdout_failure_predictions.items()
            },
            "holdout_trajectory_cluster_bootstrap": clustered_bootstrap(
                holdout_failures,
                "operational_silent_failure",
                holdout_failure_predictions,
                samples=bootstrap_samples,
                seed=seed,
                baseline="margin_0",
            ),
        },
        models,
        holdout_all_predictions,
    )


def _composition_gate(
    holdout: list[dict[str, Any]],
    state_predictions: dict[str, list[float]],
    monitor_predictions: dict[str, list[float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    definitions = {
        "previous_magnitude_immediate": (
            "previous_command_magnitude",
            "margin_0",
        ),
        "executed_command_immediate": ("executed_command_k5", "margin_0"),
        "state_command_immediate": ("state_and_executed_command_k5", "margin_0"),
        "executed_command_temporal25": ("executed_command_k5", "margin_25"),
        "state_command_temporal25": (
            "state_and_executed_command_k5",
            "margin_25",
        ),
        "state_command_temporal25_k1": (
            "state_and_executed_command_k1",
            "margin_25",
        ),
        "state_command_temporal25_k3": (
            "state_and_executed_command_k3",
            "margin_25",
        ),
    }
    predictions = {
        name: [
            consequence * monitor
            for consequence, monitor in zip(
                state_predictions[state_name],
                monitor_predictions[monitor_name],
                strict=True,
            )
        ]
        for name, (state_name, monitor_name) in definitions.items()
    }
    return (
        {
            "definitions": {
                name: {"consequence": values[0], "monitor_miss": values[1]}
                for name, values in definitions.items()
            },
            "holdout": {
                name: binary_metrics(
                    holdout, "operational_silent_failure", values
                )
                for name, values in predictions.items()
            },
            "holdout_trajectory_cluster_bootstrap": clustered_bootstrap(
                holdout,
                "operational_silent_failure",
                predictions,
                samples=bootstrap_samples,
                seed=seed,
                baseline="previous_magnitude_immediate",
            ),
        },
        predictions,
    )


def main() -> None:
    args = _arguments()
    if len(args.campaign_dirs) != 2 or len(args.analysis_paths) != 2:
        raise ValueError("state-temporal analysis requires exactly two campaign shards")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    import numpy as np

    by_split, scores = _source_rows(args.campaign_dirs, args.analysis_paths)
    score_index = scored_record_index(scores)
    states, state_audit = load_context_states(args.campaign_dirs, np)
    traces, trace_audit = load_physical_score_traces(args.campaign_dirs, np)
    branches = {
        split: attach_interfaces(
            physical_consequence_rows(rows, score_index), states, traces
        )
        for split, rows in by_split.items()
    }
    state_gate, state_predictions = _state_gate(
        branches["development"],
        branches["holdout"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    monitor_gate, monitor_models, monitor_predictions = _monitor_gate(
        branches["development"],
        branches["holdout"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    composition_gate, composition_predictions = _composition_gate(
        branches["holdout"],
        state_predictions,
        monitor_predictions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 2,
    )

    previous = None
    if args.previous_composition is not None:
        document = load_json(args.previous_composition)
        previous = {
            "path": str(args.previous_composition.resolve()),
            "sha256": file_sha256(args.previous_composition),
            "analysis_unit": (
                "changed layer intervention; not directly interchangeable with the "
                "physical-command unit used for post-fault temporal evidence"
            ),
            "holdout_evaluation": document["retrospective_holdout_evaluation"],
        }

    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "analysis": "raw-state and temporal-evidence residual-risk audit",
        "status": (
            "hypothesis-generating: the representation and analysis were fixed after "
            "aggregate outcomes from this sample were known"
        ),
        "units": {
            "consequence": (
                "one distinct executed command from one exactly restored simulator state"
            ),
            "monitor_and_residual": (
                "one physical command continuation; post-fault SAFE evidence is shared "
                "by exact-command layer aliases"
            ),
        },
        "design": {
            "state": (
                "Within-task standardized distance over every raw flattened MuJoCo "
                "state coordinate. State and command blocks receive equal distance "
                "weight; no object names or task-specific semantic features are chosen."
            ),
            "neighbors": (
                "One nearest command per restored context prevents contexts with more "
                "exact-command groups from receiving more votes. k=1,3,5 are all reported."
            ),
            "temporal_monitor": (
                "Maximum SAFE score-to-band ratio over its pre-existing 0,5,10,25-step "
                "windows, represented as distance below the frozen alarm band."
            ),
            "composition": (
                "P(physical silent failure) = predicted task-consequence probability "
                "times predicted eventual SAFE-miss probability."
            ),
        },
        "provenance": {
            "experiment_code_revision": git_revision(project_root),
            "experiment_code_dirty": git_dirty(project_root),
            "campaigns": [
                {
                    "path": str(path.resolve()),
                    "run_sha256": file_sha256(path / "run.json"),
                }
                for path in args.campaign_dirs
            ],
            "analyses": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.analysis_paths
            ],
        },
        "audit": {"captured_states": state_audit, "physical_safe_traces": trace_audit},
        "population": {
            split: {
                "changed_physical_commands": len(rows),
                "task_failures": sum(row["task_failure"] for row in rows),
                "silent_failures": sum(
                    row["operational_silent_failure"] for row in rows
                ),
                "trajectory_clusters": len(
                    {(row["task_id"], row["episode_index"]) for row in rows}
                ),
            }
            for split, rows in branches.items()
        },
        "state_consequence_gate": state_gate,
        "temporal_monitor_gate": {**monitor_gate, "development_models": monitor_models},
        "composed_residual_risk": composition_gate,
        "previous_composition": previous,
    }
    write_json_atomic(args.output, output)
    record_rows = []
    for index, row in enumerate(branches["holdout"]):
        record = {
            key: value
            for key, value in row.items()
            if key not in {"state", "delta_command", "executed_command"}
        }
        for name, values in state_predictions.items():
            record[f"consequence_{name}"] = values[index]
        for name, values in monitor_predictions.items():
            record[f"monitor_{name}"] = values[index]
        for name, values in composition_predictions.items():
            record[f"residual_{name}"] = values[index]
        record_rows.append(record)
    columns = sorted({key for row in record_rows for key in row})
    write_csv_atomic(
        args.records_csv,
        [{key: row.get(key, "") for key in columns} for row in record_rows],
    )
    print(
        json.dumps(
            {
                "population": output["population"],
                "state_holdout": state_gate["holdout"],
                "monitor_holdout": monitor_gate["holdout"],
                "composition_holdout": composition_gate["holdout"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
