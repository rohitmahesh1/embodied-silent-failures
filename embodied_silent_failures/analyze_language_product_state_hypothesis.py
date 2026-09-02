from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    write_csv_atomic,
    write_json_atomic,
)
from embodied_silent_failures.language_product_state_hypothesis import (
    FEATURE_STAGES,
    OUTCOMES,
    STATE_DESCRIPTIONS,
    eligible_rows,
    feature_matrix,
    feature_names,
    load_product_state,
    state_widths,
)
from embodied_silent_failures.language_product_state_models import (
    MODEL_FAMILIES,
    alias_audit,
    alias_weights,
    binary_metrics,
    clustered_differences,
    fit_predict,
)
from embodied_silent_failures.provenance import git_dirty, git_revision


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether fault provenance adds held-out information after the "
            "policy, physical, and SAFE state produced by the fault is known."
        )
    )
    parser.add_argument(
        "--shard-dir", action="append", dest="shard_dirs", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def _trajectory(row: dict[str, Any]) -> str:
    return f"task{row['task_id']}:episode{row['episode_index']}"


def _physical_counts(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    positive = [row for row in rows if bool(row[outcome])]
    return {
        "rows": len(rows),
        "positive_rows": len(positive),
        "physical_branches": len({row["physical_run"] for row in rows}),
        "positive_physical_branches": len(
            {row["physical_run"] for row in positive}
        ),
        "trajectory_clusters": len({_trajectory(row) for row in rows}),
        "positive_trajectory_clusters": len({_trajectory(row) for row in positive}),
    }


def main() -> None:
    args = _arguments()
    if len(args.shard_dirs) < 1:
        raise ValueError("at least one product-state shard is required")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    project_root = Path(__file__).resolve().parents[1]
    if git_dirty(project_root):
        raise ValueError("commit the analysis implementation before producing results")

    import numpy as np

    all_rows, sources = load_product_state(np, args.shard_dirs)
    widths = state_widths(all_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "product-state conditional provenance test",
        "analysis_revision": git_revision(project_root),
        "question": (
            "Does fault origin or recorded propagation history improve held-out "
            "prediction after conditioning on the immediate policy, physical, and "
            "SAFE state produced by a command-changing fault?"
        ),
        "scope": {
            "population": (
                "causally eligible, command-changing, single-inference t-1 OpenVLA "
                "language-block replacements"
            ),
            "primary_unit": "uniform language-layer intervention",
            "dependence": (
                "exact-command aliases share physical continuations; uncertainty "
                "resamples complete task/episode trajectories"
            ),
            "split": (
                "fit and model selection use the declared development trajectories; "
                "evaluation uses the declared holdout trajectories"
            ),
            "status": (
                "retrospective and hypothesis-generating because aggregate holdout "
                "outcomes were inspected before this analysis was specified"
            ),
            "prediction_time": (
                "immediately after the affected command's first environment step; "
                "no later SAFE scores, rollout length, or terminal evidence are inputs"
            ),
        },
        "feature_ladder": {
            "product": (
                "task, rollout phase, paired clean and faulted commands, physical "
                "state before and after one action, and cumulative SAFE score plus "
                "feature displacement at the fault"
            ),
            "product_and_origin": (
                "product plus the action-token position and a one-hot OpenVLA "
                "source block"
            ),
            "product_origin_and_path": (
                "product and origin plus source injection and final-language-boundary "
                "propagation magnitudes"
            ),
            "observation_product": (
                "LIBERO's canonical robot proprio-state and object-state observations"
            ),
            "simulator_product": "the complete flattened MuJoCo simulator state",
        },
        "models": {
            "linear": (
                "standardized L2 logistic regression selected from four declared "
                "regularization strengths by grouped development log loss"
            ),
            "extra_trees": (
                "400 fixed extremely randomized trees with minimum leaf size 5"
            ),
            "tuned_forest": (
                "random forest selected from four declared depth, feature, and leaf "
                "settings by grouped development log loss"
            ),
        },
        "sources": sources,
        "state_widths": widths,
        "outcomes": {},
    }
    prediction_rows = []

    for outcome_index, outcome in enumerate(OUTCOMES):
        rows = eligible_rows(all_rows, outcome)
        development = [row for row in rows if row["analysis_split"] == "development"]
        holdout = [row for row in rows if row["analysis_split"] == "holdout"]
        if len(development) + len(holdout) != len(rows):
            raise ValueError("product-state rows contain an unknown analysis split")
        development_labels = np.asarray(
            [int(row[outcome]) for row in development], dtype=int
        )
        holdout_labels = np.asarray([int(row[outcome]) for row in holdout], dtype=int)
        if len(np.unique(development_labels)) != 2 or len(
            np.unique(holdout_labels)
        ) != 2:
            raise ValueError(f"{outcome} requires both classes in both splits")
        groups = [_trajectory(row) for row in development]
        holdout_weights = alias_weights(np, holdout)
        outcome_result: dict[str, Any] = {
            "development": _physical_counts(development, outcome),
            "holdout": _physical_counts(holdout, outcome),
            "exact_product_state_alias_audit": {
                "development": alias_audit(development, outcome),
                "holdout": alias_audit(holdout, outcome),
            },
            "descriptions": {},
        }

        for description_index, description in enumerate(STATE_DESCRIPTIONS):
            matrices = {
                stage: (
                    feature_matrix(np, development, description, stage, widths),
                    feature_matrix(np, holdout, description, stage, widths),
                )
                for stage in FEATURE_STAGES
            }
            description_result: dict[str, Any] = {
                "feature_dimensions": {
                    stage: len(feature_names(description, stage, widths))
                    for stage in FEATURE_STAGES
                },
                "models": {},
            }
            for family_index, family in enumerate(MODEL_FAMILIES):
                predictions = {}
                model_records = {}
                for stage_index, stage in enumerate(FEATURE_STAGES):
                    development_matrix, holdout_matrix = matrices[stage]
                    stage_seed = (
                        args.seed
                        + outcome_index * 10_000
                        + description_index * 1_000
                        + family_index * 100
                        + stage_index
                    )
                    values, model = fit_predict(
                        np,
                        family,
                        development_matrix,
                        development_labels,
                        groups,
                        holdout_matrix,
                        stage_seed,
                    )
                    predictions[stage] = values
                    model_records[stage] = {
                        "model": model,
                        "holdout_uniform_layer": binary_metrics(
                            np, holdout_labels, values
                        ),
                        "holdout_equal_physical_branch_sensitivity": binary_metrics(
                            np,
                            holdout_labels,
                            values,
                            sample_weight=holdout_weights,
                        ),
                    }
                    for row, label, probability in zip(
                        holdout, holdout_labels, values, strict=True
                    ):
                        prediction_rows.append(
                            {
                                "outcome": outcome,
                                "state_description": description,
                                "model_family": family,
                                "feature_stage": stage,
                                "record_id": row["record_id"],
                                "context_id": row["context_id"],
                                "task_id": row["task_id"],
                                "episode_index": row["episode_index"],
                                "physical_run": row["physical_run"],
                                "layer_index": row["layer_index"],
                                "label": int(label),
                                "probability": float(probability),
                            }
                        )
                for stage_index, stage in enumerate(FEATURE_STAGES[1:], start=1):
                    increment = clustered_differences(
                        np,
                        holdout,
                        holdout_labels,
                        predictions["product"],
                        predictions[stage],
                        samples=args.bootstrap_samples,
                        seed=(
                            args.seed
                            + outcome_index * 10_000
                            + description_index * 1_000
                            + family_index * 100
                            + stage_index
                        ),
                    )
                    model_records[stage]["increment_over_product"] = increment
                description_result["models"][family] = model_records
            outcome_result["descriptions"][description] = description_result
        result["outcomes"][outcome] = outcome_result

    write_csv_atomic(args.predictions_csv, prediction_rows)
    result["prediction_artifact"] = artifact_record(args.predictions_csv)
    write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
