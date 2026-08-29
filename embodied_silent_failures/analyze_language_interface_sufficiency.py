from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_composition import clustered_bootstrap
from embodied_silent_failures.language_gates import scored_record_index
from embodied_silent_failures.language_interface_sufficiency import (
    LADDERS,
    MODEL_FAMILIES,
    classification_predictions,
    interface_rows,
    per_boundary_regression,
    regression_cluster_bootstrap,
    regression_predictions,
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
            "Test whether archived scalar interfaces retain propagation history."
        )
    )
    parser.add_argument(
        "--campaign-dir",
        action="append",
        dest="campaign_dirs",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--analysis", action="append", dest="analysis_paths", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def _sources(
    campaign_dirs: list[Path], analysis_paths: list[Path]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[Path]]:
    analyses = [load_json(path) for path in analysis_paths]
    by_split = {
        str(document["analysis_split"]): document["records"] for document in analyses
    }
    if set(by_split) != {"development", "holdout"}:
        raise ValueError("interface analysis requires development and holdout")
    score_paths = [
        campaign / "scoring" / "language-safe.json" for campaign in campaign_dirs
    ]
    scores = [load_json(path) for path in score_paths]
    shards = sorted(
        int(document["source_campaign"]["worker_shard"]) for document in scores
    )
    if shards != [0, 1]:
        raise ValueError("interface analysis requires worker shards 0 and 1")
    expected_hashes = {file_sha256(path) for path in score_paths}
    for document, path in zip(analyses, analysis_paths, strict=True):
        if {item["sha256"] for item in document["artifacts"]} != expected_hashes:
            raise ValueError(f"analysis does not cite the supplied SAFE scores: {path}")
    return by_split, scored_record_index(scores), score_paths


def _regression_analysis(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
    result = {}
    predictions = {}
    for offset, family in enumerate(MODEL_FAMILIES):
        metrics, values = regression_predictions(
            development, holdout, family=family, seed=seed + offset * 10
        )
        predictions[family] = values
        result[family] = {
            "holdout": metrics,
            "incremental_history": regression_cluster_bootstrap(
                holdout,
                values,
                samples=bootstrap_samples,
                seed=seed + offset * 10 + 3,
            ),
            "holdout_by_boundary": per_boundary_regression(holdout, values),
        }
    return result, predictions


def _classification_analysis(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
    result = {}
    predictions = {}
    for offset, family in enumerate(MODEL_FAMILIES):
        metrics, values = classification_predictions(
            development, holdout, family=family, seed=seed + offset * 10
        )
        predictions[family] = values
        result[family] = {
            "holdout": metrics,
            "holdout_trajectory_cluster_bootstrap": clustered_bootstrap(
                holdout,
                "command_changed",
                values,
                samples=bootstrap_samples,
                seed=seed + offset * 10 + 3,
                baseline="local",
            ),
        }
    return result, predictions


def _prediction_records(
    tasks: list[
        tuple[
            str,
            list[dict[str, Any]],
            dict[str, dict[str, list[float]]],
            str,
        ]
    ],
) -> list[dict[str, Any]]:
    records = []
    for task, rows, predictions, target in tasks:
        for index, row in enumerate(rows):
            record = {
                "task": task,
                "record_id": row["record_id"],
                "context_id": row["context_id"],
                "task_id": row["task_id"],
                "episode_index": row["episode_index"],
                "source_layer": row["source_layer"],
                "boundary": row["boundary"],
                "target": row[target],
            }
            for family in MODEL_FAMILIES:
                for ladder in LADDERS:
                    record[f"prediction_{family}_{ladder}"] = predictions[family][
                        ladder
                    ][index]
            records.append(record)
    return records


def main() -> None:
    args = _arguments()
    if len(args.campaign_dirs) != 2 or len(args.analysis_paths) != 2:
        raise ValueError("interface analysis requires exactly two campaign shards")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    by_split, score_index, score_paths = _sources(
        args.campaign_dirs, args.analysis_paths
    )
    rows = {
        split: interface_rows(values, score_index) for split, values in by_split.items()
    }
    transition, transition_predictions = _regression_analysis(
        rows["development"]["block_transition"],
        rows["holdout"]["block_transition"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    safe, safe_predictions = _regression_analysis(
        rows["development"]["safe_feature_endpoint"],
        rows["holdout"]["safe_feature_endpoint"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 100,
    )
    command, command_predictions = _classification_analysis(
        rows["development"]["command_change_endpoint"],
        rows["holdout"]["command_change_endpoint"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 200,
    )

    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "analysis": "retrospective scalar-interface sufficiency test",
        "status": (
            "hypothesis-generating: the named holdout and the need for richer "
            "interfaces were known before this comparison was fixed"
        ),
        "question": (
            "Does an archived scalar perturbation summary predict the next interface "
            "as well as the same summary plus its upstream scalar history and retained "
            "pre-fault context?"
        ),
        "interpretation_boundary": (
            "The campaign retained scalar norms, maxima, and changed-element counts at "
            "intermediate language blocks, not perturbation vectors. Failure rejects "
            "that scalar interface; no detectable gain does not prove the full neural "
            "interface is Markov or sufficient."
        ),
        "ladders": {
            "local": "current block summary and the mechanically known boundary",
            "history": (
                "local plus injection summary, source layer, path length, and every "
                "earlier archived scalar block summary"
            ),
            "context": (
                "history plus task identity, rollout phase, and action-token position"
            ),
        },
        "targets": {
            "block_transition": (
                "log normalized perturbation magnitude at the next language block"
            ),
            "safe_feature_endpoint": (
                "log normalized movement across the seven archived action-token "
                "features; this is not SAFE's selected 4,096-value final-token input"
            ),
            "command_change_endpoint": "whether the complete LIBERO command changed",
        },
        "models": {
            "ridge": (
                "standardized L2 ridge/logistic probe with alpha or C equal to 1; "
                "block-specific local slopes and no tuning or class weighting"
            ),
            "extra_trees": (
                "fixed 256-tree nonlinear sensitivity model with minimum leaf size 5, "
                "all features considered per split, and no tuning or class weighting"
            ),
            "selection": "both model families and all three ladders are reported",
        },
        "population": {
            split: {
                name: len(task_rows) for name, task_rows in split_rows.items()
            }
            for split, split_rows in rows.items()
        },
        "results": {
            "block_transition": transition,
            "safe_feature_endpoint": safe,
            "command_change_endpoint": command,
        },
        "provenance": {
            "analysis_code_revision": git_revision(project_root),
            "analysis_code_dirty": git_dirty(project_root),
            "campaigns": [
                {
                    "path": str(path.resolve()),
                    "run_sha256": file_sha256(path / "run.json"),
                    "execution": load_json(path / "run.json")["execution"],
                }
                for path in args.campaign_dirs
            ],
            "analyses": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.analysis_paths
            ],
            "scores": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in score_paths
            ],
        },
    }
    prediction_rows = _prediction_records(
        [
            (
                "block_transition",
                rows["holdout"]["block_transition"],
                transition_predictions,
                "target_log_normalized_l2",
            ),
            (
                "safe_feature_endpoint",
                rows["holdout"]["safe_feature_endpoint"],
                safe_predictions,
                "target_log_normalized_l2",
            ),
            (
                "command_change_endpoint",
                rows["holdout"]["command_change_endpoint"],
                command_predictions,
                "command_changed",
            ),
        ]
    )
    write_json_atomic(args.output, output)
    columns = sorted({key for row in prediction_rows for key in row})
    write_csv_atomic(
        args.records_csv,
        [{key: row.get(key, "") for key in columns} for row in prediction_rows],
    )
    print(
        json.dumps(
            {
                "population": output["population"],
                "block_transition": transition,
                "safe_feature_endpoint": safe,
                "command_change_endpoint": command,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
