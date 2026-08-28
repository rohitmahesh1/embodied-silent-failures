from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_gates import (
    clustered_signal_auc,
    command_signal_auc,
)
from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.safe_directions import (
    collapse_physical_failures,
    direction_group_summary,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the two SAFE directional-response worker artifacts."
    )
    parser.add_argument(
        "--directions",
        action="append",
        dest="direction_paths",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    documents = [load_json(path) for path in args.direction_paths]
    shards = [
        int(document["source_campaign"]["worker_shard"])
        for document in documents
    ]
    if sorted(shards) != [0, 1]:
        raise ValueError("SAFE direction summary requires worker shards 0 and 1")
    checkpoints = {
        document["provenance"]["monitor"]["checkpoint_sha256"]
        for document in documents
    }
    if len(checkpoints) != 1:
        raise ValueError("SAFE direction workers used different checkpoints")

    records = [record for document in documents for record in document["records"]]
    record_ids = [str(record["record_id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("SAFE direction workers contain duplicate interventions")
    eligible = [record for record in records if record["eligible_causal_outcome"]]

    outcome_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        outcome_groups[str(record["outcome_group"])].append(record)
    physical_failures = collapse_physical_failures(eligible)
    physical_outcome_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in physical_failures:
        record["eventually_detected"] = (
            record["outcome_group"] == "detected_failure"
        )
        margin = record.get("threshold_margin_after_fault")
        record["threshold_proximity_after_fault"] = (
            -float(margin) if margin is not None else None
        )
        physical_outcome_groups[str(record["outcome_group"])].append(record)
    physical_split_groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: defaultdict(list) for split in ("development", "holdout")
    }
    for record in physical_failures:
        physical_split_groups[str(record["analysis_split"])][
            str(record["outcome_group"])
        ].append(record)
    command_groups = {
        "unchanged": [record for record in eligible if not record["command_changed"]],
        "changed": [record for record in eligible if record["command_changed"]],
    }
    by_split = {
        split: [record for record in eligible if record["analysis_split"] == split]
        for split in ("development", "holdout")
    }
    physical_by_split = {
        split: [
            record
            for record in physical_failures
            if record["analysis_split"] == split
        ]
        for split in ("development", "holdout")
    }
    detection_signals = (
        "threshold_proximity_after_fault",
        "absolute_monitor_increment_delta",
        "monitor_secant_sensitivity",
        "selected_feature_normalized_l2",
    )

    output = {
        "schema_version": 1,
        "analysis": "combined frozen SAFE directional-response summary",
        "status": "hypothesis-generating; the prior holdout has already been opened",
        "method": documents[0]["method"],
        "artifacts": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in args.direction_paths
        ],
        "coverage": {
            "direction_records": len(records),
            "eligible_causal_interventions": len(eligible),
            "trajectory_clusters": len(
                {(record["task_id"], record["episode_index"]) for record in eligible}
            ),
        },
        "selected_monitor_input_predicts_command_change": {
            split: command_signal_auc(
                split_records,
                "selected_feature_normalized_l2",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + index,
            )
            for index, (split, split_records) in enumerate(by_split.items())
        },
        "by_command_survival": {
            name: direction_group_summary(values)
            for name, values in command_groups.items()
        },
        "by_terminal_outcome": {
            name: direction_group_summary(values)
            for name, values in sorted(outcome_groups.items())
        },
        "by_unique_physical_failure": {
            "method": (
                "Take the median across layer interventions that produced the same "
                "executed command and shared one physical rollout, then summarize "
                "those unique branches."
            ),
            "physical_failure_branches": len(physical_failures),
            "outcomes": {
                name: direction_group_summary(values)
                for name, values in sorted(physical_outcome_groups.items())
            },
            "splits": {
                split: {
                    name: direction_group_summary(values)
                    for name, values in sorted(groups.items())
                }
                for split, groups in physical_split_groups.items()
            },
            "signals_for_eventual_detection": {
                split: {
                    signal: clustered_signal_auc(
                        values,
                        signal,
                        "eventually_detected",
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed + 1_000 + split_index * 100 + signal_index,
                    )
                    for signal_index, signal in enumerate(detection_signals)
                }
                for split_index, (split, values) in enumerate(
                    physical_by_split.items()
                )
            },
        },
        "worker_score_reconstruction": {
            str(document["source_campaign"]["worker_shard"]): document["coverage"]
            for document in documents
        },
    }
    write_json_atomic(args.output, output)
    columns = sorted({key for record in records for key in record})
    write_csv_atomic(
        args.records_csv,
        [{key: record.get(key, "") for key in columns} for record in records],
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
