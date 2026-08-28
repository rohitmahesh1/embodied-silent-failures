from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_gates import (
    COMMAND_SIGNAL_DESCRIPTIONS,
    COMMAND_SIGNALS,
    branch_summary,
    command_signal_auc,
    physical_command_branches,
    scored_record_index,
)
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose language-block residual risk into observable causal gates."
        )
    )
    parser.add_argument(
        "--analysis", action="append", dest="analysis_paths", required=True, type=Path
    )
    parser.add_argument(
        "--scores", action="append", dest="score_paths", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--branches-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def _validate_sources(
    analyses: list[dict[str, Any]],
    analysis_paths: list[Path],
    scores: list[dict[str, Any]],
    score_paths: list[Path],
) -> None:
    splits = [str(document.get("analysis_split")) for document in analyses]
    if sorted(splits) != ["development", "holdout"]:
        raise ValueError(
            "gate analysis requires one development and one holdout analysis"
        )
    if len(scores) != 2:
        raise ValueError("gate analysis requires the two worker score documents")
    shards = [int(document["source_campaign"]["worker_shard"]) for document in scores]
    if sorted(shards) != [0, 1]:
        raise ValueError("gate analysis requires worker shards 0 and 1")
    monitor_hashes = {document["monitor"]["checkpoint_sha256"] for document in scores}
    if len(monitor_hashes) != 1:
        raise ValueError("worker score documents used different SAFE checkpoints")

    score_hashes = {file_sha256(path) for path in score_paths}
    for document, path in zip(analyses, analysis_paths, strict=True):
        declared = {artifact["sha256"] for artifact in document["artifacts"]}
        if declared != score_hashes:
            raise ValueError(f"analysis does not cite the supplied score files: {path}")


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
    all_rows = [row for rows in by_split.values() for row in rows]
    branches = physical_command_branches(all_rows, score_index)
    split_branches = {
        split: [branch for branch in branches if branch["analysis_split"] == split]
        for split in sorted(by_split)
    }

    output = {
        "schema_version": 1,
        "analysis": (
            "exploratory three-gate decomposition of language-block residual risk"
        ),
        "status": "hypothesis-generating; the prior holdout has already been opened",
        "estimand": (
            "Command-survival metrics use a uniform language-block intervention. "
            "Physical consequence and monitor-observability metrics use one distinct "
            "executed command "
            "at one restored context."
        ),
        "gate_hypothesis": [
            "The internal perturbation must survive into a different executed command.",
            "The signed command must cross a state-dependent task-recovery boundary.",
            "The resulting failed trajectory must remain below SAFE's alarm boundary.",
        ],
        "uncertainty": (
            "Command-survival intervals resample whole task/episode trajectories. "
            "Command magnitude fifths are descriptive equal-count groups ordered by "
            "exact magnitude "
            "and physical-run ID; no independent confirmatory split remains."
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
        "command_survival": {
            split: {
                signal: command_signal_auc(
                    rows,
                    signal,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + split_index * 100 + signal_index,
                )
                for signal_index, signal in enumerate(COMMAND_SIGNALS)
            }
            for split_index, (split, rows) in enumerate(sorted(by_split.items()))
        },
        "command_signal_descriptions": COMMAND_SIGNAL_DESCRIPTIONS,
        "physical_consequence_and_observability": {
            "combined": branch_summary(branches),
            **{
                split: branch_summary(values)
                for split, values in split_branches.items()
            },
        },
    }
    write_json_atomic(args.output, output)
    columns = sorted({key for branch in branches for key in branch})
    write_csv_atomic(
        args.branches_csv,
        [{key: branch.get(key, "") for key in columns} for branch in branches],
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
