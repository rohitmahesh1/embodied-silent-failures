from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.action_monitor_analysis import (
    attach_safe_arrays,
    coupling_summary,
    nested_holdout_models,
    rank_mismatch_diagnostic,
)
from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import file_sha256, git_state, load_json
from embodied_silent_failures.safe_trajectory_analysis import (
    binary_metric_summary,
    trajectory_bootstrap_auc,
)


ACTION_METRICS = (
    "same_feature_action_js_at_fault",
    "same_feature_action_logit_l2_at_fault",
    "same_feature_clean_choice_margin_erosion_at_fault",
    "full_action_mean_js_at_fault",
    "generated_action_token_change_fraction_at_fault",
    "executed_command_l2_at_fault",
    "same_feature_action_js_sum",
    "full_action_js_sum",
    "executed_command_l2_energy",
)

POLICY_MEASURES = (
    "same_feature_action_js_at_fault",
    "full_action_mean_js_at_fault",
    "executed_command_l2_at_fault",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether action-relevant final features receive weak SAFE responses "
            "on failed physical continuations."
        )
    )
    parser.add_argument("--action-geometry", action="append", required=True, type=Path)
    parser.add_argument("--action-arrays", action="append", required=True, type=Path)
    parser.add_argument("--safe-geometry", action="append", required=True, type=Path)
    parser.add_argument("--safe-arrays", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def _load_shard(
    np,
    action_path: Path,
    action_array_path: Path,
    safe_path: Path,
    safe_array_path: Path,
) -> tuple[list[dict], dict]:
    action = load_json(action_path)
    safe = load_json(safe_path)
    if file_sha256(action_array_path) != action["array_archive"]["sha256"]:
        raise ValueError(f"action arrays differ from {action_path}")
    if file_sha256(safe_array_path) != safe["array_archive"]["sha256"]:
        raise ValueError(f"SAFE arrays differ from {safe_path}")
    with np.load(action_array_path, allow_pickle=False) as arrays:
        action_runs = arrays["physical_runs"].astype(str).tolist()
    with np.load(safe_array_path, allow_pickle=False) as arrays:
        safe_runs = arrays["physical_runs"].astype(str).tolist()
        safe_arrays = {
            name: arrays[name]
            for name in ("monitor_increment_delta", "selected_feature_l2")
        }
    action_records = {str(row["physical_run"]): row for row in action["records"]}
    safe_index = {run: index for index, run in enumerate(safe_runs)}
    if action_runs != [str(row["physical_run"]) for row in action["records"]]:
        raise ValueError(f"action array order differs from {action_path}")
    missing = sorted(set(action_runs) - set(safe_index))
    if missing:
        raise ValueError(f"SAFE geometry omits {len(missing)} action runs")
    rows = [
        attach_safe_arrays(action_records[run], safe_arrays, safe_index[run])
        for run in action_runs
    ]
    source = {
        "action_geometry": {
            "path": str(action_path.resolve()),
            "sha256": file_sha256(action_path),
        },
        "action_arrays": {
            "path": str(action_array_path.resolve()),
            "sha256": file_sha256(action_array_path),
        },
        "safe_geometry": {
            "path": str(safe_path.resolve()),
            "sha256": file_sha256(safe_path),
        },
        "safe_arrays": {
            "path": str(safe_array_path.resolve()),
            "sha256": file_sha256(safe_array_path),
        },
    }
    return rows, source


def _metric_comparisons(rows: list[dict], samples: int, seed: int) -> dict:
    failed = lambda row: bool(row["policy_failure"])
    succeeded = lambda row: not bool(row["policy_failure"])
    silent = lambda row: row["outcome_group"] == "silent_failure"
    detected = lambda row: row["outcome_group"] == "detected_failure"
    not_silent = lambda row: row["outcome_group"] != "silent_failure"
    return {
        metric: {
            "failure_vs_success": binary_metric_summary(
                rows, metric=metric, positive=failed, negative=succeeded
            ),
            "silent_vs_detected_failure": binary_metric_summary(
                rows, metric=metric, positive=silent, negative=detected
            ),
            "silent_failure_vs_all_other": binary_metric_summary(
                rows, metric=metric, positive=silent, negative=not_silent
            ),
            "failure_vs_success_trajectory_bootstrap": trajectory_bootstrap_auc(
                rows,
                metric=metric,
                positive=failed,
                negative=succeeded,
                samples=samples,
                seed=seed + offset,
            ),
        }
        for offset, metric in enumerate(ACTION_METRICS)
    }


def main() -> None:
    args = _arguments()
    counts = {
        len(args.action_geometry),
        len(args.action_arrays),
        len(args.safe_geometry),
        len(args.safe_arrays),
    }
    if len(counts) != 1:
        raise ValueError("action and SAFE shard counts differ")
    if args.bootstrap_samples < 1:
        raise ValueError("at least one bootstrap sample is required")
    import numpy as np

    rows = []
    sources = []
    seen = set()
    for paths in zip(
        args.action_geometry,
        args.action_arrays,
        args.safe_geometry,
        args.safe_arrays,
        strict=True,
    ):
        shard_rows, source = _load_shard(np, *paths)
        runs = {str(row["physical_run"]) for row in shard_rows}
        overlap = seen.intersection(runs)
        if overlap:
            raise ValueError(f"duplicate physical continuations: {len(overlap)}")
        seen.update(runs)
        rows.extend(shard_rows)
        sources.append(source)

    topology_counts = {
        name: sum(name in row["representative_topologies"] for row in rows)
        for name in sorted(
            {
                topology
                for row in rows
                for topology in row["representative_topologies"]
            }
        )
    }
    comparable_rows = [row for row in rows if row["same_feature_comparable"]]
    split_rows = {
        split: [row for row in comparable_rows if row["analysis_split"] == split]
        for split in ("development", "holdout")
    }
    rank_mismatch = {
        metric: rank_mismatch_diagnostic(
            split_rows["development"],
            split_rows["holdout"],
            action_metric=metric,
        )
        for metric in POLICY_MEASURES
    }
    nested_models = {
        metric: nested_holdout_models(
            split_rows["development"],
            split_rows["holdout"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1_000 + index,
            action_metric=metric,
        )
        for index, metric in enumerate(POLICY_MEASURES)
    }
    output = {
        "schema_version": 2,
        "analysis": "same-feature OpenVLA action sensitivity and SAFE response",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("action_monitor_analysis.py")
            ),
        },
        "analysis_contract": {
            "status": "exploratory post-hoc analysis after opening the holdout",
            "question": (
                "does an intervention change OpenVLA's action evidence without producing "
                "a commensurate SAFE response, and does that mismatch carry information "
                "about terminal policy failure"
            ),
            "same_feature": (
                "OpenVLA action-head logits and SAFE both consume the seventh generated "
                "token's final language-model feature; because the command contains seven "
                "separately generated tokens, full-token and executed-command measures are "
                "reported as broader policy-output diagnostics"
            ),
            "policy_measures": {
                "same_feature_action_js_at_fault": (
                    "Jensen-Shannon divergence for the seventh token, whose final feature "
                    "is also SAFE's input"
                ),
                "full_action_mean_js_at_fault": (
                    "mean Jensen-Shannon divergence across all seven generated action tokens"
                ),
                "executed_command_l2_at_fault": (
                    "Euclidean displacement of the exact command sent to LIBERO"
                ),
            },
            "primary_monitor_measure": (
                "absolute difference between paired SAFE-MLP score increments at the "
                "intervention step"
            ),
            "outcome": (
                "terminal task failure; no branch alarmed in the common 25-step window, "
                "so full-trajectory silent-versus-detected labels are reported only as "
                "secondary outcome decompositions"
            ),
            "guardrail": (
                "the transparent rank mismatch and fixed nested logistic models are "
                "diagnostics, not a proposed detector or confirmatory risk rule"
            ),
            "topology_scope": (
                "primary models exclude action-only boundaries, where logits or actions "
                "are changed downstream of the feature SAFE reads; those records remain "
                "in the source artifact for separate structural comparisons"
            ),
        },
        "sources": sources,
        "coverage": {
            "physical_continuations": len(rows),
            "same_feature_comparable": len(comparable_rows),
            "representative_topologies": topology_counts,
            "development": len(split_rows["development"]),
            "holdout": len(split_rows["holdout"]),
            "failures": sum(bool(row["policy_failure"]) for row in rows),
        },
        "results": {
            split: {
                "continuations": len(selected),
                "failures": sum(bool(row["policy_failure"]) for row in selected),
                "metric_comparisons": _metric_comparisons(
                    selected, args.bootstrap_samples, args.seed + 100 * index
                ),
                "policy_monitor_coupling": {
                    metric: coupling_summary(selected, action_metric=metric)
                    for metric in POLICY_MEASURES
                },
            }
            for index, (split, selected) in enumerate(split_rows.items())
        },
        "rank_mismatch_by_policy_measure": rank_mismatch,
        "nested_holdout_models_by_policy_measure": nested_models,
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
