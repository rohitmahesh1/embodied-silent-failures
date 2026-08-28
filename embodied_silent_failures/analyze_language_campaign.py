from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze scored OpenVLA language-block interventions."
    )
    parser.add_argument(
        "--scores", action="append", dest="score_paths", required=True, type=Path
    )
    parser.add_argument(
        "--split", choices=("development", "holdout"), default="development"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_rate(
    rows: list[dict[str, Any]],
    field: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.get(field) is not None]
    numerator = sum(bool(row[field]) for row in eligible)
    result = {
        "numerator": numerator,
        "denominator": len(eligible),
        "estimate": numerator / len(eligible) if eligible else None,
        "trajectory_cluster_bootstrap_95": None,
    }
    if not eligible:
        return result

    by_trajectory: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for row in eligible:
        key = (int(row["task_id"]), int(row["episode_index"]))
        by_trajectory[key].append(bool(row[field]))
    keys = sorted(by_trajectory)
    result["trajectory_clusters"] = len(keys)
    if samples <= 0:
        return result

    cluster_totals = {
        key: (sum(by_trajectory[key]), len(by_trajectory[key])) for key in keys
    }
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        selected_totals = [cluster_totals[key] for key in selected]
        estimates.append(
            sum(value[0] for value in selected_totals)
            / sum(value[1] for value in selected_totals)
        )
    result["trajectory_cluster_bootstrap_95"] = [
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    ]
    result["bootstrap_samples"] = samples
    return result


def analysis_row(record: dict[str, Any], primary_alpha: str) -> dict[str, Any]:
    context = record["context"]
    local = record["local_measurements"]
    eligible = bool(
        record.get("status") == "scored"
        and record.get("composition_verified")
        and record.get("control_success")
        and record.get("terminal_success") is not None
        and record.get("monitor_horizon") == "complete_physical_trace"
    )
    task_failure = not bool(record["terminal_success"]) if eligible else None
    windows = record.get("alarms", {}).get(primary_alpha, {})
    alarm_any = (
        bool(windows["post_fault_any"]["triggered"])
        if eligible and "post_fault_any" in windows
        else None
    )
    alarm_10 = (
        bool(windows["within_10_steps"]["triggered"])
        if eligible and "within_10_steps" in windows
        else None
    )
    alarm_25 = (
        bool(windows["within_25_steps"]["triggered"])
        if eligible and "within_25_steps" in windows
        else None
    )
    immediate_alarm = bool(record["alarm_at_fault"]) if eligible else None
    before = bool(record["alarm_before_fault"]) if eligible else None
    propagation = local.get("propagation", [])
    final_propagation = propagation[-1] if propagation else {}
    executed_command = local.get("executed_command")
    return {
        "record_id": record["record_id"],
        "status": record["status"],
        "context_id": record["context_id"],
        "analysis_split": context["analysis_split"],
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "phase": context["phase"],
        "worker_shard": int(context["worker_shard"]),
        "policy_step": int(context["policy_step"]),
        "action_token_position": int(context["action_token_position"]),
        "layer_index": int(record["layer_index"]),
        "normalized_depth": int(record["layer_index"]) / 31,
        "site_id": local.get("site_id"),
        "command_changed": (
            not bool(executed_command["exact_equal"])
            if isinstance(executed_command, dict)
            else None
        ),
        "command_id": record.get("command_id"),
        "command_group_size": int(record.get("command_group_size", 1)),
        "physical_run": record.get("physical_run"),
        "terminal_evidence": record.get("terminal_evidence"),
        "control_success": bool(record.get("control_success")),
        "eligible_causal_outcome": eligible,
        "terminal_success": record.get("terminal_success") if eligible else None,
        "task_failure": task_failure,
        "injection_l2": local.get("injection", {}).get("difference_l2"),
        "injection_normalized_l2": local.get("injection", {}).get(
            "normalized_difference_l2"
        ),
        "final_propagation_l2": final_propagation.get("difference_l2"),
        "final_propagation_normalized_l2": final_propagation.get(
            "normalized_difference_l2"
        ),
        "safe_feature_l2": local.get("safe_feature", {}).get("difference_l2"),
        "safe_feature_normalized_l2": local.get("safe_feature", {}).get(
            "normalized_difference_l2"
        ),
        "command_l2": local.get("executed_command", {}).get("difference_l2"),
        "command_normalized_l2": local.get("executed_command", {}).get(
            "normalized_difference_l2"
        ),
        "score_at_fault": record.get("score_at_fault"),
        "control_score_at_fault": record.get("control_score_at_fault"),
        "score_change_from_control_at_fault": record.get(
            "score_change_from_control_at_fault"
        ),
        "safe_alarm_before_fault": before,
        "safe_alarm_at_fault": immediate_alarm,
        "safe_alarm_within_10": alarm_10,
        "safe_alarm_within_25": alarm_25,
        "safe_alarm_post_fault_any": alarm_any,
        "fault_specific_immediate_alarm": (
            immediate_alarm and not bool(record["control_alarm_at_fault"])
            if eligible
            else None
        ),
        "missed_failure_at_command_time": (
            task_failure and not immediate_alarm if eligible else None
        ),
        "silent_failure_within_10": (
            task_failure and not alarm_10 if eligible else None
        ),
        "silent_failure_within_25": (
            task_failure and not alarm_25 if eligible else None
        ),
        "silent_failure_post_fault_any": (
            task_failure and not alarm_any if eligible else None
        ),
        "operational_silent_failure": (
            task_failure and not before and not alarm_any if eligible else None
        ),
    }


RATE_FIELDS = (
    "command_changed",
    "task_failure",
    "safe_alarm_before_fault",
    "safe_alarm_at_fault",
    "safe_alarm_within_10",
    "safe_alarm_within_25",
    "safe_alarm_post_fault_any",
    "fault_specific_immediate_alarm",
    "missed_failure_at_command_time",
    "silent_failure_within_10",
    "silent_failure_within_25",
    "silent_failure_post_fault_any",
    "operational_silent_failure",
)


def group_summary(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    return {
        "interventions": len(rows),
        "eligible_causal_outcomes": sum(row["eligible_causal_outcome"] for row in rows),
        "distinct_physical_runs": len(
            {row["physical_run"] for row in rows if row.get("physical_run")}
        ),
        "rates": {
            field: clustered_rate(rows, field, samples=samples, seed=seed + index)
            for index, field in enumerate(RATE_FIELDS)
        },
    }


def _group_rows(
    rows: list[dict[str, Any]], field: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[field])].append(row)
    return result


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    score_values = [load_json(path) for path in args.score_paths]
    if len({value["source_campaign"]["worker_shard"] for value in score_values}) != len(
        score_values
    ):
        raise ValueError("language score inputs repeat a worker shard")
    monitor_hashes = {
        value["monitor"]["checkpoint_sha256"] for value in score_values
    }
    if len(monitor_hashes) != 1:
        raise ValueError("language score inputs used different SAFE checkpoints")

    contexts = [item for value in score_values for item in value["contexts"]]
    records = [item for value in score_values for item in value["records"]]
    context_ids = [str(value["context_id"]) for value in contexts]
    if len(context_ids) != 150 or len(set(context_ids)) != 150:
        raise ValueError("combined language scores must cover 150 unique contexts")
    record_ids = [str(value["record_id"]) for value in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("combined language scores contain duplicate interventions")

    selected_contexts = [
        value
        for value in contexts
        if value.get("context", {}).get("analysis_split") == args.split
    ]
    selected_context_ids = {str(value["context_id"]) for value in selected_contexts}
    selected_records = [
        value for value in records if str(value["context_id"]) in selected_context_ids
    ]
    primary_alpha = format(float(score_values[0]["monitor"]["primary_alpha"]), "g")
    rows = [analysis_row(record, primary_alpha) for record in selected_records]

    groups = {}
    for field in (
        "worker_shard",
        "phase",
        "action_token_position",
        "layer_index",
        "task_id",
    ):
        groups[field] = {
            name: group_summary(values, samples=0, seed=args.seed)
            for name, values in sorted(_group_rows(rows, field).items())
        }
    complete_contexts = [
        value for value in selected_contexts if value["status"] == "complete"
    ]
    output = {
        "schema_version": 1,
        "analysis": "language-block residual risk",
        "analysis_split": args.split,
        "estimand": (
            "One intervention is one language block at one sampled context. Exact-command "
            "members retain distinct SAFE evidence but share terminal physical evidence."
        ),
        "uncertainty": (
            "Percentile bootstrap resamples whole task/episode trajectories; it does not "
            "treat command-group members or phases as independent robot trials."
        ),
        "holdout_policy": (
            "Development analysis selects risk descriptions. Holdout outcomes are analyzed "
            "only after that description and its evaluation procedure are frozen."
        ),
        "artifacts": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in args.score_paths
        ],
        "coverage": {
            "planned_contexts": len(selected_contexts),
            "complete_contexts": len(complete_contexts),
            "unresolved_contexts": len(selected_contexts) - len(complete_contexts),
            "planned_interventions": len(selected_contexts) * 32,
            "recorded_interventions": len(rows),
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "successful_control_contexts": sum(
                bool(value.get("control_success")) for value in complete_contexts
            ),
            "failed_control_contexts": sum(
                value.get("control_success") is False for value in complete_contexts
            ),
            "unique_faulted_commands": sum(
                int(value.get("unique_faulted_commands", 0)) for value in complete_contexts
            ),
        },
        "overall": group_summary(
            rows, samples=args.bootstrap_samples, seed=args.seed
        ),
        "groups": groups,
        "records": rows,
    }
    write_json_atomic(args.output, output)
    columns = sorted({key for row in rows for key in row})
    write_csv_atomic(
        args.records_csv,
        [{key: row.get(key, "") for key in columns} for row in rows],
    )
    print(
        json.dumps(
            {
                "analysis_split": args.split,
                "coverage": output["coverage"],
                "overall": output["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
