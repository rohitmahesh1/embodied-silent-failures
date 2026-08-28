from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


COMMAND_SIGNALS = (
    "injection_normalized_l2",
    "final_propagation_normalized_l2",
    "safe_feature_normalized_l2",
)
COMMAND_SIGNAL_DESCRIPTIONS = {
    "injection_normalized_l2": "change at the chosen language-block output",
    "final_propagation_normalized_l2": "change after the final language block",
    "safe_feature_normalized_l2": (
        "change across all seven recorded action-token features; this is not the "
        "single final-token feature selected by the frozen SAFE monitor"
    ),
}
COMMAND_COMPONENTS = (
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def roc_auc(labels: list[bool], values: list[float]) -> float:
    if len(labels) != len(values) or not labels:
        raise ValueError("ROC-AUC labels and values must have equal nonzero length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both outcome classes")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ROC-AUC values must be finite")

    ordered = sorted(range(len(values)), key=values.__getitem__)
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2
        positive_rank_sum += average_rank * sum(
            labels[ordered[index]] for index in range(start, stop)
        )
        start = stop
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def command_signal_auc(
    rows: list[dict[str, Any]],
    signal: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("eligible_causal_outcome") and row.get(signal) is not None
    ]
    labels = [bool(row["command_changed"]) for row in eligible]
    values = [float(row[signal]) for row in eligible]
    result = {
        "signal": signal,
        "interventions": len(eligible),
        "command_changes": sum(labels),
        "roc_auc": roc_auc(labels, values),
        "trajectory_cluster_bootstrap_95": None,
        "bootstrap_valid_samples": 0,
    }
    if bootstrap_samples <= 0:
        return result

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(eligible):
        groups[(int(row["task_id"]), int(row["episode_index"]))].append(index)
    group_values = list(groups.values())
    positive_counts = [sum(labels[index] for index in group) for group in group_values]
    negative_counts = [
        len(group) - positive
        for group, positive in zip(group_values, positive_counts, strict=True)
    ]
    concordance = []
    for positive_group in group_values:
        positive_values = [values[index] for index in positive_group if labels[index]]
        row = []
        for negative_group in group_values:
            negative_values = [
                values[index] for index in negative_group if not labels[index]
            ]
            row.append(
                sum(
                    1.0 if positive > negative else 0.5 if positive == negative else 0.0
                    for positive in positive_values
                    for negative in negative_values
                )
            )
        concordance.append(row)
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        multiplicities = [0] * len(group_values)
        for _group in group_values:
            multiplicities[rng.randrange(len(group_values))] += 1
        positives = sum(
            count * multiplicity
            for count, multiplicity in zip(
                positive_counts, multiplicities, strict=True
            )
        )
        negatives = sum(
            count * multiplicity
            for count, multiplicity in zip(
                negative_counts, multiplicities, strict=True
            )
        )
        if not positives or not negatives:
            continue
        favorable = sum(
            left_count * right_count * concordance[left][right]
            for left, left_count in enumerate(multiplicities)
            for right, right_count in enumerate(multiplicities)
        )
        estimates.append(favorable / (positives * negatives))
    result["bootstrap_valid_samples"] = len(estimates)
    result["trajectory_cluster_bootstrap_95"] = (
        [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]
        if estimates
        else None
    )
    return result


def scored_record_index(
    score_documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records = [record for document in score_documents for record in document["records"]]
    result = {str(record["record_id"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("language score documents contain duplicate record IDs")
    return result


def physical_command_branches(
    analysis_rows: list[dict[str, Any]],
    score_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in analysis_rows:
        if not row.get("eligible_causal_outcome") or not row.get("command_changed"):
            continue
        record_id = str(row["record_id"])
        if record_id not in score_index:
            raise ValueError(f"analysis record is absent from scores: {record_id}")
        physical_run = row.get("physical_run")
        if not physical_run:
            raise ValueError(f"changed command has no physical run: {record_id}")
        grouped[str(physical_run)].append((row, score_index[record_id]))

    branches = []
    for physical_run, members in sorted(grouped.items()):
        first_row, first_score = members[0]
        first_local = first_score["local_measurements"]
        clean = tuple(float(value) for value in first_local["clean_executed_command"])
        faulted = tuple(
            float(value) for value in first_local["faulted_executed_command"]
        )
        if len(clean) != len(COMMAND_COMPONENTS) or len(faulted) != len(
            COMMAND_COMPONENTS
        ):
            raise ValueError(
                f"physical branch has a non-seven-dimensional command: {physical_run}"
            )

        stable_fields = (
            "analysis_split",
            "context_id",
            "task_id",
            "episode_index",
            "phase",
            "policy_step",
            "action_token_position",
            "command_id",
            "physical_run",
            "task_failure",
            "operational_silent_failure",
            "safe_alarm_at_fault",
            "safe_alarm_within_25",
            "safe_alarm_post_fault_any",
        )
        for row, score in members:
            local = score["local_measurements"]
            if any(
                row.get(field) != first_row.get(field) for field in stable_fields
            ):
                raise ValueError(
                    "physical branch members disagree on outcome or identity: "
                    f"{physical_run}"
                )
            member_clean = tuple(
                float(value) for value in local["clean_executed_command"]
            )
            member_faulted = tuple(
                float(value) for value in local["faulted_executed_command"]
            )
            if member_clean != clean:
                raise ValueError(
                    f"physical branch members disagree on clean command: {physical_run}"
                )
            if member_faulted != faulted:
                raise ValueError(
                    "physical branch members disagree on faulted command: "
                    f"{physical_run}"
                )

        differences = tuple(
            value - reference for reference, value in zip(clean, faulted, strict=True)
        )
        branch = {
            "physical_run": physical_run,
            "analysis_split": first_row["analysis_split"],
            "context_id": first_row["context_id"],
            "task_id": int(first_row["task_id"]),
            "episode_index": int(first_row["episode_index"]),
            "phase": first_row["phase"],
            "policy_step": int(first_row["policy_step"]),
            "action_token_position": int(first_row["action_token_position"]),
            "command_id": first_row["command_id"],
            "member_interventions": len(members),
            "member_layers": sorted(int(row["layer_index"]) for row, _ in members),
            "task_failure": bool(first_row["task_failure"]),
            "operational_silent_failure": bool(
                first_row["operational_silent_failure"]
            ),
            "safe_alarm_at_fault": bool(first_row["safe_alarm_at_fault"]),
            "safe_alarm_within_25": bool(first_row["safe_alarm_within_25"]),
            "safe_alarm_post_fault_any": bool(
                first_row["safe_alarm_post_fault_any"]
            ),
            "command_l2": math.sqrt(sum(value * value for value in differences)),
        }
        for name, clean_value, faulted_value, difference in zip(
            COMMAND_COMPONENTS, clean, faulted, differences, strict=True
        ):
            branch[f"clean_{name}"] = clean_value
            branch[f"faulted_{name}"] = faulted_value
            branch[f"delta_{name}"] = difference
        branches.append(branch)
    return branches


def equal_count_fifths(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not branches:
        return []
    ordered = sorted(branches, key=lambda row: (row["command_l2"], row["physical_run"]))
    result = []
    for fifth in range(5):
        start = fifth * len(ordered) // 5
        stop = (fifth + 1) * len(ordered) // 5
        selected = ordered[start:stop]
        if not selected:
            continue
        failures = sum(row["task_failure"] for row in selected)
        silent = sum(row["operational_silent_failure"] for row in selected)
        result.append(
            {
                "fifth": fifth + 1,
                "branches": len(selected),
                "minimum_command_l2": float(selected[0]["command_l2"]),
                "maximum_command_l2": float(selected[-1]["command_l2"]),
                "task_failures": failures,
                "task_failure_rate": failures / len(selected),
                "silent_failures": silent,
                "silent_failure_rate": silent / len(selected),
            }
        )
    return result


def branch_summary(branches: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [branch for branch in branches if branch["task_failure"]]
    silent = [branch for branch in failures if branch["operational_silent_failure"]]
    detected = [
        branch for branch in failures if not branch["operational_silent_failure"]
    ]
    return {
        "changed_command_branches": len(branches),
        "task_failure_branches": len(failures),
        "detected_failure_branches": len(detected),
        "silent_failure_branches": len(silent),
        "alarm_at_fault_branches": sum(
            branch["safe_alarm_at_fault"] for branch in branches
        ),
        "alarm_within_25_branches": sum(
            branch["safe_alarm_within_25"] for branch in branches
        ),
        "command_magnitude_fifths": equal_count_fifths(branches),
    }
