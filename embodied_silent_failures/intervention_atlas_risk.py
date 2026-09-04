from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, load_json

PRIMARY_WINDOW = "post_fault_any"
ALARM_WINDOW_NAMES = (
    "within_5_steps",
    "within_10_steps",
    "within_25_steps",
    PRIMARY_WINDOW,
)
GRAPH_FEATURES = ("topology", "hook_kind", "owner", "depth_band", "output_family")
CONTEXT_FEATURES = ("phase_fraction",)
ACTION_FEATURES = (
    "temporal_replacement_size",
    "action_logit_change",
    "raw_action_change",
    "executed_command_change",
    "changed_action_token_fraction",
)
MONITOR_FEATURES = ("safe_feature_change", "safe_contribution_change")
MODEL_SPECS = {
    "graph_only_residual": {
        "outcome": "silent_failure",
        "features": GRAPH_FEATURES,
    },
    "graph_context_residual": {
        "outcome": "silent_failure",
        "features": GRAPH_FEATURES + CONTEXT_FEATURES,
    },
    "conventional_vulnerability": {
        "outcome": "policy_failure",
        "features": GRAPH_FEATURES + CONTEXT_FEATURES + ACTION_FEATURES,
    },
    "local_residual": {
        "outcome": "silent_failure",
        "features": GRAPH_FEATURES + CONTEXT_FEATURES + ACTION_FEATURES,
    },
    "monitor_aware_residual": {
        "outcome": "silent_failure",
        "features": (
            GRAPH_FEATURES + CONTEXT_FEATURES + ACTION_FEATURES + MONITOR_FEATURES
        ),
    },
}


def _stratum_parts(record: dict[str, Any]) -> dict[str, str]:
    values = str(record["sampling"]["stratum"]).split(":")
    if len(values) != len(GRAPH_FEATURES):
        raise ValueError(f"sampling stratum has {len(values)} fields, expected five")
    return dict(zip(GRAPH_FEATURES, values, strict=True))


def _finite(record_id: str, name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{record_id} has non-finite {name}")
    return number


def flatten_record(record: dict[str, Any], primary_alpha: float) -> dict[str, Any]:
    if not record.get("primary_eligible"):
        raise ValueError("only primary-eligible atlas records can be flattened")
    record_id = str(record["record_id"])
    local = record["local_measurements"]
    alpha = format(float(primary_alpha), "g")
    alarm = bool(
        record["safe_faulted_evidence"]["alarms"][alpha][PRIMARY_WINDOW][
            "triggered"
        ]
    )
    faulted_alarms = record["safe_faulted_evidence"]["alarms"][alpha]
    clean_evidence_alarms = record["safe_clean_evidence_same_suffix"]["alarms"][alpha]
    policy_failure = bool(record["policy_failure"])
    row = {
        "record_id": record_id,
        "context_id": str(record["context_id"]),
        "site_id": str(record["site_id"]),
        "task_id": int(record["context"]["task_id"]),
        "episode_index": int(record["context"]["episode_index"]),
        "phase": str(record["context"]["phase"]),
        "physical_run": str(record["physical_run"]),
        "policy_failure": policy_failure,
        "safe_alarm": alarm,
        "safe_miss_given_failure": bool(policy_failure and not alarm),
        "silent_failure": bool(policy_failure and not alarm),
        **_stratum_parts(record),
        "phase_fraction": _finite(
            record_id, "phase_fraction", record["context"]["phase_fraction"]
        ),
        "temporal_replacement_size": _finite(
            record_id,
            "temporal_replacement_size",
            local["fault"]["comparison"]["normalized_difference_l2"],
        ),
        "action_logit_change": _finite(
            record_id,
            "action_logit_change",
            local["action_logits"]["normalized_difference_l2"],
        ),
        "raw_action_change": _finite(
            record_id,
            "raw_action_change",
            local["raw_action"]["normalized_difference_l2"],
        ),
        "executed_command_change": _finite(
            record_id,
            "executed_command_change",
            local["executed_command"]["normalized_difference_l2"],
        ),
        "changed_action_token_fraction": (
            int(local["action_tokens"]["changed_token_count"]) / 7
        ),
        "safe_feature_change": _finite(
            record_id,
            "safe_feature_change",
            local["safe_input"]["normalized_difference_l2"],
        ),
        "safe_contribution_change": _finite(
            record_id,
            "safe_contribution_change",
            record["safe_contribution"]["faulted_minus_clean"],
        ),
        "site_inverse_probability_weight": _finite(
            record_id,
            "site_inverse_probability_weight",
            record["sampling"]["site_inverse_probability_weight"],
        ),
    }
    for window in ALARM_WINDOW_NAMES:
        row[f"safe_alarm:{window}"] = bool(faulted_alarms[window]["triggered"])
        row[f"clean_evidence_alarm:{window}"] = bool(
            clean_evidence_alarms[window]["triggered"]
        )
    return row


def load_analysis_rows(
    paths: list[Path], *, analysis_split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    artifacts = []
    monitor_identity = None
    manifest_sha256 = None
    for path in paths:
        analysis = load_json(path)
        if analysis.get("analysis_split") != analysis_split:
            raise ValueError(f"atlas analysis is not {analysis_split}: {path}")
        current_monitor = {
            key: analysis["monitor"][key]
            for key in (
                "checkpoint_sha256",
                "configuration_sha256",
                "split_manifest_sha256",
                "clean_score_archive_sha256",
                "primary_alpha",
            )
        }
        if monitor_identity is None:
            monitor_identity = current_monitor
        elif monitor_identity != current_monitor:
            raise ValueError("atlas shards used different frozen SAFE monitors")
        current_manifest = str(analysis["source"]["manifest_sha256"])
        if manifest_sha256 is None:
            manifest_sha256 = current_manifest
        elif manifest_sha256 != current_manifest:
            raise ValueError("atlas shards used different intervention manifests")
        rows.extend(
            flatten_record(record, float(current_monitor["primary_alpha"]))
            for record in analysis["records"]
            if record.get("primary_eligible")
        )
        artifacts.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("atlas analysis shards contain duplicate intervention records")
    return rows, {
        "analysis_split": analysis_split,
        "artifacts": artifacts,
        "manifest_sha256": manifest_sha256,
        "monitor": monitor_identity,
    }


def attach_trajectory_weights(
    rows: list[dict[str, Any]], manifest_path: Path
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    probabilities = {
        (int(value["task_id"]), int(value["episode_index"])): float(
            value["trajectory_inclusion_probability"]
        )
        for value in manifest["clean_trajectories"]
    }
    for row in rows:
        key = (row["task_id"], row["episode_index"])
        probability = probabilities.get(key)
        if probability is None or not 0 < probability <= 1:
            raise ValueError(f"missing trajectory inclusion probability for {key}")
        row["trajectory_inclusion_probability"] = probability
        row["graph_population_weight"] = (
            row["site_inverse_probability_weight"] / probability
        )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
    }


def model_features(row: dict[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    result = {}
    for name in names:
        value = row[name]
        if name in GRAPH_FEATURES:
            result[f"{name}={value}"] = 1.0
        elif name in {
            "temporal_replacement_size",
            "action_logit_change",
            "raw_action_change",
            "executed_command_change",
            "safe_feature_change",
        }:
            if float(value) < 0:
                raise ValueError(f"{name} cannot be negative")
            result[name] = math.log1p(float(value))
        else:
            result[name] = float(value)
    return result


def trajectory_groups(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[f"task{row['task_id']}:episode{row['episode_index']}"] .append(index)
    return dict(sorted(groups.items()))


def weighted_rate(
    rows: list[dict[str, Any]], outcome: str, weight: str | None = None
) -> float | None:
    if not rows:
        return None
    weights = [1.0 if weight is None else float(row[weight]) for row in rows]
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("rate weights must have positive sum")
    return sum(w * int(row[outcome]) for row, w in zip(rows, weights, strict=True)) / denominator


def rate_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"all": rows}
    groups.update(
        {
            f"topology:{topology}": [
                row for row in rows if row["topology"] == topology
            ]
            for topology in sorted({row["topology"] for row in rows})
        }
    )
    output = {}
    for name, selected in groups.items():
        failures = [row for row in selected if row["policy_failure"]]
        output[name] = {
            "interventions": len(selected),
            "physical_runs": len({row["physical_run"] for row in selected}),
            "policy_failures": len(failures),
            "silent_failures": sum(row["silent_failure"] for row in selected),
            "detection_timing_given_policy_failure": {
                window: {
                    "detected": sum(row[f"safe_alarm:{window}"] for row in failures),
                    "probability": weighted_rate(
                        [
                            {
                                **row,
                                "detected_in_window": row[f"safe_alarm:{window}"],
                            }
                            for row in failures
                        ],
                        "detected_in_window",
                    ),
                }
                for window in ALARM_WINDOW_NAMES
            },
            "matched_fault_step_evidence": {
                "faulted_evidence_suppressed_alarm": sum(
                    not row[f"safe_alarm:{PRIMARY_WINDOW}"]
                    and row[f"clean_evidence_alarm:{PRIMARY_WINDOW}"]
                    for row in failures
                ),
                "faulted_evidence_induced_alarm": sum(
                    row[f"safe_alarm:{PRIMARY_WINDOW}"]
                    and not row[f"clean_evidence_alarm:{PRIMARY_WINDOW}"]
                    for row in failures
                ),
            },
            "sampled_population": {
                "policy_failure_probability": weighted_rate(
                    selected, "policy_failure"
                ),
                "safe_miss_given_policy_failure": weighted_rate(
                    failures, "safe_miss_given_failure"
                ),
                "silent_failure_probability": weighted_rate(
                    selected, "silent_failure"
                ),
            },
            "graph_population_ipw": {
                "policy_failure_probability": weighted_rate(
                    selected, "policy_failure", "graph_population_weight"
                ),
                "safe_miss_given_policy_failure": weighted_rate(
                    failures, "safe_miss_given_failure", "graph_population_weight"
                ),
                "silent_failure_probability": weighted_rate(
                    selected, "silent_failure", "graph_population_weight"
                ),
            },
        }
    return output
