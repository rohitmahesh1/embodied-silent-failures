from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.replay import ACTION_COLUMNS
from embodied_silent_failures.temporal_campaign import validate_campaign_manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired outcomes from an OpenVLA temporal-fault pilot."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--safe-scores", required=True, type=Path)
    parser.add_argument("--safe-score-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    return parser.parse_args()


def _completion(path: Path) -> dict[str, Any] | None:
    values = sorted(path.glob("*.complete.json"))
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"attempt has multiple completion records: {path}")
    return load_json(values[0])


def _csv_row(path: Path, step: int) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not 0 <= step < len(rows):
        raise IndexError(f"policy step {step} is outside {path}")
    return rows[step]


def _action(row: dict[str, str]) -> list[float]:
    return [float(row[column]) for column in ACTION_COLUMNS]


def _safe_feature(path: Path, step: int) -> Any:
    with path.open("rb") as file:
        value = pickle.load(file)["hidden_states"]
    return value[step, -1, :].to(dtype=value.dtype).float()


def _difference(left: list[float], right: list[float]) -> dict[str, float | bool]:
    values = [a - b for a, b in zip(left, right)]
    return {
        "action_changed": any(value != 0 for value in values),
        "action_l2": math.sqrt(sum(value * value for value in values)),
        "action_linf": max(abs(value) for value in values),
    }


def _safe_index(json_value: dict[str, Any], archive_path: Path) -> dict[str, Any]:
    import numpy as np

    archive = np.load(archive_path)
    runs = [str(value) for value in archive["runs"]]
    lengths = archive["lengths"].astype(int)
    scores = archive["scores"]
    bands = archive["bands"]
    alphas = archive["alphas"].astype(float)
    primary = float(json_value["monitor"]["primary_alpha"])
    candidates = [
        index
        for index, alpha in enumerate(alphas)
        if math.isclose(alpha, primary, rel_tol=0, abs_tol=1e-8)
    ]
    if len(candidates) != 1:
        raise ValueError("SAFE archive does not contain one primary alpha")
    band = bands[candidates[0]]
    return {
        run: {
            "scores": scores[index, : lengths[index]],
            "band": band[: lengths[index]],
        }
        for index, run in enumerate(runs)
    }


def _weighted_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return None
    weights = [1 / float(row["site_inclusion_probability"]) for row in eligible]
    return sum(weight * bool(row[key]) for weight, row in zip(weights, eligible)) / sum(
        weights
    )


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    failures = [row for row in complete if row["task_failure"]]
    return {
        "planned": len(rows),
        "complete": len(complete),
        "errors": len(rows) - len(complete),
        "masked_replacements": sum(row["replacement_exact_equal"] for row in complete),
        "action_changes": sum(row["action_changed"] for row in complete),
        "task_failures": len(failures),
        "silent_failures_post_fault_any": sum(
            row["silent_failure_post_fault_any"] for row in complete
        ),
        "silent_failures_within_10": sum(
            row["silent_failure_within_10"] for row in complete
        ),
        "preexisting_safe_alarms": sum(row["safe_alarm_before_fault"] for row in complete),
        "weighted_action_change_rate": _weighted_rate(complete, "action_changed"),
        "weighted_task_failure_rate": _weighted_rate(complete, "task_failure"),
        "weighted_joint_residual_risk_post_fault_any": _weighted_rate(
            complete, "silent_failure_post_fault_any"
        ),
    }


def main() -> None:
    args = _arguments()
    manifest = load_json(args.manifest)
    validate_campaign_manifest(manifest)
    sites = {site["site_id"]: site for site in manifest["sites"]}
    safe_json = load_json(args.safe_scores)
    safe_records = {record["run"]: record for record in safe_json["records"]}
    safe_arrays = _safe_index(safe_json, args.safe_score_archive)
    clean_meta = {
        (int(item["task_id"]), int(item["episode_index"])): item
        for item in manifest["clean_trajectories"]
    }
    rows = []
    for attempt in manifest["attempts"]:
        attempt_id = str(attempt["attempt_id"])
        site = sites[str(attempt["site_id"])]
        attempt_dir = args.campaign_dir / "attempts" / attempt_id
        completion = _completion(attempt_dir)
        base = {
            **attempt,
            "topology": "|".join(site["topologies"]),
            "stratum": site["stratum"],
            "owner": "|".join(site["architecture"]["observed_owners"]),
            "literal_module_role": site["architecture"]["literal_module_role"],
            "output_port": site["identity"]["output_port"],
        }
        if completion is None:
            error_path = attempt_dir / "attempt.error.json"
            error = load_json(error_path) if error_path.is_file() else {}
            rows.append(
                {
                    **base,
                    "status": str(error.get("status", "missing")),
                    "error_reason": str(error.get("reason", "missing_completion")),
                }
            )
            continue

        step = int(attempt["policy_step"])
        clean = clean_meta[(int(attempt["task_id"]), int(attempt["episode_index"]))]
        clean_csv = args.baseline_dir / clean["artifacts"]["csv"]["staged_name"]
        clean_pickle = args.baseline_dir / clean["artifacts"]["pickle"]["staged_name"]
        fault_csv = attempt_dir / completion["files"]["csv"]
        fault_pickle = attempt_dir / completion["files"]["pickle"]
        action_delta = _difference(
            _action(_csv_row(fault_csv, step)),
            _action(_csv_row(clean_csv, step)),
        )
        clean_feature = _safe_feature(clean_pickle, step)
        fault_feature = _safe_feature(fault_pickle, step)
        feature_difference = fault_feature - clean_feature
        score = safe_records.get(attempt_id)
        arrays = safe_arrays.get(attempt_id)
        if score is None or arrays is None:
            raise ValueError(f"SAFE did not score completed attempt {attempt_id}")
        before = arrays["scores"][:step] >= arrays["band"][:step]
        safe_alarm_before = bool(before.any())
        primary = format(float(safe_json["monitor"]["primary_alpha"]), "g")
        windows = score["alarms"][primary]
        post_alarm = bool(windows["post_fault_any"]["triggered"])
        alarm_10 = bool(windows["within_10_steps"]["triggered"])
        task_failure = not bool(completion["success"])
        comparison = completion["fault"]["comparison"]
        rows.append(
            {
                **base,
                "status": "complete",
                "error_reason": "",
                "fault_policy_steps": int(completion["policy_steps"]),
                "task_failure": task_failure,
                **action_delta,
                "safe_feature_l2": float(feature_difference.norm().item()),
                "safe_feature_linf": float(feature_difference.abs().max().item()),
                "safe_feature_changed": bool((feature_difference != 0).any()),
                "replacement_exact_equal": bool(comparison["exact_equal"]),
                "replacement_normalized_difference_l2": comparison[
                    "normalized_difference_l2"
                ],
                "replacement_changed_element_count": comparison[
                    "changed_element_count"
                ],
                "safe_alarm_before_fault": safe_alarm_before,
                "safe_alarm_post_fault_any": post_alarm,
                "safe_alarm_within_10": alarm_10,
                "silent_failure_post_fault_any": task_failure and not post_alarm,
                "silent_failure_within_10": task_failure and not alarm_10,
                "clear_prefix_silent_failure": (
                    task_failure and not safe_alarm_before and not post_alarm
                ),
                "maximum_prefix_observation_error": completion[
                    "paired_prefix_validation"
                ]["maximum_numeric_observation_error"],
                "maximum_prefix_action_error": completion[
                    "paired_prefix_validation"
                ]["maximum_executed_action_error"],
                "maximum_prefix_safe_feature_error": completion[
                    "paired_prefix_validation"
                ]["maximum_safe_feature_error"],
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"topology:{row['topology']}"] .append(row)
        groups[f"phase:{row['phase']}"] .append(row)
        groups[f"stratum:{row['stratum']}"] .append(row)
    output = {
        "schema_version": 1,
        "analysis": "exploratory paired temporal-fault pilot",
        "estimand": (
            "Risk over a site-uniform eligible table and a uniform draw from the "
            "frozen clean-success trajectory frame within each declared phase."
        ),
        "inference_scope": (
            "This pilot estimates injectability and event rates; it is not a frozen "
            "confirmatory significance test."
        ),
        "artifacts": {
            "manifest": {"path": str(args.manifest), "sha256": file_sha256(args.manifest)},
            "safe_scores": {
                "path": str(args.safe_scores),
                "sha256": file_sha256(args.safe_scores),
            },
            "safe_score_archive": {
                "path": str(args.safe_score_archive),
                "sha256": file_sha256(args.safe_score_archive),
            },
        },
        "overall": _group_summary(rows),
        "groups": {name: _group_summary(values) for name, values in sorted(groups.items())},
        "records": rows,
    }
    write_json_atomic(args.output, output)
    columns = sorted({key for row in rows for key in row})
    write_csv_atomic(
        args.records_csv,
        [{key: row.get(key, "") for key in columns} for row in rows],
    )
    print(json.dumps(output["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
