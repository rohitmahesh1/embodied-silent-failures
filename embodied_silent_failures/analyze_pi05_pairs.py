from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.analysis import exact_binomial_two_sided
from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import load_json


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1 - estimate) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator if denominator else None,
        "wilson_95": _wilson(numerator, denominator),
    }


def _pairs(directory: Path) -> dict[tuple[int, int], dict[str, Any]]:
    values = {}
    for path in sorted(directory.glob("pairs/*/pair.complete.json")):
        item = load_json(path)
        key = int(item["task_id"]), int(item["episode_index"])
        if key in values:
            raise ValueError(f"duplicate pi0.5 pair: {key}")
        values[key] = item
    if not values:
        raise ValueError(f"no completed pi0.5 pairs found in {directory}")
    return values


def _scores(path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    value = load_json(path)
    records = {}
    for item in value.get("records", []):
        key = int(item["task_id"]), int(item["episode_index"]), item["label"]
        if key in records:
            raise ValueError(f"duplicate pi0.5 SAFE score: {key}")
        records[key] = item
    if not records:
        raise ValueError(f"no pi0.5 SAFE scores found in {path}")
    return records


def _safe_detected(score: dict[str, Any]) -> bool:
    alarm = score["alarm"]
    return bool(
        alarm["alarm_before_intervention"]
        or alarm["windows"]["through_terminal_outcome"]["triggered"]
    )


def _analyze_stale(
    pairs: dict[tuple[int, int], dict[str, Any]],
    scores: dict[tuple[int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    outcomes = Counter()
    freshness = Counter()
    safe_causal = Counter()
    safe_all_stale = Counter()
    by_task = Counter()
    records = []
    for key, pair in sorted(pairs.items()):
        if pair.get("pair_condition") != "stale_main_camera":
            raise ValueError(f"stale analysis received another pair condition: {key}")
        branches = pair["branches"]
        if set(branches) != {"current", "stale"}:
            raise ValueError(f"stale pair has invalid branches: {key}")
        current_success = bool(branches["current"]["success"])
        stale_success = bool(branches["stale"]["success"])
        if current_success and stale_success:
            category = "both_success"
        elif current_success and not stale_success:
            category = "stale_only_failure"
        elif not current_success and stale_success:
            category = "current_only_failure"
        else:
            category = "both_failure"
        outcomes[category] += 1
        by_task[(key[0], category)] += 1

        stale_intervention = branches["stale"]["intervention"]
        current_intervention = branches["current"]["intervention"]
        for name in (
            "source_metadata_alarm",
            "relabelled_metadata_alarm",
            "exact_duplicate_alarm",
            "selected_gate_alarm",
        ):
            freshness[f"stale_{name}"] += int(
                bool(stale_intervention["freshness"][name])
            )
            freshness[f"current_{name}"] += int(
                bool(current_intervention["freshness"][name])
            )

        stale_score = scores[(*key, "stale")]
        current_score = scores[(*key, "current")]
        stale_detected = _safe_detected(stale_score)
        safe_all_stale["detected"] += int(stale_detected)
        safe_all_stale["preexisting_alarm"] += int(
            stale_score["alarm"]["alarm_before_intervention"]
        )
        if category == "stale_only_failure":
            safe_causal["total"] += 1
            safe_causal["detected"] += int(stale_detected)
            safe_causal["silent"] += int(not stale_detected)
            safe_causal["clear_before_intervention"] += int(
                not stale_score["alarm"]["alarm_before_intervention"]
            )
            for name, window in stale_score["alarm"]["windows"].items():
                safe_causal[f"window_{name}"] += int(window["triggered"])
        records.append(
            {
                "task_id": key[0],
                "episode_index": key[1],
                "outcome": category,
                "safe_stale_detected": stale_detected,
                "safe_current_detected": _safe_detected(current_score),
                "freshness_exact_duplicate": bool(
                    stale_intervention["freshness"]["exact_duplicate_alarm"]
                ),
            }
        )

    count = len(pairs)
    causal = outcomes["stale_only_failure"]
    return {
        "pairs": count,
        "outcomes": {
            **dict(sorted(outcomes.items())),
            "exact_mcnemar_two_sided_p": exact_binomial_two_sided(
                outcomes["stale_only_failure"], outcomes["current_only_failure"]
            ),
            "stale_causal_failure_rate": _rate(causal, count),
        },
        "freshness_at_intervention": {
            **dict(sorted(freshness.items())),
            "stale_exact_duplicate_detection_rate": _rate(
                freshness["stale_exact_duplicate_alarm"], count
            ),
            "current_exact_duplicate_false_alarm_rate": _rate(
                freshness["current_exact_duplicate_alarm"], count
            ),
            "response_applied": False,
        },
        "safe_mlp": {
            "all_stale_branches": {
                **dict(sorted(safe_all_stale.items())),
                "detection_rate": _rate(safe_all_stale["detected"], count),
            },
            "causal_stale_failures": {
                **dict(sorted(safe_causal.items())),
                "silent_cofailure_rate": _rate(safe_causal["silent"], causal),
                "residual_silent_risk_per_pair": _rate(
                    safe_causal["silent"], count
                ),
            },
        },
        "by_task": {
            str(task): {
                category: by_task[(task, category)]
                for category in (
                    "both_success",
                    "stale_only_failure",
                    "current_only_failure",
                    "both_failure",
                )
            }
            for task in sorted({key[0] for key in pairs})
        },
        "records": records,
    }


def _analyze_null(
    pairs: dict[tuple[int, int], dict[str, Any]],
    scores: dict[tuple[int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    outcome_discordant = 0
    alarm_discordant = 0
    both_success = 0
    for key, pair in sorted(pairs.items()):
        if pair.get("pair_condition") != "current_current_null":
            raise ValueError(f"null analysis received another pair condition: {key}")
        branches = pair["branches"]
        if set(branches) != {"current_a", "current_b"}:
            raise ValueError(f"null pair has invalid branches: {key}")
        left = bool(branches["current_a"]["success"])
        right = bool(branches["current_b"]["success"])
        both_success += int(left and right)
        outcome_discordant += int(left != right)
        left_alarm = _safe_detected(scores[(*key, "current_a")])
        right_alarm = _safe_detected(scores[(*key, "current_b")])
        alarm_discordant += int(left_alarm != right_alarm)
    count = len(pairs)
    return {
        "pairs": count,
        "both_success": both_success,
        "outcome_discordance": _rate(outcome_discordant, count),
        "safe_alarm_discordance": _rate(alarm_discordant, count),
        "purpose": (
            "estimate paired execution noise when neither branch receives a stale frame"
        ),
    }


def analyze(
    stale_dir: Path,
    stale_scores: Path,
    *,
    null_dir: Path | None = None,
    null_scores: Path | None = None,
) -> dict[str, Any]:
    stale = _analyze_stale(_pairs(stale_dir), _scores(stale_scores))
    null = None
    if (null_dir is None) != (null_scores is None):
        raise ValueError("null pair directory and SAFE scores must be supplied together")
    if null_dir is not None and null_scores is not None:
        null = _analyze_null(_pairs(null_dir), _scores(null_scores))
    return {
        "schema_version": 1,
        "analysis": "paired pi0.5 stale-main-camera monitor comparison",
        "stale": stale,
        "current_current_null": null,
        "interpretation_boundary": {
            "population": (
                "task-stratified successful baseline rollouts with one uniformly "
                "sampled noninitial policy decision"
            ),
            "stale_fault": (
                "the main policy camera is one policy decision, or five environment "
                "steps, old; wrist camera and robot state remain current"
            ),
            "freshness": (
                "shadow detection only; source metadata is a retained-metadata "
                "counterfactual and exact duplication does not require metadata"
            ),
            "safe": (
                "SAFE-MLP is trained only on clean pi0.5 outcomes and frozen before "
                "these paired interventions"
            ),
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired pi0.5 stale-camera and null campaigns."
    )
    parser.add_argument("--stale-dir", required=True, type=Path)
    parser.add_argument("--stale-scores", required=True, type=Path)
    parser.add_argument("--null-dir", type=Path)
    parser.add_argument("--null-scores", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = analyze(
        args.stale_dir,
        args.stale_scores,
        null_dir=args.null_dir,
        null_scores=args.null_scores,
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
