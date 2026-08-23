from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.analyze_freshness import (
    _exact_binomial_two_sided,
    _wilson,
)
from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.stale_monitor_inputs import (
    SAFE_WINDOWS,
    freshness_alarms,
    load_verified_inputs,
    safe_alarms,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen SAFE-MLP with simple freshness checks on matched "
            "one-frame stale-image failures."
        )
    )
    parser.add_argument("--stale-dir", required=True, type=Path)
    parser.add_argument("--current-control-dir", required=True, type=Path)
    parser.add_argument("--safe-stale-results", required=True, type=Path)
    parser.add_argument("--safe-current-results", required=True, type=Path)
    parser.add_argument("--freshness-stale-dir", required=True, type=Path)
    parser.add_argument("--freshness-current-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _rate(count: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "total": total,
        "rate": count / total if total else None,
        "wilson_95": _wilson(count, total),
    }


def compare(
    stale_dir: Path,
    current_dir: Path,
    safe_stale_path: Path,
    safe_current_path: Path,
    freshness_stale_dir: Path,
    freshness_current_dir: Path,
) -> dict[str, Any]:
    inputs = load_verified_inputs(
        stale_dir,
        current_dir,
        safe_stale_path,
        safe_current_path,
        freshness_stale_dir,
        freshness_current_dir,
    )
    stale = inputs["stale"]
    current = inputs["current"]
    freshness_stale = inputs["freshness_stale"]
    freshness_current = inputs["freshness_current"]
    safe_stale = inputs["safe_stale"]
    safe_current = inputs["safe_current"]
    outcome_trials = inputs["outcome_trials"]

    both_success = stale_only_failure = current_only_failure = both_failure = 0
    causal_trials: list[tuple[int, int]] = []
    for trial in outcome_trials:
        stale_success = bool(stale[trial]["success"])
        current_success = bool(current[trial]["success"])
        if stale_success and current_success:
            both_success += 1
        elif not stale_success and current_success:
            stale_only_failure += 1
            causal_trials.append(trial)
        elif stale_success and not current_success:
            current_only_failure += 1
        else:
            both_failure += 1

    freshness_names = ("source_metadata", "relabeled_metadata", "exact_duplicate")
    freshness_detection = {name: 0 for name in freshness_names}
    safe_detection = {window: 0 for window in SAFE_WINDOWS}
    causal_records = []
    for trial in causal_trials:
        freshness = freshness_alarms(freshness_stale[trial])
        safe = safe_alarms(safe_stale[trial])
        for name, alarm in freshness.items():
            freshness_detection[name] += int(alarm)
        for window, alarm in safe.items():
            safe_detection[window] += int(alarm)
        causal_records.append(
            {
                "task_id": trial[0],
                "episode_index": trial[1],
                "policy_step": int(stale[trial]["fault"]["policy_step"]),
                "freshness_alarms": freshness,
                "safe_mlp_alarms": safe,
            }
        )

    current_freshness_intervention = {name: 0 for name in freshness_names}
    current_freshness_all_steps = {name: 0 for name in freshness_names}
    current_policy_steps = 0
    safe_current_alarms = {window: 0 for window in SAFE_WINDOWS}
    for trial in outcome_trials:
        intervention = freshness_alarms(freshness_current[trial])
        for name, alarm in intervention.items():
            current_freshness_intervention[name] += int(alarm)
        summary = freshness_current[trial]["freshness"]
        current_policy_steps += int(summary["evaluated_policy_steps"])
        current_freshness_all_steps["source_metadata"] += int(
            summary["source_metadata_alarms"]
        )
        current_freshness_all_steps["relabeled_metadata"] += int(
            summary["relabelled_metadata_alarms"]
        )
        current_freshness_all_steps["exact_duplicate"] += int(
            summary["exact_duplicate_alarms"]
        )
        for window, alarm in safe_alarms(safe_current[trial]).items():
            safe_current_alarms[window] += int(alarm)

    detector_comparisons = {}
    for name in freshness_names:
        freshness_only = sum(
            record["freshness_alarms"][name]
            and not record["safe_mlp_alarms"]["within_25_steps"]
            for record in causal_records
        )
        safe_only = sum(
            not record["freshness_alarms"][name]
            and record["safe_mlp_alarms"]["within_25_steps"]
            for record in causal_records
        )
        detector_comparisons[f"{name}_vs_safe_mlp_within_25_steps"] = {
            "freshness_only": freshness_only,
            "safe_mlp_only": safe_only,
            "exact_mcnemar_two_sided_p": _exact_binomial_two_sided(
                freshness_only, safe_only
            ),
        }

    eligible_controls = sum(bool(current[trial]["success"]) for trial in outcome_trials)
    return {
        "schema_version": 1,
        "question": (
            "Are exact one-frame stale-image failures intrinsically hard to detect, "
            "or are they a blind spot of frozen SAFE-MLP?"
        ),
        "outcome_population": {
            "paired_trials": len(outcome_trials),
            "eligible_successful_controls": eligible_controls,
            "both_success": both_success,
            "stale_only_failure": stale_only_failure,
            "current_control_only_failure": current_only_failure,
            "both_failure": both_failure,
            "stale_effect_exact_mcnemar_two_sided_p": _exact_binomial_two_sided(
                stale_only_failure, current_only_failure
            ),
            "causal_stale_failure_rate": _rate(len(causal_trials), eligible_controls),
        },
        "causal_failure_detection": {
            "trials": len(causal_trials),
            "freshness_at_intervention": {
                name: _rate(count, len(causal_trials))
                for name, count in freshness_detection.items()
            },
            "safe_mlp_alpha_0_1": {
                window: _rate(count, len(causal_trials))
                for window, count in safe_detection.items()
            },
            "paired_comparisons": detector_comparisons,
        },
        "matched_current_control_alarms": {
            "trials": len(outcome_trials),
            "freshness_at_intervention": {
                name: _rate(count, len(outcome_trials))
                for name, count in current_freshness_intervention.items()
            },
            "freshness_over_all_policy_steps": {
                "policy_steps": current_policy_steps,
                **{
                    name: _rate(count, current_policy_steps)
                    for name, count in current_freshness_all_steps.items()
                },
            },
            "safe_mlp_alpha_0_1": {
                window: _rate(count, len(outcome_trials))
                for window, count in safe_current_alarms.items()
            },
        },
        "causal_failure_trials": causal_records,
        "audit": inputs["audit"],
        "provenance": inputs["provenance"],
        "interpretation_boundary": {
            "outcomes": (
                "Task outcomes come only from the original no-response stale and "
                "current-image campaigns. Outcomes from the later hold-response "
                "campaign are not analyzed."
            ),
            "freshness_evidence": (
                "The later campaign contributes only shadow freshness observations "
                "recorded before its hold response at the same seed, state, and step."
            ),
            "source_metadata": (
                "The source-step field is a simulator proxy for metadata that remains "
                "bound to the pixels; it is not a measured camera timestamp pipeline."
            ),
            "relabeled_metadata": (
                "This counterfactual assigns current metadata to old pixels and shows "
                "what timestamp or frame-ID checks miss after downstream relabeling."
            ),
            "exact_duplicate": (
                "SHA-256 equality is exact for this one-frame replay in LIBERO; it "
                "does not establish robustness to compression, sensor noise, or "
                "near-duplicate real camera frames."
            ),
            "safe_mlp": (
                "Only the accepted frozen seed-0 SAFE-MLP is compared. SAFE-LSTM is "
                "excluded because its perfect training fit and weak unseen-task "
                "performance made it an invalid comparator in this study."
            ),
            "inference": (
                "The comparison supports a conclusion about this predeclared exact "
                "lag-1 fault class, not stale observations in general."
            ),
        },
    }


def main() -> None:
    args = _parse_arguments()
    result = compare(
        args.stale_dir,
        args.current_control_dir,
        args.safe_stale_results,
        args.safe_current_results,
        args.freshness_stale_dir,
        args.freshness_current_dir,
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
