from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256


PRIMARY_ALPHA = "0.1"
SAFE_WINDOWS = (
    "within_5_steps",
    "within_10_steps",
    "within_25_steps",
    "post_fault_any",
)

# score_safe.py records these hashes from the accepted seed-0 SAFE-MLP artifacts.
# Pinning all four prevents the discarded LSTM or a retrained MLP from being
# silently substituted in this comparison.
FROZEN_SAFE_MLP = {
    "checkpoint_sha256": (
        "2aec4590b4370808e6154ef60206cb5daf383ade53a672be84bb699a9ce2c031"
    ),
    "configuration_sha256": (
        "2b447944d0218278c47918777dbf5777b5cf29a207a73231e89161abd9dcd4c6"
    ),
    "split_manifest_sha256": (
        "a269cc28adc73834b25cd1f506f7c147ccc95e92eee93542e523db326177b136"
    ),
    "clean_score_archive_sha256": (
        "1ad81cca5249903ebbd3ac268f6511a0a3ba056429eee305af2b1d118dbe08f9"
    ),
}


def _trial(value: dict[str, Any]) -> tuple[int, int]:
    return int(value["task_id"]), int(value["episode_index"])


def _load_completions(
    directory: Path, condition: str
) -> dict[tuple[int, int], dict[str, Any]]:
    results: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(directory.rglob("*.complete.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("condition") != condition:
            raise ValueError(f"unexpected completion marker in {path}")
        key = _trial(value)
        if key in results:
            raise ValueError(f"duplicate completion marker for trial {key}")
        results[key] = value
    if not results:
        raise ValueError(f"no {condition} completion markers in {directory}")
    return results


def _require_equal(
    left: dict[str, Any], right: dict[str, Any], fields: tuple[str, ...], context: str
) -> None:
    for field in fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"{context} disagree on {field}")


def _validate_stale_control_pair(
    stale: dict[str, Any], current: dict[str, Any], context: str
) -> None:
    _require_equal(
        stale,
        current,
        (
            "task_id",
            "episode_index",
            "task_suite_name",
            "task_description",
            "initial_state_sha256",
            "trial_seed",
            "maximum_policy_steps",
        ),
        context,
    )
    stale_fault = stale["fault"]
    current_fault = current["fault"]
    if stale_fault.get("kind") != "stale_image":
        raise ValueError(f"{context} stale result has the wrong intervention")
    if current_fault.get("kind") != "current_image_control":
        raise ValueError(f"{context} control result has the wrong intervention")
    comparisons = (
        ("policy_step", "policy_step"),
        ("source_policy_step", "matched_stale_source_policy_step"),
        ("image_lag", "matched_stale_image_lag"),
        ("trial_seed", "trial_seed"),
    )
    for stale_field, current_field in comparisons:
        if stale_fault.get(stale_field) != current_fault.get(current_field):
            raise ValueError(
                f"{context} faults disagree on {stale_field}/{current_field}"
            )


def _validate_cross_campaign(
    outcome: dict[str, Any], evidence: dict[str, Any], context: str
) -> None:
    _require_equal(
        outcome,
        evidence,
        (
            "task_id",
            "episode_index",
            "task_suite_name",
            "task_description",
            "initial_state_sha256",
            "trial_seed",
            "maximum_policy_steps",
        ),
        context,
    )
    left = outcome["fault"]
    right = evidence["fault"]
    for field in ("kind", "policy_step", "trial_seed"):
        if left.get(field) != right.get(field):
            raise ValueError(f"{context} fault records disagree on {field}")
    if left["kind"] == "stale_image":
        fields = ("source_policy_step", "image_lag")
    else:
        fields = (
            "input_policy_step",
            "matched_stale_source_policy_step",
            "matched_stale_image_lag",
        )
    for field in fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"{context} fault records disagree on {field}")


def _validate_replay(result: dict[str, Any], context: str) -> None:
    replay = result.get("counterfactual_replay")
    if not isinstance(replay, dict) or replay.get("enabled") is not True:
        raise ValueError(f"{context} does not record enabled counterfactual replay")
    if int(replay.get("replayed_policy_steps", -1)) != int(
        result["fault"]["policy_step"]
    ):
        raise ValueError(f"{context} replay does not end at the intervention step")
    error = float(replay.get("maximum_numeric_observation_error", math.inf))
    tolerance = float(replay.get("observation_tolerance", -math.inf))
    if not math.isfinite(error) or not math.isfinite(tolerance) or error > tolerance:
        raise ValueError(f"{context} replay exceeds its observation tolerance")


def _load_safe_results(
    path: Path, condition: str
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 2:
        raise ValueError(f"unsupported SAFE result schema in {path}")
    monitor = value.get("monitor", {})
    for field, expected in FROZEN_SAFE_MLP.items():
        if monitor.get(field) != expected:
            raise ValueError(
                f"SAFE monitor {field} is {monitor.get(field)!r}, expected {expected!r}"
            )
    if not math.isclose(
        float(monitor.get("primary_alpha", -1)), float(PRIMARY_ALPHA), abs_tol=1e-8
    ):
        raise ValueError("SAFE result does not use the frozen primary alpha")

    records: dict[tuple[int, int], dict[str, Any]] = {}
    for record in value.get("records", []):
        if record.get("condition") != condition:
            raise ValueError(f"SAFE record has unexpected condition in {path}")
        key = _trial(record)
        if key in records:
            raise ValueError(f"duplicate SAFE record for trial {key}")
        if PRIMARY_ALPHA not in record.get("alarms", {}):
            raise ValueError(f"SAFE record has no alpha-{PRIMARY_ALPHA} alarms")
        records[key] = record
    if not records:
        raise ValueError(f"no SAFE records in {path}")
    return records, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "experiment_code_revision": value.get("experiment_code_revision"),
        "safe_revision": value.get("safe_revision"),
        "monitor": monitor,
        "alarm_rule": value.get("alarm_rule"),
    }


def _validate_safe_record(
    completion: dict[str, Any], record: dict[str, Any], context: str
) -> None:
    _require_equal(
        completion,
        record,
        ("task_id", "episode_index", "condition", "success"),
        context,
    )
    left = completion["fault"]
    right = record["fault"]
    fields = ["kind", "policy_step", "trial_seed"]
    if left["kind"] == "stale_image":
        fields.extend(("source_policy_step", "image_lag"))
    else:
        fields.extend(
            (
                "input_policy_step",
                "matched_stale_source_policy_step",
                "matched_stale_image_lag",
            )
        )
    for field in fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"{context} fault records disagree on {field}")


def freshness_alarms(result: dict[str, Any]) -> dict[str, bool]:
    evidence = result["fault"].get("freshness_at_intervention")
    if not isinstance(evidence, dict):
        raise ValueError(f"trial {_trial(result)} has no freshness intervention record")
    required_booleans = (
        "source_metadata_alarm",
        "relabelled_metadata_alarm",
        "exact_duplicate_alarm",
    )
    for field in required_booleans:
        if not isinstance(evidence.get(field), bool):
            raise ValueError(f"trial {_trial(result)} has invalid {field}")

    exact_duplicate = evidence.get("input_sha256") == evidence.get(
        "previous_input_sha256"
    )
    if not isinstance(evidence.get("input_sha256"), str) or not isinstance(
        evidence.get("previous_input_sha256"), str
    ):
        raise ValueError(f"trial {_trial(result)} has invalid image digests")
    if evidence["exact_duplicate_alarm"] != exact_duplicate:
        raise ValueError(
            f"trial {_trial(result)} duplicate alarm disagrees with hashes"
        )

    source_age = evidence.get("source_metadata_age_steps")
    if not isinstance(source_age, int) or source_age < 0:
        raise ValueError(f"trial {_trial(result)} has invalid source metadata age")
    if evidence["source_metadata_alarm"] != (source_age > 0):
        raise ValueError(f"trial {_trial(result)} source alarm disagrees with age")

    return {
        "source_metadata": evidence["source_metadata_alarm"],
        "relabeled_metadata": evidence["relabelled_metadata_alarm"],
        "exact_duplicate": evidence["exact_duplicate_alarm"],
    }


def safe_alarms(record: dict[str, Any]) -> dict[str, bool]:
    alarms = record["alarms"][PRIMARY_ALPHA]
    result = {}
    for window in SAFE_WINDOWS:
        window_value = alarms.get(window)
        if not isinstance(window_value, dict) or not isinstance(
            window_value.get("triggered"), bool
        ):
            raise ValueError(f"SAFE record for {_trial(record)} has invalid {window}")
        result[window] = window_value["triggered"]
    return result


def load_verified_inputs(
    stale_dir: Path,
    current_dir: Path,
    safe_stale_path: Path,
    safe_current_path: Path,
    freshness_stale_dir: Path,
    freshness_current_dir: Path,
) -> dict[str, Any]:
    stale = _load_completions(stale_dir, "stale_image")
    current = _load_completions(current_dir, "current_image_control")
    outcome_trials = sorted(set(stale) & set(current))
    if not outcome_trials:
        raise ValueError("stale and current outcome campaigns have no paired trials")

    freshness_stale = _load_completions(freshness_stale_dir, "stale_image")
    freshness_current = _load_completions(
        freshness_current_dir, "current_image_control"
    )
    freshness_pairs = set(freshness_stale) & set(freshness_current)
    missing_freshness = sorted(set(outcome_trials) - freshness_pairs)
    if missing_freshness:
        raise ValueError(
            f"{len(missing_freshness)} outcome pairs lack freshness evidence: "
            f"{missing_freshness[:5]}"
        )

    safe_stale, safe_stale_provenance = _load_safe_results(
        safe_stale_path, "stale_image"
    )
    safe_current, safe_current_provenance = _load_safe_results(
        safe_current_path, "current_image_control"
    )
    missing_safe = sorted(
        set(outcome_trials) - (set(safe_stale) & set(safe_current))
    )
    if missing_safe:
        raise ValueError(
            f"{len(missing_safe)} outcome pairs lack SAFE-MLP evidence: "
            f"{missing_safe[:5]}"
        )

    for trial in outcome_trials:
        for label, result in (
            ("outcome stale", stale[trial]),
            ("outcome control", current[trial]),
            ("freshness stale", freshness_stale[trial]),
            ("freshness control", freshness_current[trial]),
        ):
            _validate_replay(result, f"{label} {trial}")
        _validate_stale_control_pair(stale[trial], current[trial], f"outcome {trial}")
        _validate_stale_control_pair(
            freshness_stale[trial], freshness_current[trial], f"freshness {trial}"
        )
        _validate_cross_campaign(
            stale[trial], freshness_stale[trial], f"stale campaigns {trial}"
        )
        _validate_cross_campaign(
            current[trial], freshness_current[trial], f"control campaigns {trial}"
        )
        _validate_safe_record(stale[trial], safe_stale[trial], f"stale SAFE {trial}")
        _validate_safe_record(
            current[trial], safe_current[trial], f"control SAFE {trial}"
        )
        freshness_alarms(freshness_stale[trial])
        freshness_alarms(freshness_current[trial])
        safe_alarms(safe_stale[trial])
        safe_alarms(safe_current[trial])

    return {
        "stale": stale,
        "current": current,
        "freshness_stale": freshness_stale,
        "freshness_current": freshness_current,
        "safe_stale": safe_stale,
        "safe_current": safe_current,
        "outcome_trials": outcome_trials,
        "audit": {
            "outcome_pairs": len(outcome_trials),
            "unpaired_outcome_stale": len(set(stale) - set(current)),
            "unpaired_outcome_current": len(set(current) - set(stale)),
            "freshness_pairs": len(freshness_pairs),
            "outcome_pairs_with_freshness": len(outcome_trials),
            "outcome_pairs_with_safe_mlp": len(outcome_trials),
            "freshness_pairs_not_used_for_outcomes": len(
                freshness_pairs - set(outcome_trials)
            ),
        },
        "provenance": {
            "safe_stale": safe_stale_provenance,
            "safe_current": safe_current_provenance,
            "frozen_safe_mlp_identity": FROZEN_SAFE_MLP,
        },
    }
