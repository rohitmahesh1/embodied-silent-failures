import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import file_sha256


BOOTSTRAP_SEED = 20260813
BOOTSTRAP_SAMPLES = 100_000


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired Qwen alarm timelines around an intervention."
    )
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-source-videos", action="store_true")
    return parser.parse_args()


def _load_trials(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    values = []
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            raise ValueError(f"Qwen trial is not complete: {path}")
        values.append((path, value))
    if not values:
        raise ValueError(f"no Qwen trials found in {directory}")
    return values


def _pair_trials(
    trials: Iterable[tuple[Path, dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for _path, trial in trials:
        key = (int(trial["task_id"]), int(trial["episode_index"]))
        source = trial.get("source")
        if source not in {"stale", "control"}:
            raise ValueError(f"unsupported paired Qwen source: {source}")
        if source in grouped[key]:
            raise ValueError(f"duplicate {source} trial for {key}")
        grouped[key][source] = trial
    incomplete = [key for key, pair in grouped.items() if set(pair) != {"stale", "control"}]
    if incomplete:
        raise ValueError(f"incomplete Qwen pairs: {incomplete}")
    return [(grouped[key]["stale"], grouped[key]["control"]) for key in sorted(grouped)]


def _timeline(trial: dict[str, Any]) -> dict[int, dict[str, Any]]:
    timeline = trial.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("Qwen trial has no timeline")
    steps = [int(item["policy_step"]) for item in timeline]
    if steps != sorted(set(steps)) or steps != trial.get("expected_query_steps"):
        raise ValueError("Qwen timeline is not the complete expected query sequence")
    return dict(zip(steps, timeline, strict=True))


def _source_pairing_audit(
    stale: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    completions = []
    for trial in (stale, control):
        path = Path(trial["completion_path"])
        if not path.is_file() or file_sha256(path) != trial["completion_sha256"]:
            raise ValueError(f"source completion is missing or changed: {path}")
        completions.append(json.loads(path.read_text(encoding="utf-8")))
    stale_completion, control_completion = completions
    fault_step = int(stale["fault"]["policy_step"])
    fields = ("task_id", "episode_index", "initial_state_sha256", "trial_seed")
    if any(stale_completion.get(key) != control_completion.get(key) for key in fields):
        raise ValueError("paired source completions disagree on trial identity")
    if int(control["fault"]["policy_step"]) != fault_step:
        raise ValueError("paired Qwen trials disagree on intervention step")
    if stale_completion.get("success") is not False or control_completion.get("success") is not True:
        raise ValueError("causal cohort is not stale-failure/control-success")
    replay = [item.get("counterfactual_replay") for item in completions]
    if any(not isinstance(item, dict) or item.get("enabled") is not True for item in replay):
        raise ValueError("paired source rollout did not use counterfactual prefix replay")
    if any(int(item["replayed_policy_steps"]) != fault_step for item in replay):
        raise ValueError("counterfactual prefix does not end at the intervention")
    if any(float(item["maximum_numeric_observation_error"]) > float(item["observation_tolerance"]) for item in replay):
        raise ValueError("counterfactual prefix exceeded its observation tolerance")
    return {
        "initial_state_sha256": stale_completion["initial_state_sha256"],
        "trial_seed": int(stale_completion["trial_seed"]),
        "replayed_policy_steps": fault_step,
        "maximum_numeric_observation_error": max(
            float(item["maximum_numeric_observation_error"]) for item in replay
        ),
        "observation_tolerance": max(float(item["observation_tolerance"]) for item in replay),
    }


def _alarm_difference(
    stale: dict[int, dict[str, Any]],
    control: dict[int, dict[str, Any]],
    steps: Iterable[int],
) -> tuple[float, int, int]:
    differences = []
    invalid = 0
    for step in steps:
        left = stale[step].get("alarm")
        right = control[step].get("alarm")
        if left is None or right is None:
            invalid += 1
            continue
        differences.append(int(left) - int(right))
    if not differences:
        raise ValueError("paired time window has no determinate Qwen comparisons")
    return sum(differences) / len(differences), len(differences), invalid


def _paired_alarm_counts(
    stale: dict[int, dict[str, Any]],
    control: dict[int, dict[str, Any]],
    steps: Iterable[int],
) -> Counter[tuple[bool | None, bool | None]]:
    return Counter(
        (stale[step].get("alarm"), control[step].get("alarm")) for step in steps
    )


def _first_alarm(
    timeline: dict[int, dict[str, Any]], steps: Iterable[int]
) -> int | None:
    return next(
        (step for step in steps if timeline[step].get("alarm") is True), None
    )


def _exact_binomial_two_sided(left: int, right: int) -> float | None:
    trials = left + right
    if trials == 0:
        return None
    tail = sum(math.comb(trials, value) for value in range(min(left, right) + 1))
    return min(1.0, 2 * tail / (2**trials))


def _exact_sign_flip(values: list[float]) -> dict[str, Any]:
    nonzero = [value for value in values if value != 0]
    if len(nonzero) > 22:
        raise ValueError("exact sign-flip test is limited to 22 nonzero pairs")
    observed = abs(sum(values) / len(values))
    extreme = 0
    assignments = 1 << len(nonzero)
    for mask in range(assignments):
        total = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(nonzero)
        )
        if abs(total / len(values)) + 1e-15 >= observed:
            extreme += 1
    return {
        "method": "exact paired sign-flip reference distribution",
        "nonzero_pairs": len(nonzero),
        "assignments": assignments,
        "two_sided_p": extreme / assignments,
    }


def _bootstrap_mean_interval(values: list[float]) -> dict[str, Any]:
    generator = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    estimates = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    low = estimates[round(0.025 * (BOOTSTRAP_SAMPLES - 1))]
    high = estimates[round(0.975 * (BOOTSTRAP_SAMPLES - 1))]
    return {
        "method": "percentile pair bootstrap",
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "confidence": 0.95,
        "interval": [low, high],
    }


def _pixel_audit(stale: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import numpy as np

    captures = [cv2.VideoCapture(str(Path(item["video_path"]))) for item in (stale, control)]
    if any(not capture.isOpened() for capture in captures):
        raise RuntimeError("cannot open one or more paired rollout videos")
    fault_step = int(stale["fault"]["policy_step"])
    absolute_sum = 0
    squared_sum = 0
    channel_values = 0
    unequal_values = 0
    maximum = 0
    unequal_frames = 0
    try:
        for policy_step in range(fault_step + 1):
            decoded = [capture.read() for capture in captures]
            if any(not readable for readable, _frame in decoded):
                raise ValueError("paired video ended before the intervention")
            left = decoded[0][1].astype(np.int16)
            right = decoded[1][1].astype(np.int16)
            difference = np.abs(left - right)
            absolute_sum += int(difference.sum())
            squared_sum += int(np.square(difference.astype(np.int32)).sum())
            channel_values += int(difference.size)
            unequal = int(np.count_nonzero(difference))
            unequal_values += unequal
            unequal_frames += unequal > 0
            maximum = max(maximum, int(difference.max()))
    finally:
        for capture in captures:
            capture.release()
    mean_absolute_error = absolute_sum / channel_values
    root_mean_squared_error = math.sqrt(squared_sum / channel_values)
    return {
        "decoded_frames": fault_step + 1,
        "unequal_frames": unequal_frames,
        "unequal_channel_fraction": unequal_values / channel_values,
        "mean_absolute_channel_error": mean_absolute_error,
        "root_mean_squared_channel_error": root_mean_squared_error,
        "maximum_absolute_channel_error": maximum,
        "psnr_db": (
            math.inf
            if root_mean_squared_error == 0
            else 20 * math.log10(255 / root_mean_squared_error)
        ),
    }


def analyze(trial_dir: Path, *, audit_source_videos: bool = False) -> dict[str, Any]:
    trials = _load_trials(trial_dir)
    pairs = _pair_trials(trials)
    configuration_hashes = {trial["configuration_sha256"] for _path, trial in trials}
    if len(configuration_hashes) != 1:
        raise ValueError("paired Qwen trials do not share one scoring configuration")

    pair_records = []
    relative: dict[
        int, list[tuple[bool | None, bool | None, int]]
    ] = defaultdict(list)
    first_exposed = Counter()
    pre_alarm_counts: Counter[tuple[bool | None, bool | None]] = Counter()
    post_alarm_counts: Counter[tuple[bool | None, bool | None]] = Counter()
    first_alarm_status = {"stale": Counter(), "control": Counter()}
    incident_pair_onset = Counter()
    incident_delays: dict[str, list[int]] = {"stale": [], "control": []}
    difference_in_differences = []
    pre_frame_hash_mismatches = 0
    pre_frame_hash_comparisons = 0
    pixel_audits = []
    for stale, control in pairs:
        pairing = _source_pairing_audit(stale, control)
        stale_timeline = _timeline(stale)
        control_timeline = _timeline(control)
        common_steps = sorted(stale_timeline.keys() & control_timeline.keys())
        fault_step = int(stale["fault"]["policy_step"])
        pre_steps = [step for step in common_steps if step <= fault_step]
        # run_openvla.py::_run_trial at commit 2f3bfa2 records the camera image
        # before env.step, so the first Qwen frame that can show the changed
        # action's physical consequence is the first query strictly after fault_step.
        post_steps = [step for step in common_steps if step > fault_step]
        if not pre_steps or not post_steps:
            raise ValueError("paired Qwen trial has no common pre/post query support")

        pre_difference, pre_count, pre_invalid = _alarm_difference(
            stale_timeline, control_timeline, pre_steps
        )
        post_difference, post_count, post_invalid = _alarm_difference(
            stale_timeline, control_timeline, post_steps
        )
        change = post_difference - pre_difference
        difference_in_differences.append(change)
        pre_alarm_counts.update(
            _paired_alarm_counts(stale_timeline, control_timeline, pre_steps)
        )
        post_alarm_counts.update(
            _paired_alarm_counts(stale_timeline, control_timeline, post_steps)
        )

        first_alarms = {
            "stale": _first_alarm(stale_timeline, common_steps),
            "control": _first_alarm(control_timeline, common_steps),
        }
        pre_has_alarm = {
            "stale": _first_alarm(stale_timeline, pre_steps) is not None,
            "control": _first_alarm(control_timeline, pre_steps) is not None,
        }
        for source, timeline in (("stale", stale_timeline), ("control", control_timeline)):
            first = first_alarms[source]
            if first is None:
                first_alarm_status[source]["none_on_common_support"] += 1
            elif first <= fault_step:
                first_alarm_status[source]["before_or_at_intervention"] += 1
            else:
                first_alarm_status[source]["after_intervention"] += 1
                incident_delays[source].append(first - fault_step)
        if not pre_has_alarm["stale"] and not pre_has_alarm["control"]:
            left, right = first_alarms["stale"], first_alarms["control"]
            if left is None and right is None:
                incident_pair_onset["neither"] += 1
            elif left is None:
                incident_pair_onset["control_only"] += 1
            elif right is None:
                incident_pair_onset["stale_only"] += 1
            elif left < right:
                incident_pair_onset["stale_earlier"] += 1
            elif right < left:
                incident_pair_onset["control_earlier"] += 1
            else:
                incident_pair_onset["same_query"] += 1

        mismatches = sum(
            stale_timeline[step]["frame_sha256"][-1]
            != control_timeline[step]["frame_sha256"][-1]
            for step in pre_steps
        )
        pre_frame_hash_mismatches += mismatches
        pre_frame_hash_comparisons += len(pre_steps)
        for offset, step in enumerate(post_steps, start=1):
            relative[offset].append(
                (
                    stale_timeline[step].get("alarm"),
                    control_timeline[step].get("alarm"),
                    step - fault_step,
                )
            )
        immediate = (
            stale_timeline[post_steps[0]].get("alarm"),
            control_timeline[post_steps[0]].get("alarm"),
        )
        first_exposed[immediate] += 1

        pixel = _pixel_audit(stale, control) if audit_source_videos else None
        if pixel is not None:
            pixel_audits.append(pixel)
        pair_records.append(
            {
                "task_id": int(stale["task_id"]),
                "episode_index": int(stale["episode_index"]),
                "fault_policy_step": fault_step,
                "first_exposed_query_step": post_steps[0],
                "last_common_query_step": common_steps[-1],
                "pre_determinate_queries": pre_count,
                "post_determinate_queries": post_count,
                "pre_indeterminate_queries": pre_invalid,
                "post_indeterminate_queries": post_invalid,
                "pre_stale_minus_control_alarm_fraction": pre_difference,
                "post_stale_minus_control_alarm_fraction": post_difference,
                "difference_in_differences": change,
                "pre_latest_frame_hash_mismatches": mismatches,
                "source_pairing": pairing,
                **({"pre_intervention_pixel_audit": pixel} if pixel is not None else {}),
            }
        )

    relative_queries = []
    for offset, observations in sorted(relative.items()):
        counts = Counter(
            (left, right)
            for left, right, _policy_step_offset in observations
            if left is not None and right is not None
        )
        policy_step_offsets = [item[2] for item in observations]
        determinate = sum(counts.values())
        relative_queries.append(
            {
                "query_offset_after_intervention": offset,
                "policy_step_offset_range": [
                    min(policy_step_offsets),
                    max(policy_step_offsets),
                ],
                "supported_pairs": len(observations),
                "determinate_pairs": determinate,
                "both_alarm": counts[(True, True)],
                "stale_only_alarm": counts[(True, False)],
                "control_only_alarm": counts[(False, True)],
                "neither_alarm": counts[(False, False)],
                "stale_minus_control_alarm_rate": (
                    (counts[(True, False)] - counts[(False, True)]) / determinate
                    if determinate
                    else None
                ),
            }
        )

    immediate_stale_only = first_exposed[(True, False)]
    immediate_control_only = first_exposed[(False, True)]
    input_manifest = [
        {"file": path.name, "sha256": file_sha256(path)} for path, _trial in trials
    ]
    input_manifest_sha256 = hashlib.sha256(
        json.dumps(input_manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    estimate = sum(difference_in_differences) / len(difference_in_differences)
    signs = Counter(
        "positive" if value > 0 else "negative" if value < 0 else "zero"
        for value in difference_in_differences
    )

    def alarm_count_summary(
        counts: Counter[tuple[bool | None, bool | None]]
    ) -> dict[str, int]:
        return {
            "both_alarm": counts[(True, True)],
            "stale_only_alarm": counts[(True, False)],
            "control_only_alarm": counts[(False, True)],
            "neither_alarm": counts[(False, False)],
            "indeterminate": sum(count for key, count in counts.items() if None in key),
        }

    def delay_summary(values: list[int]) -> dict[str, Any]:
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            None
            if not ordered
            else ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "count": len(ordered),
            "median_policy_steps": median,
            "minimum_policy_steps": ordered[0] if ordered else None,
            "maximum_policy_steps": ordered[-1] if ordered else None,
        }
    return {
        "schema_version": 1,
        "analysis": "paired Qwen temporal response around stale-image intervention",
        "source": {
            "configuration_sha256": next(iter(configuration_hashes)),
            "input_manifest_sha256": input_manifest_sha256,
            "trial_count": len(trials),
            "pair_count": len(pairs),
        },
        "design": {
            "unit": "matched task, episode, initial state, and trial seed",
            "pre_period": "common Qwen query steps at or before the intervention",
            "post_period": "common Qwen query steps strictly after the intervention",
            "common_support": "each pair contributes equally and only while both trajectories exist",
            "primary_estimand": (
                "pair-mean change from pre to post in stale-minus-control alarm fraction"
            ),
            "selection_note": (
                "the cohort was selected to contain stale failures with successful controls; "
                "it estimates monitor response in this mechanism-enriched cohort, not prevalence"
            ),
        },
        "pairing_audit": {
            "all_source_completions_match_recorded_hashes": True,
            "all_initial_states_and_trial_seeds_match": True,
            "all_prefixes_replayed_to_the_intervention_within_tolerance": True,
            "pre_latest_frame_hash_comparisons": pre_frame_hash_comparisons,
            "pre_latest_frame_hash_mismatches": pre_frame_hash_mismatches,
            "decoded_video_note": (
                "Qwen consumed separately encoded AVC video; decoded hashes can differ even "
                "when source rollout observations match"
            ),
            **(
                {
                    "pre_intervention_pixel_audit": {
                        "pair_count": len(pixel_audits),
                        "mean_pair_mae": sum(
                            item["mean_absolute_channel_error"] for item in pixel_audits
                        )
                        / len(pixel_audits),
                        "maximum_pair_mae": max(
                            item["mean_absolute_channel_error"] for item in pixel_audits
                        ),
                        "maximum_absolute_channel_error": max(
                            item["maximum_absolute_channel_error"] for item in pixel_audits
                        ),
                        "minimum_psnr_db": min(item["psnr_db"] for item in pixel_audits),
                    }
                }
                if pixel_audits
                else {}
            ),
        },
        "primary_result": {
            "estimate": estimate,
            "pair_directions": dict(sorted(signs.items())),
            "confidence_interval": _bootstrap_mean_interval(difference_in_differences),
            "permutation_reference": _exact_sign_flip(difference_in_differences),
            "interpretation": (
                "positive values mean the stale condition's alarm fraction increased more "
                "from pre to post than the control condition's"
            ),
        },
        "first_exposed_query": {
            "both_alarm": first_exposed[(True, True)],
            "stale_only_alarm": immediate_stale_only,
            "control_only_alarm": immediate_control_only,
            "neither_alarm": first_exposed[(False, False)],
            "indeterminate": sum(
                count for key, count in first_exposed.items() if None in key
            ),
            "exact_mcnemar_two_sided_p": _exact_binomial_two_sided(
                immediate_stale_only, immediate_control_only
            ),
        },
        "query_level_negative_control": alarm_count_summary(pre_alarm_counts),
        "query_level_post_intervention": alarm_count_summary(post_alarm_counts),
        "first_alarm_timing_on_common_support": {
            "stale": {
                **dict(sorted(first_alarm_status["stale"].items())),
                "incident_delay": delay_summary(incident_delays["stale"]),
            },
            "control": {
                **dict(sorted(first_alarm_status["control"].items())),
                "incident_delay": delay_summary(incident_delays["control"]),
            },
            "pairs_with_no_pre_intervention_alarm": {
                "pair_count": sum(incident_pair_onset.values()),
                **dict(sorted(incident_pair_onset.items())),
                "note": "Descriptive secondary result; no hypothesis test was assigned.",
            },
        },
        "relative_queries": relative_queries,
        "pairs": pair_records,
        "inference_note": (
            "The bootstrap and sign-flip reference use pairs, not queries. The reference "
            "p-value assumes exchangeable condition labels for monitor response within this "
            "selected cohort; relative-query results are descriptive and untested."
        ),
    }


def main() -> None:
    args = _parse_arguments()
    result = analyze(
        args.trial_dir.resolve(), audit_source_videos=args.audit_source_videos
    )
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result["primary_result"], sort_keys=True))


if __name__ == "__main__":
    main()
