import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


PRESERVED_SUCCESS = "preserved_success"
FALSE_ALARM = "false_alarm"
DETECTED_FAULT_FAILURE = "detected_fault_failure"
SILENT_FAULT_FAILURE = "silent_fault_failure"
BASELINE_FAILURE = "baseline_failure"


@dataclass(frozen=True)
class Alarm:
    triggered: bool
    first_step: int | None


@dataclass(frozen=True)
class PairedOutcome:
    task_id: int
    episode_index: int
    category: str
    clean_success: bool
    fault_success: bool
    alarm: bool
    first_alarm_step: int | None
    fault: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def alarm_from_scores(scores: Sequence[float], threshold: float) -> Alarm:
    if not scores:
        raise ValueError("monitor scores cannot be empty")
    if not math.isfinite(threshold):
        raise ValueError("monitor threshold must be finite")

    for step, score in enumerate(scores):
        if not math.isfinite(score):
            raise ValueError(f"monitor score at step {step} is not finite")
        if score >= threshold:
            return Alarm(triggered=True, first_step=step)
    return Alarm(triggered=False, first_step=None)


def classify_pair(
    clean_result: dict[str, Any],
    fault_result: dict[str, Any],
    alarm: Alarm,
) -> PairedOutcome:
    for key in ("task_id", "episode_index", "initial_state_sha256"):
        if clean_result.get(key) != fault_result.get(key):
            raise ValueError(f"paired results disagree on {key}")
    if clean_result.get("condition") != "clean":
        raise ValueError("the paired reference is not a clean rollout")
    if fault_result.get("condition") != "activation_fault":
        raise ValueError("the paired treatment is not an activation-fault rollout")

    fault = fault_result.get("fault")
    if not isinstance(fault, dict):
        raise ValueError("fault rollout has no injection record")
    if fault.get("trial_seed") != fault_result.get("trial_seed"):
        raise ValueError("fault record and rollout disagree on trial seed")

    clean_success = bool(clean_result["success"])
    fault_success = bool(fault_result["success"])
    if not clean_success:
        category = BASELINE_FAILURE
    elif fault_success and alarm.triggered:
        category = FALSE_ALARM
    elif fault_success:
        category = PRESERVED_SUCCESS
    elif alarm.triggered:
        category = DETECTED_FAULT_FAILURE
    else:
        category = SILENT_FAULT_FAILURE

    return PairedOutcome(
        task_id=int(fault_result["task_id"]),
        episode_index=int(fault_result["episode_index"]),
        category=category,
        clean_success=clean_success,
        fault_success=fault_success,
        alarm=alarm.triggered,
        first_alarm_step=alarm.first_step,
        fault=fault,
    )


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    estimate = successes / trials
    denominator = 1 + z * z / trials
    center = (estimate + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1 - estimate) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator if denominator else None,
        "wilson_95": _wilson_interval(numerator, denominator),
    }


def summarize_outcomes(outcomes: Iterable[PairedOutcome]) -> dict[str, Any]:
    values = list(outcomes)
    counts = Counter(outcome.category for outcome in values)
    eligible = sum(outcome.clean_success for outcome in values)
    fault_failures = counts[DETECTED_FAULT_FAILURE] + counts[SILENT_FAULT_FAILURE]
    fault_successes = counts[PRESERVED_SUCCESS] + counts[FALSE_ALARM]

    return {
        "pairs": len(values),
        "eligible_clean_successes": eligible,
        "excluded_baseline_failures": counts[BASELINE_FAILURE],
        "categories": dict(sorted(counts.items())),
        "policy_vulnerability": _rate(fault_failures, eligible),
        "monitor_miss_given_fault_failure": _rate(
            counts[SILENT_FAULT_FAILURE], fault_failures
        ),
        "residual_silent_risk": _rate(counts[SILENT_FAULT_FAILURE], eligible),
        "alarm_on_faulted_success": _rate(counts[FALSE_ALARM], fault_successes),
    }


def summarize_by_fault_field(
    outcomes: Iterable[PairedOutcome], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[PairedOutcome]] = {}
    for outcome in outcomes:
        if field not in outcome.fault:
            raise ValueError(f"fault record has no {field}")
        key = str(outcome.fault[field])
        groups.setdefault(key, []).append(outcome)
    return {key: summarize_outcomes(groups[key]) for key in sorted(groups)}
