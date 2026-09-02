from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable


ALARM_HORIZONS = {
    "at_intervention": 1,
    "within_5_steps": 5,
    "within_10_steps": 10,
    "within_25_steps": 25,
    "post_fault_any": None,
}

CELL_NAMES = (
    "clean_action_clean_evidence",
    "clean_action_faulted_evidence",
    "faulted_action_clean_evidence",
    "faulted_action_faulted_evidence",
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


def alarm_summary(np: Any, scores: Any, band: Any, step: int) -> dict[str, Any]:
    scores = np.asarray(scores)
    band = np.asarray(band)
    if scores.ndim != 1 or band.ndim != 1:
        raise ValueError("SAFE scores and threshold band must be one-dimensional")
    if not 0 <= step < len(scores):
        raise ValueError("intervention step is outside the SAFE score trace")
    if len(band) < len(scores):
        raise ValueError("SAFE threshold band is shorter than the score trace")

    alarms = {}
    for name, horizon in ALARM_HORIZONS.items():
        stop = len(scores) if horizon is None else min(len(scores), step + horizon)
        candidates = np.flatnonzero(scores[step:stop] >= band[step:stop])
        alarms[name] = {
            "triggered": bool(len(candidates)),
            "first_step": int(step + candidates[0]) if len(candidates) else None,
        }
    return {
        "score_at_intervention": float(scores[step]),
        "threshold_at_intervention": float(band[step]),
        "margin_at_intervention": float(scores[step] - band[step]),
        "alarm_before_intervention": bool(np.any(scores[:step] >= band[:step])),
        "alarms": alarms,
    }


def factorial_cells(
    np: Any,
    natural_scores: Any,
    control_scores: Any,
    band: Any,
    step: int,
) -> dict[str, Any]:
    """Swap one SAFE-MLP contribution in its cumulative score trace."""
    natural = np.asarray(natural_scores, dtype=np.float32)
    control = np.asarray(control_scores, dtype=np.float32)
    if natural.ndim != 1 or control.ndim != 1:
        raise ValueError("factorial inputs must be one-dimensional score traces")
    if step >= len(natural) or step >= len(control):
        raise ValueError("factorial traces end before the intervention")
    if not np.isfinite(natural).all() or not np.isfinite(control).all():
        raise ValueError("factorial inputs contain non-finite SAFE scores")

    clean_previous = float(control[step - 1]) if step else 0.0
    faulted_previous = float(natural[step - 1]) if step else 0.0
    clean_contribution = float(control[step]) - clean_previous
    faulted_contribution = float(natural[step]) - faulted_previous
    contribution_change = faulted_contribution - clean_contribution

    # SAFE b6036abe, failure_prob/model/indep.py::IndepModel.forward,
    # projects each n_history_steps=1 feature independently and then applies
    # torch.cumsum when the frozen config has cumsum=true, rmean=false. Swapping
    # one feature therefore shifts every cumulative score from this step onward.
    faulted_action_clean_evidence = natural.astype(np.float64)
    faulted_action_clean_evidence[step:] -= contribution_change
    clean_action_faulted_evidence = control.astype(np.float64)
    clean_action_faulted_evidence[step:] += contribution_change
    traces = {
        "clean_action_clean_evidence": control,
        "clean_action_faulted_evidence": clean_action_faulted_evidence,
        "faulted_action_clean_evidence": faulted_action_clean_evidence,
        "faulted_action_faulted_evidence": natural,
    }
    return {
        "cells": {
            name: alarm_summary(np, scores, band, step)
            for name, scores in traces.items()
        },
        "evidence_contribution": {
            "clean": clean_contribution,
            "faulted": faulted_contribution,
            "faulted_minus_clean": contribution_change,
        },
    }


def _trajectory_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["episode_index"])


def clustered_mean(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "estimate": None,
            "trajectory_cluster_bootstrap_95": None,
            "trajectory_clusters": 0,
        }
    by_trajectory: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trajectory[_trajectory_key(row)].append(row)
    keys = sorted(by_trajectory)
    estimate = sum(value(row) for row in rows) / len(rows)
    result = {
        "estimate": estimate,
        "trajectory_clusters": len(keys),
        "trajectory_cluster_bootstrap_95": None,
    }
    if samples <= 0:
        return result

    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        sample_rows = [row for key in selected for row in by_trajectory[key]]
        estimates.append(sum(value(row) for row in sample_rows) / len(sample_rows))
    result.update(
        {
            "trajectory_cluster_bootstrap_95": [
                _percentile(estimates, 0.025),
                _percentile(estimates, 0.975),
            ],
            "bootstrap_samples": samples,
        }
    )
    return result


def command_balanced_difference(
    rows: list[dict[str, Any]], window: str
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        command_id = row.get("command_id")
        if command_id is not None:
            groups[(str(row["context_id"]), str(command_id))].append(row)
    differences = []
    for values in groups.values():
        differences.append(
            sum(
                int(row[f"restored_{window}"]) - int(row[f"shared_{window}"])
                for row in values
            )
            / len(values)
        )
    return {
        "command_groups": len(differences),
        "mean_with_equal_command_group_weight": (
            sum(differences) / len(differences) if differences else None
        ),
    }


def paired_detection_summary(
    rows: list[dict[str, Any]],
    window: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    shared_field = f"shared_{window}"
    restored_field = f"restored_{window}"
    clean_faulted_field = f"fault_evidence_clean_action_{window}"
    control_field = f"control_{window}"
    difference = lambda row: float(int(row[restored_field]) - int(row[shared_field]))
    clean_action_difference = lambda row: float(
        int(row[clean_faulted_field]) - int(row[control_field])
    )
    by_physical_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_trajectory: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("physical_run"):
            by_physical_run[str(row["physical_run"])].append(row)
        by_trajectory[_trajectory_key(row)].append(row)
    discordant_trajectories = sum(
        any(value[restored_field] != value[shared_field] for value in group)
        for group in by_trajectory.values()
    )
    zero_discordance_upper = (
        1 - math.pow(0.05, 1 / len(by_trajectory))
        if by_trajectory and discordant_trajectories == 0
        else None
    )
    return {
        "interventions": len(rows),
        "trajectories": len({_trajectory_key(row) for row in rows}),
        "physical_continuations": len(
            {str(row["physical_run"]) for row in rows if row.get("physical_run")}
        ),
        "physical_continuations_detected_with_shared_evidence": sum(
            any(value[shared_field] for value in group)
            for group in by_physical_run.values()
        ),
        "physical_continuations_with_mixed_layer_detections": sum(
            len({bool(value[shared_field]) for value in group}) > 1
            for group in by_physical_run.values()
        ),
        "shared_evidence_detected": sum(bool(row[shared_field]) for row in rows),
        "clean_evidence_restored_detected": sum(
            bool(row[restored_field]) for row in rows
        ),
        "clean_action_clean_evidence_detected": sum(
            bool(row[control_field]) for row in rows
        ),
        "clean_action_faulted_evidence_detected": sum(
            bool(row[clean_faulted_field]) for row in rows
        ),
        "restoration_recovers_detection": sum(
            bool(row[restored_field]) and not bool(row[shared_field]) for row in rows
        ),
        "faulted_evidence_adds_detection": sum(
            bool(row[shared_field]) and not bool(row[restored_field]) for row in rows
        ),
        "faulted_evidence_adds_clean_action_alarm": sum(
            bool(row[clean_faulted_field]) and not bool(row[control_field])
            for row in rows
        ),
        "faulted_evidence_suppresses_clean_action_alarm": sum(
            bool(row[control_field]) and not bool(row[clean_faulted_field])
            for row in rows
        ),
        "paired_detection_rate_difference": clustered_mean(
            rows, difference, samples=samples, seed=seed
        ),
        "trajectories_with_any_paired_difference": discordant_trajectories,
        "zero_difference_trajectory_probability_one_sided_95_upper": (
            zero_discordance_upper
        ),
        "zero_difference_bootstrap_limitation": (
            "The percentile bootstrap is degenerate when every observed paired "
            "difference is zero; it does not prove the population effect is zero."
            if rows and not any(difference(row) for row in rows)
            else None
        ),
        "clean_action_detection_rate_difference": clustered_mean(
            rows, clean_action_difference, samples=samples, seed=seed + 10_000
        ),
        "command_group_balanced_sensitivity": command_balanced_difference(rows, window),
    }


def score_shift_summary(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    values = [float(row["faulted_minus_clean_score"]) for row in rows]
    if not values:
        return {
            "interventions": 0,
            "mean": None,
            "median": None,
            "quartiles": None,
            "trajectory_cluster_bootstrap_mean_95": None,
        }
    clustered = clustered_mean(
        rows,
        lambda row: float(row["faulted_minus_clean_score"]),
        samples=samples,
        seed=seed,
    )
    positive_gaps = [
        (
            float(row["threshold_at_intervention"])
            - float(row["clean_score_at_intervention"]),
            abs(float(row["faulted_minus_clean_score"])),
        )
        for row in rows
        if float(row["threshold_at_intervention"])
        > float(row["clean_score_at_intervention"])
    ]
    relative_shifts = [shift / gap for gap, shift in positive_gaps]
    gaps = [gap for gap, _shift in positive_gaps]
    return {
        "interventions": len(rows),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.5),
        "quartiles": [_percentile(values, 0.25), _percentile(values, 0.75)],
        "minimum": min(values),
        "maximum": max(values),
        "faulted_evidence_raises_score": sum(value > 0 for value in values),
        "faulted_evidence_lowers_score": sum(value < 0 for value in values),
        "unchanged_score": sum(value == 0 for value in values),
        "trajectory_cluster_bootstrap_mean_95": clustered[
            "trajectory_cluster_bootstrap_95"
        ],
        "trajectory_clusters": clustered["trajectory_clusters"],
        "bootstrap_samples": samples,
        "clean_score_gap_below_threshold": {
            "n": len(gaps),
            "minimum": min(gaps) if gaps else None,
            "median": _percentile(gaps, 0.5) if gaps else None,
            "maximum": max(gaps) if gaps else None,
        },
        "absolute_shift_over_clean_threshold_gap": {
            "n": len(relative_shifts),
            "median": _percentile(relative_shifts, 0.5)
            if relative_shifts
            else None,
            "q25": _percentile(relative_shifts, 0.25)
            if relative_shifts
            else None,
            "q75": _percentile(relative_shifts, 0.75)
            if relative_shifts
            else None,
            "maximum": max(relative_shifts) if relative_shifts else None,
        },
    }
