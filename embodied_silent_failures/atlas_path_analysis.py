from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from embodied_silent_failures.atlas_path_alignment import (
    ALIGNMENT_WINDOWS,
    HORIZONS,
    STATE_STREAMS,
)


PRIMARY_HORIZON = 25
PRIMARY_WINDOW = 25
PRIMARY_STREAM = "simulator_state"
METRICS = (
    "same_clock_distance",
    "nearest_path_distance",
    "unexplained_fraction",
    "absolute_phase_offset",
    "behind_steps",
    "ahead_steps",
)


def trajectory_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["episode_index"])


def path_measurements(
    row: dict[str, Any], *, horizon: int, window: int, stream: str
) -> dict[str, float] | None:
    aligned = (
        row.get("alignments", {})
        .get(str(horizon), {})
        .get(str(window))
    )
    if aligned is None or stream not in aligned["streams"]:
        return None
    record = aligned["streams"][stream]
    offset = int(record["nearest_path"]["signed_step_offset"])
    return {
        "same_clock_distance": float(
            record["same_clock"]["symmetric_normalized_difference_l2"]
        ),
        "nearest_path_distance": float(
            record["nearest_path"]["symmetric_normalized_difference_l2"]
        ),
        "unexplained_fraction": float(record["unexplained_fraction"]),
        "absolute_phase_offset": float(abs(offset)),
        "behind_steps": float(max(0, -offset)),
        "ahead_steps": float(max(0, offset)),
    }


def _context_contributions(
    rows: list[dict[str, Any]], values: list[float]
) -> dict[tuple[int, int], tuple[int, float]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[str(row["context_id"])].append((row, float(value)))

    by_trajectory: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
    for members in grouped.values():
        failures = [value for row, value in members if not row["faulted_success"]]
        successes = [value for row, value in members if row["faulted_success"]]
        if not failures or not successes:
            continue
        wins = sum(left > right for left in failures for right in successes)
        ties = sum(left == right for left in failures for right in successes)
        pairs = len(failures) * len(successes)
        by_trajectory[trajectory_key(members[0][0])].append(
            (pairs, wins + 0.5 * ties)
        )
    return {
        trajectory: (
            sum(pairs for pairs, _wins in values),
            sum(wins for _pairs, wins in values),
        )
        for trajectory, values in by_trajectory.items()
    }


def _concordance(contributions: dict[tuple[int, int], tuple[int, float]]) -> float:
    return sum(wins for _pairs, wins in contributions.values()) / sum(
        pairs for pairs, _wins in contributions.values()
    )


def _distribution(values: list[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array),
        "quantiles": {
            "0.10": float(np.quantile(array, 0.10)),
            "0.50": float(np.quantile(array, 0.50)),
            "0.90": float(np.quantile(array, 0.90)),
        },
        "mean": float(array.mean()),
    }


def signal_point(rows: list[dict[str, Any]], values: list[float]) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = np.asarray([int(not row["faulted_success"]) for row in rows], dtype=int)
    contributions = _context_contributions(rows, values)
    return {
        "rows": len(rows),
        "failures": int(labels.sum()),
        "larger_value_failure_roc_auc": (
            float(roc_auc_score(labels, values)) if len(set(labels)) == 2 else None
        ),
        "within_context": {
            "trajectories": len(contributions),
            "failure_success_pairs": sum(
                pairs for pairs, _wins in contributions.values()
            ),
            "pair_weighted_concordance": (
                _concordance(contributions) if contributions else None
            ),
        },
        "failed": _distribution(
            [
                value
                for row, value in zip(rows, values, strict=True)
                if not row["faulted_success"]
            ]
        ),
        "successful": _distribution(
            [
                value
                for row, value in zip(rows, values, strict=True)
                if row["faulted_success"]
            ]
        ),
    }


def signal_with_uncertainty(
    rows: list[dict[str, Any]],
    values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    result = signal_point(rows, values)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[trajectory_key(row)].append(index)
    grouped_indices = list(groups.values())
    labels = np.asarray([int(not row["faulted_success"]) for row in rows], dtype=int)
    values_array = np.asarray(values, dtype=float)
    contributions = _context_contributions(rows, values)
    trajectories = list(contributions)
    rng = random.Random(seed)
    auc_samples = []
    concordance_samples = []
    for _ in range(bootstrap_samples):
        selected_groups = [
            grouped_indices[rng.randrange(len(grouped_indices))]
            for _ in grouped_indices
        ]
        indices = np.asarray(
            [index for group in selected_groups for index in group], dtype=int
        )
        if len(set(labels[indices])) == 2:
            auc_samples.append(
                float(roc_auc_score(labels[indices], values_array[indices]))
            )
        if trajectories:
            selected = [
                trajectories[rng.randrange(len(trajectories))]
                for _ in trajectories
            ]
            pairs = sum(contributions[key][0] for key in selected)
            wins = sum(contributions[key][1] for key in selected)
            if pairs:
                concordance_samples.append(wins / pairs)
    result["trajectory_bootstrap_interval_95"] = {
        "roc_auc": (
            np.quantile(auc_samples, [0.025, 0.975]).tolist()
            if auc_samples
            else None
        ),
        "within_context_concordance": (
            np.quantile(concordance_samples, [0.025, 0.975]).tolist()
            if concordance_samples
            else None
        ),
        "valid_auc_samples": len(auc_samples),
        "valid_concordance_samples": len(concordance_samples),
    }
    return result


def paired_signal_difference(
    rows: list[dict[str, Any]],
    candidate: list[float],
    baseline: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = np.asarray([int(not row["faulted_success"]) for row in rows], dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[trajectory_key(row)].append(index)
    grouped_indices = list(groups.values())
    candidate_context = _context_contributions(rows, candidate.tolist())
    baseline_context = _context_contributions(rows, baseline.tolist())
    trajectories = sorted(set(candidate_context) & set(baseline_context))

    auc_point = float(
        roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)
    )
    context_point = _concordance(candidate_context) - _concordance(baseline_context)
    rng = random.Random(seed)
    auc_samples = []
    context_samples = []
    for _ in range(bootstrap_samples):
        selected_groups = [
            grouped_indices[rng.randrange(len(grouped_indices))]
            for _ in grouped_indices
        ]
        indices = np.asarray(
            [index for group in selected_groups for index in group], dtype=int
        )
        if len(set(labels[indices])) == 2:
            auc_samples.append(
                float(
                    roc_auc_score(labels[indices], candidate[indices])
                    - roc_auc_score(labels[indices], baseline[indices])
                )
            )
        if trajectories:
            selected = [
                trajectories[rng.randrange(len(trajectories))]
                for _ in trajectories
            ]
            candidate_pairs = sum(candidate_context[key][0] for key in selected)
            candidate_wins = sum(candidate_context[key][1] for key in selected)
            baseline_pairs = sum(baseline_context[key][0] for key in selected)
            baseline_wins = sum(baseline_context[key][1] for key in selected)
            if candidate_pairs and baseline_pairs:
                context_samples.append(
                    candidate_wins / candidate_pairs - baseline_wins / baseline_pairs
                )

    def summary(point: float, samples: list[float]) -> dict[str, Any]:
        return {
            "estimate": point,
            "interval_95": (
                np.quantile(samples, [0.025, 0.975]).tolist() if samples else None
            ),
            "probability_positive": (
                sum(value > 0 for value in samples) / len(samples)
                if samples
                else None
            ),
            "valid_samples": len(samples),
        }

    return {
        "failure_roc_auc_candidate_minus_baseline": summary(
            auc_point, auc_samples
        ),
        "within_context_concordance_candidate_minus_baseline": summary(
            context_point, context_samples
        ),
    }


def _measurement_rows(
    rows: list[dict[str, Any]], *, horizon: int, window: int, stream: str
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    selected_rows = []
    measurements = []
    for row in rows:
        values = path_measurements(
            row, horizon=horizon, window=window, stream=stream
        )
        if values is None:
            continue
        selected_rows.append(row)
        measurements.append(values)
    return selected_rows, measurements


def _grid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS:
        by_window = {}
        for window in ALIGNMENT_WINDOWS:
            by_stream = {}
            for stream in STATE_STREAMS:
                selected, measurements = _measurement_rows(
                    rows, horizon=horizon, window=window, stream=stream
                )
                by_stream[stream] = {
                    name: signal_point(
                        selected, [values[name] for values in measurements]
                    )
                    for name in METRICS
                }
            by_window[str(window)] = by_stream
        output[str(horizon)] = by_window
    return output


def _declared_panels() -> tuple[tuple[int, int, str], ...]:
    return (
        (PRIMARY_HORIZON, PRIMARY_WINDOW, PRIMARY_STREAM),
        (PRIMARY_HORIZON, 5, PRIMARY_STREAM),
        (PRIMARY_HORIZON, 10, PRIMARY_STREAM),
        (PRIMARY_HORIZON, PRIMARY_WINDOW, "object-state"),
        (PRIMARY_HORIZON, PRIMARY_WINDOW, "robot0_proprio-state"),
    )


def analyze_path_alignment(
    rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    split_rows = {
        split: [row for row in rows if row["analysis_split"] == split]
        for split in ("development", "holdout")
    }
    uncertainty = {}
    for panel_index, (horizon, window, stream) in enumerate(_declared_panels()):
        panel_name = f"h{horizon}:w{window}:{stream}"
        by_split = {}
        for split_index, (split, members) in enumerate(split_rows.items()):
            selected, measurements = _measurement_rows(
                members, horizon=horizon, window=window, stream=stream
            )
            metric_summaries = {
                name: signal_with_uncertainty(
                    selected,
                    [values[name] for values in measurements],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + panel_index * 100 + split_index * 10 + metric_index,
                )
                for metric_index, name in enumerate(METRICS)
            }
            metric_summaries["nearest_path_over_same_clock"] = (
                paired_signal_difference(
                    selected,
                    [values["nearest_path_distance"] for values in measurements],
                    [values["same_clock_distance"] for values in measurements],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 10_000 + panel_index * 100 + split_index,
                )
            )
            by_split[split] = metric_summaries
        uncertainty[panel_name] = by_split

    reentry = {}
    for stream_index, stream in enumerate(STATE_STREAMS):
        by_split = {}
        for split_index, (split, members) in enumerate(split_rows.items()):
            selected = []
            values = []
            for row in members:
                early = path_measurements(
                    row, horizon=5, window=PRIMARY_WINDOW, stream=stream
                )
                late = path_measurements(
                    row, horizon=25, window=PRIMARY_WINDOW, stream=stream
                )
                if early is None or late is None:
                    continue
                selected.append(row)
                values.append(
                    {
                        "path_log_growth": math.log1p(
                            late["nearest_path_distance"]
                        )
                        - math.log1p(early["nearest_path_distance"]),
                        "absolute_phase_offset_change": (
                            late["absolute_phase_offset"]
                            - early["absolute_phase_offset"]
                        ),
                        "behind_steps_change": (
                            late["behind_steps"] - early["behind_steps"]
                        ),
                    }
                )
            by_split[split] = {
                name: signal_with_uncertainty(
                    selected,
                    [item[name] for item in values],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 20_000 + stream_index * 100 + split_index * 10 + index,
                )
                for index, name in enumerate(
                    (
                        "path_log_growth",
                        "absolute_phase_offset_change",
                        "behind_steps_change",
                    )
                )
            }
        reentry[stream] = by_split

    return {
        "population": {
            "physical_continuations": len(rows),
            "failures": sum(not row["faulted_success"] for row in rows),
            "development": len(split_rows["development"]),
            "holdout": len(split_rows["holdout"]),
        },
        "point_estimate_grid": {
            split: _grid(members) for split, members in split_rows.items()
        },
        "declared_uncertainty_panels": uncertainty,
        "reentry_from_h5_to_h25": reentry,
    }
