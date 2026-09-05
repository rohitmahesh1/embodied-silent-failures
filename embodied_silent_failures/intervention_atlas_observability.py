from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


WINDOW_STEPS = (1, 5, 10, 25, 50, 100)
PRIMARY_WINDOW_STEPS = 25


def fresh_evidence_shift(
    faulted_scores: Any,
    control_scores: Any,
    *,
    fault_step: int,
    window_steps: int,
) -> float | None:
    """Return fault-minus-control SAFE evidence accumulated in a fixed window."""
    import numpy as np

    if fault_step < 0 or window_steps < 1:
        raise ValueError("fault step must be nonnegative and window must be positive")
    stop = fault_step + window_steps - 1
    if stop >= len(faulted_scores) or stop >= len(control_scores):
        return None
    faulted_previous = float(faulted_scores[fault_step - 1]) if fault_step else 0.0
    control_previous = float(control_scores[fault_step - 1]) if fault_step else 0.0
    values = (
        float(faulted_scores[stop])
        - faulted_previous
        - float(control_scores[stop])
        + control_previous
    )
    if not np.isfinite(values):
        raise ValueError("fixed-window SAFE evidence shift is non-finite")
    return values


def _base_row(record: dict[str, Any], split: str) -> dict[str, Any]:
    context = record["context"]
    return {
        "record_id": str(record["record_id"]),
        "physical_run": str(record["physical_run"]),
        "context_id": str(record["context_id"]),
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "phase": str(context["phase"]),
        "fault_step": int(context["policy_step"]),
        "analysis_split": split,
        "policy_failure": bool(record["policy_failure"]),
    }


def _attach_shifts(
    row: dict[str, Any],
    faulted_scores: Any,
    control_scores: Any,
) -> dict[str, Any]:
    import numpy as np

    step = int(row["fault_step"])
    if step >= len(faulted_scores) or step >= len(control_scores):
        raise ValueError("SAFE trace ends before the declared fault step")
    before_fault = (
        float(faulted_scores[step - 1] - control_scores[step - 1])
        if step
        else 0.0
    )
    row["pre_fault_score_difference"] = before_fault
    row["faulted_score_length"] = len(faulted_scores)
    row["control_score_length"] = len(control_scores)
    row["fresh_evidence_shift"] = {
        str(window): fresh_evidence_shift(
            faulted_scores,
            control_scores,
            fault_step=step,
            window_steps=window,
        )
        for window in WINDOW_STEPS
    }
    if not np.isfinite(before_fault):
        raise ValueError("pre-fault SAFE score difference is non-finite")
    return row


def observability_rows(
    site_groups: list[tuple[str, list[dict[str, Any]]]],
    site_score_index: dict[str, Any],
    physical_score_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    site_rows = []
    physical_members: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for split, records in site_groups:
        for record in records:
            if not record.get("primary_eligible"):
                continue
            base = _base_row(record, split)
            control_run = f"{base['context_id']}-control"
            control = physical_score_index[control_run]["scores"]
            faulted = site_score_index[base["record_id"]]
            base["physical_run_is_control"] = base["physical_run"] == control_run
            site_rows.append(_attach_shifts(base, faulted, control))
            physical_members[base["physical_run"]].append((split, record))

    physical_rows = []
    for run, members in sorted(physical_members.items()):
        representative = next(
            (
                (split, record)
                for split, record in members
                if record["site_id"] == record["representative_site_id"]
            ),
            members[0],
        )
        split, record = representative
        base = _base_row(record, split)
        control_run = f"{base['context_id']}-control"
        if run == control_run:
            continue
        member_labels = {bool(value[1]["policy_failure"]) for value in members}
        if len(member_labels) != 1:
            raise ValueError(f"physical run {run} has mixed policy outcomes")
        physical = physical_score_index[run]
        control = physical_score_index[control_run]
        if bool(physical["record"]["success"]) == base["policy_failure"]:
            raise ValueError(f"physical outcome disagrees for {run}")
        base["member_site_count"] = len(members)
        physical_rows.append(
            _attach_shifts(base, physical["scores"], control["scores"])
        )
    return site_rows, physical_rows


def _distribution(values: list[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0}
    return {
        "count": len(array),
        "minimum": float(array.min()),
        "quantiles": {
            "0.25": float(np.quantile(array, 0.25)),
            "0.50": float(np.quantile(array, 0.50)),
            "0.75": float(np.quantile(array, 0.75)),
        },
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _concordance(
    rows: list[dict[str, Any]], window: int, group_name: str
) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_name]].append(row)
    concordance = 0.0
    comparable_pairs = 0
    groups_with_both_outcomes = 0
    for members in groups.values():
        failures = [row for row in members if row["policy_failure"]]
        nonfailures = [row for row in members if not row["policy_failure"]]
        if not failures or not nonfailures:
            continue
        groups_with_both_outcomes += 1
        for failure in failures:
            left = float(failure["fresh_evidence_shift"][str(window)])
            for nonfailure in nonfailures:
                right = float(nonfailure["fresh_evidence_shift"][str(window)])
                concordance += float(left > right) + 0.5 * float(left == right)
                comparable_pairs += 1
    return {
        "roc_auc": (
            concordance / comparable_pairs if comparable_pairs else None
        ),
        "failure_nonfailure_pairs": comparable_pairs,
        "groups_with_both_outcomes": groups_with_both_outcomes,
    }


def window_summary(rows: list[dict[str, Any]], window: int) -> dict[str, Any]:
    import numpy as np
    from sklearn import metrics

    selected = [
        row
        for row in rows
        if row["fresh_evidence_shift"][str(window)] is not None
    ]
    labels = np.asarray([int(row["policy_failure"]) for row in selected])
    values = np.asarray(
        [float(row["fresh_evidence_shift"][str(window)]) for row in selected]
    )
    if len(np.unique(labels)) != 2:
        raise ValueError("fixed-window analysis requires both policy outcomes")
    failures = values[labels == 1]
    nonfailures = values[labels == 0]
    return {
        "window_steps": window,
        "available_interventions": len(selected),
        "policy_failures": int(labels.sum()),
        "policy_nonfailures": int((labels == 0).sum()),
        "fault_minus_control_fresh_evidence": {
            "policy_failure": _distribution(failures.tolist()),
            "policy_nonfailure": _distribution(nonfailures.tolist()),
        },
        "positive_shift_rate": {
            "policy_failure": float((failures > 0).mean()),
            "policy_nonfailure": float((nonfailures > 0).mean()),
        },
        "failure_classification": {
            "pooled_roc_auc": float(metrics.roc_auc_score(labels, values)),
            "within_task": _concordance(selected, window, "task_id"),
            "within_context": _concordance(selected, window, "context_id"),
        },
    }


def _trajectory_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["task_id"], row["episode_index"])].append(index)
    return list(groups.values())


def trajectory_auc_distribution(
    rows: list[dict[str, Any]], window: int
) -> dict[str, Any]:
    import numpy as np
    from sklearn import metrics

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["fresh_evidence_shift"][str(window)] is not None:
            grouped[(row["task_id"], row["episode_index"])].append(row)
    aucs = []
    for members in grouped.values():
        labels = np.asarray([int(row["policy_failure"]) for row in members])
        if len(np.unique(labels)) != 2:
            continue
        aucs.append(
            float(
                metrics.roc_auc_score(
                    labels,
                    [row["fresh_evidence_shift"][str(window)] for row in members],
                )
            )
        )
    return {
        "trajectories_with_both_outcomes": len(aucs),
        "roc_auc_distribution": _distribution(aucs),
        "trajectories_above_chance": sum(value > 0.5 for value in aucs),
        "trajectories_at_chance": sum(value == 0.5 for value in aucs),
        "trajectories_below_chance": sum(value < 0.5 for value in aucs),
    }


def primary_window_bootstrap(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn import metrics

    window = PRIMARY_WINDOW_STEPS
    selected = [
        row
        for row in rows
        if row["fresh_evidence_shift"][str(window)] is not None
    ]
    groups = _trajectory_groups(selected)
    labels = np.asarray([int(row["policy_failure"]) for row in selected])
    values = np.asarray(
        [float(row["fresh_evidence_shift"][str(window)]) for row in selected]
    )
    estimates = {
        "pooled_roc_auc": [],
        "failure_median_minus_nonfailure_median": [],
    }
    rng = random.Random(seed)
    for _ in range(samples):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        indices = np.asarray([index for group in chosen for index in group])
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) != 2:
            continue
        sample_values = values[indices]
        estimates["pooled_roc_auc"].append(
            float(metrics.roc_auc_score(sample_labels, sample_values))
        )
        estimates["failure_median_minus_nonfailure_median"].append(
            float(
                np.median(sample_values[sample_labels == 1])
                - np.median(sample_values[sample_labels == 0])
            )
        )
    output = {}
    for name, values_for_metric in estimates.items():
        output[name] = {
            "successful_resamples": len(values_for_metric),
            "interval_95": [
                float(np.quantile(values_for_metric, 0.025)),
                float(np.quantile(values_for_metric, 0.975)),
            ],
        }
    return {
        "window_steps": window,
        "resampling_unit": "task and clean-rollout episode trajectory",
        "requested_resamples": samples,
        "metrics": output,
    }


def split_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, split in enumerate(("development", "holdout")):
        selected = [row for row in rows if row["analysis_split"] == split]
        by_task = {}
        for task in sorted({row["task_id"] for row in selected}):
            task_rows = [row for row in selected if row["task_id"] == task]
            if len({row["policy_failure"] for row in task_rows}) == 2:
                by_task[str(task)] = window_summary(
                    task_rows, PRIMARY_WINDOW_STEPS
                )
        result[split] = {
            "interventions": len(selected),
            "policy_failures": sum(row["policy_failure"] for row in selected),
            "windows": {
                str(window): window_summary(selected, window)
                for window in WINDOW_STEPS
            },
            "primary_window_trajectory_bootstrap": primary_window_bootstrap(
                selected,
                samples=bootstrap_samples,
                seed=seed + offset,
            ),
            "primary_window_by_task": by_task,
            "primary_window_by_trajectory": trajectory_auc_distribution(
                selected, PRIMARY_WINDOW_STEPS
            ),
        }
    return result


def split_auc_difference_bootstrap(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn import metrics

    prepared = {}
    for split in ("development", "holdout"):
        selected = [
            row
            for row in rows
            if row["analysis_split"] == split
            and row["fresh_evidence_shift"][str(PRIMARY_WINDOW_STEPS)] is not None
        ]
        prepared[split] = {
            "groups": _trajectory_groups(selected),
            "labels": np.asarray(
                [int(row["policy_failure"]) for row in selected]
            ),
            "values": np.asarray(
                [
                    float(
                        row["fresh_evidence_shift"][str(PRIMARY_WINDOW_STEPS)]
                    )
                    for row in selected
                ]
            ),
        }
    point = {
        split: float(
            metrics.roc_auc_score(value["labels"], value["values"])
        )
        for split, value in prepared.items()
    }
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        sampled_auc = {}
        valid = True
        for split, value in prepared.items():
            groups = value["groups"]
            chosen = [groups[rng.randrange(len(groups))] for _ in groups]
            indices = np.asarray([index for group in chosen for index in group])
            labels = value["labels"][indices]
            if len(np.unique(labels)) != 2:
                valid = False
                break
            sampled_auc[split] = float(
                metrics.roc_auc_score(labels, value["values"][indices])
            )
        if valid:
            differences.append(sampled_auc["holdout"] - sampled_auc["development"])
    return {
        "window_steps": PRIMARY_WINDOW_STEPS,
        "development_roc_auc": point["development"],
        "holdout_roc_auc": point["holdout"],
        "holdout_minus_development_roc_auc": (
            point["holdout"] - point["development"]
        ),
        "trajectory_bootstrap": {
            "requested_resamples": samples,
            "successful_resamples": len(differences),
            "interval_95": [
                float(np.quantile(differences, 0.025)),
                float(np.quantile(differences, 0.975)),
            ],
            "fraction_not_above_zero": float(
                np.mean(np.asarray(differences) <= 0)
            ),
        },
    }


def physical_divergence_audit(
    physical_rows: list[dict[str, Any]],
    mechanism_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    metric_names = (
        "object-state",
        "robot0_proprio-state",
        "simulator_state",
        "executed_command",
        "changed_action_token_fraction",
        "mean_absolute_action_entropy_difference",
    )
    result = {}
    for split in ("development", "holdout"):
        split_result = {}
        for window in (5, 10, 25, 50):
            measurements: dict[str, list[tuple[float, float, bool]]] = defaultdict(list)
            for row in physical_rows:
                if row["analysis_split"] != split:
                    continue
                evidence = row["fresh_evidence_shift"][str(window)]
                comparison = mechanism_index.get(row["physical_run"], {}).get(
                    "comparisons", {}
                ).get(str(window))
                if evidence is None or comparison is None:
                    continue
                for name in metric_names:
                    value = comparison.get(name)
                    if value is None:
                        continue
                    if isinstance(value, dict):
                        value = value["symmetric_normalized_difference_l2"]
                    measurements[name].append(
                        (float(evidence), float(value), row["policy_failure"])
                    )
            window_result = {}
            for name, values in measurements.items():
                failures = [value for value in values if value[2]]

                def correlation(selected: list[tuple[float, float, bool]]):
                    if len(selected) < 3:
                        return None
                    statistic = spearmanr(
                        [value[0] for value in selected],
                        [value[1] for value in selected],
                    ).statistic
                    return None if statistic != statistic else float(statistic)

                window_result[name] = {
                    "available_physical_continuations": len(values),
                    "spearman_rho": correlation(values),
                    "policy_failures": len(failures),
                    "policy_failure_spearman_rho": correlation(failures),
                }
            split_result[str(window)] = window_result
        result[split] = split_result
    return {
        "by_declared_split_and_window": result,
        "interpretation_boundary": (
            "SAFE windows contain the declared number of contributions beginning "
            "at the fault step; mechanism snapshots use the named offset after the "
            "fault, so their endpoints differ by one policy step. Correlations are "
            "descriptive and do not treat scalar distance as semantic failure state."
        ),
    }


def observability_audit(
    site_rows: list[dict[str, Any]], physical_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "site_interventions": len(site_rows),
        "site_interventions_mapped_to_control_behavior": sum(
            row["physical_run_is_control"] for row in site_rows
        ),
        "site_fraction_mapped_to_control_behavior": sum(
            row["physical_run_is_control"] for row in site_rows
        )
        / len(site_rows),
        "distinct_noncontrol_physical_continuations": len(physical_rows),
        "physical_policy_failures": sum(
            row["policy_failure"] for row in physical_rows
        ),
        "pre_fault_score_difference": {
            "site_units": _distribution(
                [abs(row["pre_fault_score_difference"]) for row in site_rows]
            ),
            "physical_units": _distribution(
                [abs(row["pre_fault_score_difference"]) for row in physical_rows]
            ),
        },
        "window_support_physical_units": {
            str(window): sum(
                row["fresh_evidence_shift"][str(window)] is not None
                for row in physical_rows
            )
            for window in WINDOW_STEPS
        },
    }
