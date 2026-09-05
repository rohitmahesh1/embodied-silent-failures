from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from embodied_silent_failures.intervention_atlas_followups import (
    classification_metrics,
    paired_metric_bootstrap,
)


MODEL_FAMILIES = (
    "task_phase",
    "task_phase_horizon",
    "task_phase_state",
    "task_phase_recovery",
    "task_phase_state_recovery",
)
REGULARIZATION_CANDIDATES = (0.001, 0.01, 0.1, 1.0)
RECOVERY_HORIZONS = (0, 1, 5, 10, 25)


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
    }


def _first_crossing(values: Any, band: Any, start: int) -> int | None:
    import numpy as np

    indices = np.flatnonzero(values[start:] >= band[start:])
    return int(start + indices[0]) if len(indices) else None


def evidence_substitution_audit(
    records: list[dict[str, Any]],
    score_index: dict[str, dict[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    failures = [
        record
        for record in records
        if record.get("primary_eligible") and record["policy_failure"]
    ]
    changed_labels = 0
    changed_times = []
    margins = []
    contribution_changes = []
    constant_residuals = []
    alarm_at_fault = 0
    physical_labels: dict[str, set[bool]] = defaultdict(set)
    for record in failures:
        score = score_index[str(record["record_id"])]
        values = score["faulted"]
        clean = score["clean_same_suffix"]
        band = score["band"]
        step = int(record["context"]["policy_step"])
        length = int(score["length"])
        values = values[:length]
        clean = clean[:length]
        band = band[:length]
        faulted_first = _first_crossing(values, band, step)
        clean_first = _first_crossing(clean, band, step)
        changed_labels += int((faulted_first is None) != (clean_first is None))
        if faulted_first != clean_first:
            changed_times.append(
                None
                if faulted_first is None or clean_first is None
                else abs(faulted_first - clean_first)
            )
        alarm_at_fault += int(values[step] >= band[step])
        margins.append(float(values[step] - band[step]))
        contribution_changes.append(
            float(record["safe_contribution"]["faulted_minus_clean"])
        )
        delta = values[step:] - clean[step:]
        constant_residuals.append(float(abs(delta - delta[0]).max()))
        physical_labels[str(record["physical_run"])].add(
            faulted_first is not None
        )
    finite_time_changes = [value for value in changed_times if value is not None]
    return {
        "failed_site_interventions": len(failures),
        "failed_physical_continuations": len(physical_labels),
        "physical_continuations_with_mixed_site_alarm_labels": sum(
            len(values) > 1 for values in physical_labels.values()
        ),
        "fault_evidence_changed_eventual_alarm_label": changed_labels,
        "fault_evidence_changed_first_alarm_step": len(changed_times),
        "first_alarm_absolute_step_change": _distribution(finite_time_changes),
        "first_alarm_changed_between_present_and_absent": (
            len(changed_times) - len(finite_time_changes)
        ),
        "alarms_at_fault_step": alarm_at_fault,
        "score_minus_threshold_at_fault": _distribution(margins),
        "faulted_minus_clean_safe_contribution": _distribution(
            contribution_changes
        ),
        "maximum_departure_from_constant_post_fault_score_offset": max(
            constant_residuals, default=0.0
        ),
        "alpha": alpha,
        "interpretation_boundary": (
            "faulted and clean evidence are compared on the same recorded physical "
            "suffix; this isolates the one-step SAFE evidence change, not the effect "
            "of the faulted command on the robot"
        ),
    }


def physical_failure_rows(
    records: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    physical_pairs: dict[str, dict[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    alpha_key = str(alpha)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("primary_eligible") and record["policy_failure"]:
            grouped[str(record["physical_run"])].append(record)
    output = []
    for run, members in sorted(grouped.items()):
        alarms = {
            bool(
                record["safe_faulted_evidence"]["alarms"][alpha_key][
                    "post_fault_any"
                ]["triggered"]
            )
            for record in members
        }
        if len(alarms) != 1:
            raise ValueError(f"physical continuation {run} has mixed alarm labels")
        first_steps = [
            record["safe_faulted_evidence"]["alarms"][alpha_key][
                "post_fault_any"
            ]["first_step"]
            for record in members
        ]
        representative = next(
            (
                record
                for record in members
                if record["site_id"] == record["representative_site_id"]
            ),
            members[0],
        )
        pair = physical_pairs[run]
        context = contexts[str(representative["context_id"])]
        step = int(representative["context"]["policy_step"])
        output.append(
            {
                "run": run,
                "context_id": str(representative["context_id"]),
                "task_id": int(representative["context"]["task_id"]),
                "episode_index": int(
                    representative["context"]["episode_index"]
                ),
                "phase": str(representative["context"]["phase"]),
                "phase_fraction": float(
                    representative["context"]["phase_fraction"]
                ),
                "analysis_split": str(context["analysis_split"]),
                "safe_alarm": alarms.pop(),
                "first_alarm_minimum": min(
                    (int(value) for value in first_steps if value is not None),
                    default=None,
                ),
                "fault_step": step,
                "faulted_observation_steps": int(pair["faulted_length"]) - step,
                "control_observation_steps": int(pair["control_length"]) - step,
                "state": context["state"],
                "comparisons": pair["comparisons"],
                "member_site_count": len(members),
            }
        )
    return output


def _representative_failures(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("primary_eligible") and record["policy_failure"]:
            grouped[str(record["physical_run"])].append(record)
    return [
        next(
            (
                record
                for record in members
                if record["site_id"] == record["representative_site_id"]
            ),
            members[0],
        )
        for _, members in sorted(grouped.items())
    ]


def monitor_process_audit(
    records: list[dict[str, Any]],
    score_index: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    import numpy as np
    from sklearn import metrics

    rows = []
    alpha_key = str(alpha)
    for record in _representative_failures(records):
        score = score_index[str(record["record_id"])]
        length = int(score["length"])
        values = score["faulted"][:length]
        band = score["band"][:length]
        step = int(record["context"]["policy_step"])
        horizon = min(25, length - step - 1)
        margin = values - band
        alarm = bool(
            record["safe_faulted_evidence"]["alarms"][alpha_key][
                "post_fault_any"
            ]["triggered"]
        )
        row = {
            "analysis_split": contexts[str(record["context_id"])][
                "analysis_split"
            ],
            "task_id": int(record["context"]["task_id"]),
            "safe_alarm": alarm,
            "score_minus_threshold_at_fault": float(margin[step]),
            "post_fault_observation_steps": length - step,
            "terminal_score_minus_threshold": float(margin[-1]),
        }
        if horizon:
            row.update(
                {
                    "first_25_score_growth_per_step": float(
                        (values[step + horizon] - values[step]) / horizon
                    ),
                    "first_25_margin_growth_per_step": float(
                        (margin[step + horizon] - margin[step]) / horizon
                    ),
                }
            )
        span = length - step - 1
        if span:
            row.update(
                {
                    "full_suffix_score_growth_per_step": float(
                        (values[-1] - values[step]) / span
                    ),
                    "full_suffix_margin_growth_per_step": float(
                        (margin[-1] - margin[step]) / span
                    ),
                }
            )
        rows.append(row)

    measurements = (
        "score_minus_threshold_at_fault",
        "first_25_score_growth_per_step",
        "first_25_margin_growth_per_step",
        "full_suffix_score_growth_per_step",
        "full_suffix_margin_growth_per_step",
        "terminal_score_minus_threshold",
    )

    def describe(selected: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for name in measurements:
            output[name] = {
                "missed": _distribution(
                    [
                        row[name]
                        for row in selected
                        if not row["safe_alarm"] and name in row
                    ]
                ),
                "detected": _distribution(
                    [
                        row[name]
                        for row in selected
                        if row["safe_alarm"] and name in row
                    ]
                ),
            }
        return output

    split_auc = {}
    for split in ("development", "holdout"):
        selected = [row for row in rows if row["analysis_split"] == split]
        split_auc[split] = {}
        for name in measurements:
            available = [row for row in selected if name in row]
            available_labels = np.asarray(
                [int(row["safe_alarm"]) for row in available]
            )
            within_task_concordance = 0.0
            within_task_pairs = 0
            for task in sorted({row["task_id"] for row in available}):
                task_rows = [row for row in available if row["task_id"] == task]
                task_labels = np.asarray(
                    [int(row["safe_alarm"]) for row in task_rows]
                )
                if len(np.unique(task_labels)) != 2:
                    continue
                comparable_pairs = int(task_labels.sum()) * int(
                    len(task_labels) - task_labels.sum()
                )
                within_task_concordance += comparable_pairs * float(
                    metrics.roc_auc_score(
                        task_labels,
                        np.asarray([row[name] for row in task_rows]),
                    )
                )
                within_task_pairs += comparable_pairs
            split_auc[split][name] = {
                "physical_continuations": len(available),
                "roc_auc": float(
                    metrics.roc_auc_score(
                        available_labels,
                        np.asarray([row[name] for row in available]),
                    )
                ),
                "pair_weighted_within_task_roc_auc": (
                    within_task_concordance / within_task_pairs
                ),
                "within_task_detected_missed_pairs": within_task_pairs,
            }

    by_task = {}
    for task in sorted({row["task_id"] for row in rows}):
        selected = [row for row in rows if row["task_id"] == task]
        by_task[str(task)] = {
            "physical_continuations": len(selected),
            "eventual_alarm_rate": sum(row["safe_alarm"] for row in selected)
            / len(selected),
            "measurements": describe(selected),
        }
    return {
        "physical_continuations": len(rows),
        "measurements_by_alarm_outcome": describe(rows),
        "univariate_roc_auc_by_declared_split": split_auc,
        "by_task": by_task,
        "interpretation_boundary": (
            "at-fault margin is available when the intervention occurs; growth "
            "and terminal quantities use the recorded post-fault suffix and are "
            "explanatory descriptions, not intervention-time predictors"
        ),
    }


def context_outcome_audit(
    rows: list[dict[str, Any]],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    def audit(selected: list[dict[str, Any]], split_seed: int) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        strata: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, row in enumerate(selected):
            grouped[str(row["context_id"])].append(row)
            strata[(int(row["task_id"]), str(row["phase"]))].append(index)
        composition = Counter()
        for members in grouped.values():
            labels = {bool(row["safe_alarm"]) for row in members}
            if len(labels) > 1:
                composition["mixed"] += 1
            elif True in labels:
                composition["all_detected"] += 1
            else:
                composition["all_missed"] += 1

        pairs = []
        labels = np.asarray([int(row["safe_alarm"]) for row in selected])
        index_by_identity = {id(row): index for index, row in enumerate(selected)}
        for members in grouped.values():
            indices = [index_by_identity[id(row)] for row in members]
            pairs.extend(
                (indices[left], indices[right])
                for left in range(len(indices))
                for right in range(left + 1, len(indices))
            )
        if not pairs:
            raise ValueError("context agreement requires repeated continuations")
        observed = float(
            np.mean([labels[left] == labels[right] for left, right in pairs])
        )
        rng = random.Random(split_seed)
        null = []
        for _ in range(permutations):
            permuted = labels.copy()
            for indices in strata.values():
                shuffled = [int(labels[index]) for index in indices]
                rng.shuffle(shuffled)
                permuted[indices] = shuffled
            null.append(
                float(
                    np.mean(
                        [
                            permuted[left] == permuted[right]
                            for left, right in pairs
                        ]
                    )
                )
            )
        return {
            "physical_continuations": len(selected),
            "contexts": len(grouped),
            "context_outcome_composition": dict(sorted(composition.items())),
            "within_context_pair_agreement": observed,
            "task_and_phase_permutation_null": {
                "permutations": permutations,
                "median": float(np.median(null)),
                "quantiles": {
                    "0.025": float(np.quantile(null, 0.025)),
                    "0.975": float(np.quantile(null, 0.975)),
                },
                "one_sided_p_value": (1 + sum(value >= observed for value in null))
                / (permutations + 1),
            },
        }

    return {
        "development": audit(
            [row for row in rows if row["analysis_split"] == "development"],
            seed,
        ),
        "holdout": audit(
            [row for row in rows if row["analysis_split"] == "holdout"],
            seed + 1,
        ),
        "combined_exploratory": audit(rows, seed + 2),
        "interpretation_boundary": (
            "the permutation preserves task, early/middle/late phase, and each "
            "stratum's alarm count; combined results are exploratory, while the "
            "declared development and holdout splits show replication"
        ),
    }


def alarm_horizon_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detected = [row for row in rows if row["safe_alarm"]]
    missed = [row for row in rows if not row["safe_alarm"]]
    delays = [
        int(row["first_alarm_minimum"]) - int(row["fault_step"])
        for row in detected
    ]
    task_counts = {}
    for task in sorted({row["task_id"] for row in rows}):
        selected = [row for row in rows if row["task_id"] == task]
        task_counts[str(task)] = {
            "failed_physical_continuations": len(selected),
            "eventual_alarms": sum(row["safe_alarm"] for row in selected),
            "eventual_alarm_rate": sum(row["safe_alarm"] for row in selected)
            / len(selected),
        }
    return {
        "failed_physical_continuations": len(rows),
        "eventual_alarms": len(detected),
        "eventual_misses": len(missed),
        "alarm_delay_after_fault_steps": _distribution(delays),
        "faulted_observation_steps": {
            "detected": _distribution(
                [row["faulted_observation_steps"] for row in detected]
            ),
            "missed": _distribution(
                [row["faulted_observation_steps"] for row in missed]
            ),
        },
        "control_observation_steps": {
            "detected": _distribution(
                [row["control_observation_steps"] for row in detected]
            ),
            "missed": _distribution(
                [row["control_observation_steps"] for row in missed]
            ),
        },
        "alarms_after_successful_control_ended": sum(
            row["safe_alarm"]
            and int(row["first_alarm_minimum"])
            >= int(row["fault_step"]) + int(row["control_observation_steps"])
            for row in rows
        ),
        "by_task": task_counts,
    }


def _base_features(row: dict[str, Any]) -> dict[str, float]:
    return {
        f"task={row['task_id']}": 1.0,
        "phase_fraction": float(row["phase_fraction"]),
    }


def _state_features(row: dict[str, Any]) -> dict[str, float]:
    result = {}
    task = row["task_id"]
    for name in ("object-state", "robot0_proprio-state"):
        for index, value in enumerate(row["state"][name]):
            result[f"task{task}:{name}:{index}"] = float(value)
    return result


def _recovery_features(row: dict[str, Any]) -> dict[str, float]:
    result = {}
    for horizon in RECOVERY_HORIZONS:
        comparison = row["comparisons"].get(str(horizon))
        if comparison is None:
            result[f"h{horizon}:missing"] = 1.0
            continue
        for name in (
            "object-state",
            "robot0_proprio-state",
            "executed_command",
        ):
            if name in comparison:
                result[f"h{horizon}:{name}"] = float(
                    comparison[name]["symmetric_normalized_difference_l2"]
                )
        for name in (
            "changed_action_token_fraction",
            "mean_absolute_action_entropy_difference",
        ):
            if name in comparison:
                result[f"h{horizon}:{name}"] = float(comparison[name])
    return result


def mechanism_features(row: dict[str, Any], family: str) -> dict[str, float]:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown mechanism feature family: {family}")
    result = _base_features(row)
    if family == "task_phase_horizon":
        result["faulted_observation_steps"] = float(
            row["faulted_observation_steps"]
        )
        result["control_observation_steps"] = float(
            row["control_observation_steps"]
        )
    if family in {"task_phase_state", "task_phase_state_recovery"}:
        result.update(_state_features(row))
    if family in {"task_phase_recovery", "task_phase_state_recovery"}:
        result.update(_recovery_features(row))
    return result


def _pipeline(c: float):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True, sort=True)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "logistic_regression",
                LogisticRegression(C=c, max_iter=5_000, solver="lbfgs"),
            ),
        ]
    )


def fit_mechanism_model(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    family: str,
    folds: int,
) -> tuple[dict[str, Any], Any]:
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import GroupKFold, cross_val_predict

    development_features = [mechanism_features(row, family) for row in development]
    holdout_features = [mechanism_features(row, family) for row in holdout]
    development_labels = np.asarray(
        [int(row["safe_alarm"]) for row in development], dtype=int
    )
    holdout_labels = np.asarray(
        [int(row["safe_alarm"]) for row in holdout], dtype=int
    )
    groups = np.asarray([row["context_id"] for row in development])
    split = GroupKFold(folds)
    candidates = []
    for c in REGULARIZATION_CANDIDATES:
        probabilities = cross_val_predict(
            _pipeline(c),
            development_features,
            development_labels,
            groups=groups,
            cv=split,
            method="predict_proba",
        )[:, 1]
        candidates.append(
            {
                "C": c,
                "roc_auc": float(
                    metrics.roc_auc_score(development_labels, probabilities)
                ),
            }
        )
    selected = max(candidates, key=lambda value: (value["roc_auc"], -value["C"]))
    model = _pipeline(float(selected["C"]))
    model.fit(development_features, development_labels)
    holdout_probabilities = model.predict_proba(holdout_features)[:, 1]
    return (
        {
            "selected_regularization_C": selected["C"],
            "development_grouped_cv_candidates": candidates,
            "holdout": classification_metrics(
                holdout_labels, holdout_probabilities
            ),
            "development_physical_failures": len(development),
            "holdout_physical_failures": len(holdout),
        },
        holdout_probabilities,
    )


def recovery_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for horizon in RECOVERY_HORIZONS:
        for name in (
            "object-state",
            "robot0_proprio-state",
            "executed_command",
        ):
            values = {False: [], True: []}
            for row in rows:
                comparison = row["comparisons"].get(str(horizon))
                if comparison is None or name not in comparison:
                    continue
                values[bool(row["safe_alarm"])].append(
                    float(
                        comparison[name][
                            "symmetric_normalized_difference_l2"
                        ]
                    )
                )
            output[f"h{horizon}:{name}"] = {
                "missed": _distribution(values[False]),
                "detected": _distribution(values[True]),
            }
    return output


def analyze_mechanism_models(
    rows: list[dict[str, Any]],
    *,
    folds: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    development = [row for row in rows if row["analysis_split"] == "development"]
    holdout = [row for row in rows if row["analysis_split"] == "holdout"]
    labels = np.asarray([int(row["safe_alarm"]) for row in holdout], dtype=int)
    models = {}
    predictions = {}
    for family in MODEL_FAMILIES:
        models[family], predictions[family] = fit_mechanism_model(
            development, holdout, family=family, folds=folds
        )
    comparisons = {}
    for offset, family in enumerate(MODEL_FAMILIES[1:]):
        comparisons[f"{family}_over_task_phase"] = paired_metric_bootstrap(
            holdout,
            labels,
            predictions[family],
            predictions["task_phase"],
            samples=bootstrap_samples,
            seed=seed + offset,
        )
    return {
        "models": models,
        "holdout_comparisons": comparisons,
        "recovery_metric_descriptions": recovery_metric_summary(rows),
    }
