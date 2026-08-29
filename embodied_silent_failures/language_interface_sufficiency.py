from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


LADDERS = ("local", "history", "context")
MODEL_FAMILIES = ("ridge", "extra_trees")
VECTOR_WIDTH = 4096
LOG_FLOOR = 1e-12


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite interface measurement: {name}")
    return number


def _measurement_features(
    measurement: dict[str, Any], prefix: str, *, width: int
) -> dict[str, float]:
    difference = _finite(measurement["difference_l2"], "difference_l2")
    normalized = _finite(
        measurement["normalized_difference_l2"], "normalized_difference_l2"
    )
    maximum = _finite(
        measurement["maximum_absolute_difference"], "maximum_absolute_difference"
    )
    changed = int(measurement["changed_element_count"])
    if min(difference, normalized, maximum, changed) < 0 or changed > width:
        raise ValueError("interface change summary is outside its declared range")
    return {
        f"{prefix}:log_difference_l2": math.log(max(difference, LOG_FLOOR)),
        f"{prefix}:log_normalized_l2": math.log(max(normalized, LOG_FLOOR)),
        f"{prefix}:log_maximum_absolute": math.log(max(maximum, LOG_FLOOR)),
        f"{prefix}:changed_fraction": changed / width,
        f"{prefix}:exact_zero": float(bool(measurement["exact_equal"])),
    }


def _validate_trace(local: dict[str, Any], record_id: str) -> list[dict[str, Any]]:
    source = int(local["layer_index"])
    propagation = list(local["propagation"])
    if len(propagation) != 32:
        raise ValueError(f"language trace does not contain 32 blocks: {record_id}")
    if [int(value["layer_index"]) for value in propagation] != list(range(32)):
        raise ValueError(f"language trace block order changed: {record_id}")
    if not 0 <= source < 32:
        raise ValueError(f"language intervention layer is invalid: {record_id}")
    if any(not value["exact_equal"] for value in propagation[:source]):
        raise ValueError(f"language fault changed an upstream block: {record_id}")
    injection = local["injection"]
    source_value = propagation[source]
    for key in (
        "changed_element_count",
        "difference_l2",
        "normalized_difference_l2",
        "maximum_absolute_difference",
        "exact_equal",
        "finite",
    ):
        if injection[key] != source_value[key]:
            raise ValueError(
                f"injection and source-block measurement disagree: {record_id}"
            )
    return propagation


def interface_rows(
    analysis_rows: list[dict[str, Any]],
    score_index: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build block-transition and fork-endpoint observations.

    Campaign revision b8bd7ea records these summaries in
    language_policy.py::intervention_record by comparing the clean and faulted
    final-token vector at every language block. They are scalar comparisons, not
    retained 4,096-dimensional perturbation vectors.
    """
    transitions = []
    safe_endpoints = []
    command_endpoints = []
    for analysis in analysis_rows:
        if not analysis.get("eligible_causal_outcome"):
            continue
        record_id = str(analysis["record_id"])
        score = score_index.get(record_id)
        if score is None:
            raise ValueError(f"eligible intervention has no SAFE record: {record_id}")
        local = score["local_measurements"]
        source = int(local["layer_index"])
        propagation = _validate_trace(local, record_id)
        metadata = {
            "record_id": record_id,
            "context_id": str(analysis["context_id"]),
            "task_id": int(analysis["task_id"]),
            "episode_index": int(analysis["episode_index"]),
            "phase": str(analysis["phase"]),
            "action_token_position": int(analysis["action_token_position"]),
            "source_layer": source,
        }
        common = {
            **metadata,
            "injection": local["injection"],
        }
        for current_layer in range(source, 31):
            transitions.append(
                {
                    **common,
                    "kind": "block_transition",
                    "boundary": f"block_{current_layer}_to_{current_layer + 1}",
                    "current_layer": current_layer,
                    "current": propagation[current_layer],
                    "history": propagation[source:current_layer],
                    "target": propagation[current_layer + 1],
                    "target_log_normalized_l2": math.log(
                        max(
                            _finite(
                                propagation[current_layer + 1][
                                    "normalized_difference_l2"
                                ],
                                "target normalized_difference_l2",
                            ),
                            LOG_FLOOR,
                        )
                    ),
                }
            )
        endpoint = {
            **common,
            "boundary": "language_fork",
            "current_layer": 31,
            "current": propagation[31],
            "history": propagation[source:31],
        }
        safe = local["safe_feature"]
        safe_endpoints.append(
            {
                **endpoint,
                "kind": "safe_feature_endpoint",
                "target": safe,
                "target_log_normalized_l2": math.log(
                    max(
                        _finite(
                            safe["normalized_difference_l2"],
                            "SAFE feature normalized_difference_l2",
                        ),
                        LOG_FLOOR,
                    )
                ),
            }
        )
        command_endpoints.append(
            {
                **endpoint,
                "kind": "command_change_endpoint",
                "command_changed": not bool(local["executed_command"]["exact_equal"]),
            }
        )
    return {
        "block_transition": transitions,
        "safe_feature_endpoint": safe_endpoints,
        "command_change_endpoint": command_endpoints,
    }


def feature_map(row: dict[str, Any], ladder: str) -> dict[str, float]:
    if ladder not in LADDERS:
        raise ValueError(f"unknown interface ladder: {ladder}")
    boundary = str(row["boundary"])
    result = {f"boundary={boundary}": 1.0}
    current = _measurement_features(row["current"], "current", width=VECTOR_WIDTH)
    result.update(current)
    # Ridge otherwise forces one perturbation slope across 31 distinct blocks.
    result.update({f"{key}@{boundary}": value for key, value in current.items()})
    if ladder == "local":
        return result

    result.update(
        _measurement_features(row["injection"], "injection", width=VECTOR_WIDTH)
    )
    source = int(row["source_layer"])
    current_layer = int(row["current_layer"])
    result[f"source_layer={source}"] = 1.0
    result["path_length"] = (current_layer - source) / 31
    for measurement in row["history"]:
        layer = int(measurement["layer_index"])
        result.update(
            _measurement_features(
                measurement, f"history_layer_{layer}", width=VECTOR_WIDTH
            )
        )
    if ladder == "history":
        return result

    result[f"task={int(row['task_id'])}"] = 1.0
    result[f"phase={row['phase']}"] = 1.0
    result[f"action_token={int(row['action_token_position'])}"] = 1.0
    return result


def _regression_metrics(labels: Any, predictions: Any) -> dict[str, float | int]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return {
        "rows": int(len(labels)),
        "root_mean_squared_error": float(
            math.sqrt(mean_squared_error(labels, predictions))
        ),
        "mean_absolute_error": float(mean_absolute_error(labels, predictions)),
        "r2": float(r2_score(labels, predictions)),
    }


def _regressor(family: str, seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if family == "ridge":
        return make_pipeline(
            DictVectorizer(sparse=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
    if family == "extra_trees":
        return make_pipeline(
            DictVectorizer(sparse=False),
            ExtraTreesRegressor(
                n_estimators=256,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"unknown regression family: {family}")


def _classifier(family: str, seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if family == "ridge":
        return make_pipeline(
            DictVectorizer(sparse=False),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight=None,
                max_iter=2_000,
                solver="lbfgs",
            ),
        )
    if family == "extra_trees":
        return make_pipeline(
            DictVectorizer(sparse=False),
            ExtraTreesClassifier(
                n_estimators=256,
                min_samples_leaf=5,
                max_features=1.0,
                class_weight=None,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"unknown classification family: {family}")


def regression_predictions(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    labels = [float(row["target_log_normalized_l2"]) for row in development]
    holdout_labels = [float(row["target_log_normalized_l2"]) for row in holdout]
    metrics = {}
    predictions = {}
    for offset, ladder in enumerate(LADDERS):
        model = _regressor(family, seed + offset)
        model.fit([feature_map(row, ladder) for row in development], labels)
        values = model.predict([feature_map(row, ladder) for row in holdout]).tolist()
        predictions[ladder] = [float(value) for value in values]
        metrics[ladder] = _regression_metrics(holdout_labels, values)
    return metrics, predictions


def classification_predictions(
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    family: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    from embodied_silent_failures.language_composition import binary_metrics

    labels = [bool(row["command_changed"]) for row in development]
    if len(set(labels)) != 2:
        raise ValueError("command endpoint development data requires both classes")
    metrics = {}
    predictions = {}
    for offset, ladder in enumerate(LADDERS):
        model = _classifier(family, seed + offset)
        model.fit([feature_map(row, ladder) for row in development], labels)
        values = model.predict_proba(
            [feature_map(row, ladder) for row in holdout]
        )[:, 1].tolist()
        predictions[ladder] = [float(value) for value in values]
        metrics[ladder] = binary_metrics(holdout, "command_changed", values)
    return metrics, predictions


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def regression_cluster_bootstrap(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = [float(row["target_log_normalized_l2"]) for row in rows]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(int(row["task_id"]), int(row["episode_index"]))].append(index)
    clusters = list(groups.values())
    squared_errors = {
        name: [(labels[index] - values[index]) ** 2 for index in range(len(rows))]
        for name, values in predictions.items()
    }

    def reduction(indices: list[int], name: str) -> float:
        local = sum(squared_errors["local"][index] for index in indices) / len(indices)
        candidate = sum(squared_errors[name][index] for index in indices) / len(indices)
        return 1 - candidate / local if local > 0 else 0.0

    all_indices = list(range(len(rows)))
    result = {
        name: {
            "relative_mse_reduction_from_local": reduction(all_indices, name),
            "trajectory_cluster_bootstrap_95": [],
            "valid_samples": samples,
        }
        for name in LADDERS[1:]
    }
    rng = random.Random(seed)
    estimates = {name: [] for name in LADDERS[1:]}
    for _ in range(samples):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        indices = [index for cluster in selected for index in cluster]
        for name in estimates:
            estimates[name].append(reduction(indices, name))
    for name, values in estimates.items():
        if values:
            result[name]["trajectory_cluster_bootstrap_95"] = [
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            ]
    return result


def per_boundary_regression(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[str, Any]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices[str(row["boundary"])].append(index)
    return {
        boundary: {
            ladder: _regression_metrics(
                [rows[index]["target_log_normalized_l2"] for index in selected],
                [predictions[ladder][index] for index in selected],
            )
            for ladder in LADDERS
        }
        for boundary, selected in sorted(indices.items())
    }
