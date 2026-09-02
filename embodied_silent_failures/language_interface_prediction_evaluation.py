from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_silent_failures.language_composition import (
    binary_metrics,
    clustered_bootstrap,
)
from embodied_silent_failures.language_interface_prediction import (
    SKETCH_SEEDS,
    compose_path,
    fit_ridge,
    fit_transition,
    predict_ridge,
    predict_transition,
    trajectory_folds,
    vector_metrics,
)
from embodied_silent_failures.language_interface_prediction_data import (
    TRANSITION_KINDS,
    boundary_rows,
    endpoint_features,
    fork_slices,
    load_prediction_data,
    local_risk_features,
    risk_features,
    stack_context_rows,
)
from embodied_silent_failures.provenance import file_sha256


def _fit_logistic(features: Any, labels: Any) -> tuple[Any, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(set(int(value) for value in labels)) != 2:
        raise ValueError("risk training fold requires both outcome classes")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=2_000,
        penalty="l2",
        solver="lbfgs",
    ).fit(scaler.transform(features), labels)
    return scaler, model


def _probabilities(model: tuple[Any, Any], features: Any) -> Any:
    scaler, estimator = model
    return estimator.predict_proba(scaler.transform(features))[:, 1]


def _fork_metrics(
    np: Any, actual: Any, predicted: Any, *, sketch_width: int
) -> dict[str, Any]:
    return {
        name: vector_metrics(np, actual[:, start:end], predicted[:, start:end])
        for name, (start, end) in fork_slices(sketch_width).items()
    }


def _state_family_metrics(
    np: Any, actual: Any, predicted: Any, *, sketch_width: int
) -> dict[str, Any]:
    families = ("residual", "key", "value")
    return {
        name: vector_metrics(
            np,
            actual[:, index * sketch_width : (index + 1) * sketch_width],
            predicted[:, index * sketch_width : (index + 1) * sketch_width],
        )
        for index, name in enumerate(families)
    }


def _fit_transitions(
    np: Any,
    rows: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    *,
    ridge_alpha: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, list[Any]]]]:
    models: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in TRANSITION_KINDS
    }
    predictions = {
        kind: {"actual": [], "predicted": [], "persistence": [], "boundary": []}
        for kind in TRANSITION_KINDS
    }
    for boundary in range(31):
        train_state, train_clean, train_token, train_target = boundary_rows(
            np, rows, train_indices, boundary
        )
        test_state, test_clean, test_token, test_target = boundary_rows(
            np, rows, test_indices, boundary
        )
        for kind in TRANSITION_KINDS:
            model = fit_transition(
                np,
                train_state,
                train_clean,
                train_token,
                train_target,
                alpha=ridge_alpha,
                kind=kind,
            )
            models[kind].append(model)
            predicted = predict_transition(
                np, model, test_state, test_clean, test_token
            )
            predictions[kind]["actual"].append(test_target)
            predictions[kind]["predicted"].append(predicted)
            predictions[kind]["persistence"].append(test_state)
            predictions[kind]["boundary"].extend([boundary] * len(test_target))
    return models, predictions


def _compose_test_contexts(
    np: Any,
    rows: list[dict[str, Any]],
    test_indices: list[int],
    models: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result = {}
    for kind in TRANSITION_KINDS:
        composed = []
        for index in test_indices:
            row = rows[index]
            for source_layer in range(32):
                composed.append(
                    compose_path(
                        np,
                        models[kind],
                        row["path"][source_layer, source_layer],
                        row["clean"],
                        int(row["context"]["action_token_position"]),
                        source_layer=source_layer,
                    )
                )
        result[kind] = np.asarray(composed)
    return result


def _fit_endpoint_models(
    np: Any,
    rows: list[dict[str, Any]],
    train_indices: list[int],
    train_actual_final: Any,
    train_fork: Any,
    *,
    ridge_alpha: float,
) -> dict[str, dict[str, Any]]:
    models = {}
    for kind in ("local", "history", "source_context", "direct"):
        state = (
            train_actual_final
            if kind != "direct"
            else np.zeros_like(train_actual_final)
        )
        models[kind] = fit_ridge(
            np,
            endpoint_features(np, rows, train_indices, state, kind=kind),
            train_fork,
            alpha=ridge_alpha,
            fit_intercept=True,
        )
    return models


def _predict_forks(
    np: Any,
    rows: list[dict[str, Any]],
    test_indices: list[int],
    test_actual_final: Any,
    composed: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "local_actual_state": predict_ridge(
            np,
            models["local"],
            endpoint_features(
                np, rows, test_indices, test_actual_final, kind="local"
            ),
        ),
        "history_actual_state": predict_ridge(
            np,
            models["history"],
            endpoint_features(
                np, rows, test_indices, test_actual_final, kind="history"
            ),
        ),
        "source_context_actual_state": predict_ridge(
            np,
            models["source_context"],
            endpoint_features(
                np,
                rows,
                test_indices,
                test_actual_final,
                kind="source_context",
            ),
        ),
        "direct": predict_ridge(
            np,
            models["direct"],
            endpoint_features(
                np,
                rows,
                test_indices,
                np.zeros_like(test_actual_final),
                kind="direct",
            ),
        ),
    }
    for kind in TRANSITION_KINDS:
        result[f"composed_{kind}"] = predict_ridge(
            np,
            models["local"],
            endpoint_features(
                np, rows, test_indices, composed[kind], kind="local"
            ),
        )
    return result


def _risk_fold(
    np: Any,
    rows: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    train_fork: Any,
    test_fork: Any,
    predicted_forks: dict[str, Any],
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    fork_width = int(train_fork.shape[1])
    train_local, _train_records, train_labels = local_risk_features(
        np, rows, train_indices, fork_width=fork_width
    )
    test_local, test_records, test_labels = local_risk_features(
        np, rows, test_indices, fork_width=fork_width
    )
    train_actual, _records, actual_labels = risk_features(
        np, rows, train_indices, train_fork
    )
    if not np.array_equal(train_labels, actual_labels):
        raise ValueError("local and fork training labels lost alignment")
    local_model = _fit_logistic(train_local, train_labels)
    fork_model = _fit_logistic(train_actual, train_labels)
    probabilities = {
        "local_only": _probabilities(local_model, test_local).tolist()
    }
    variants = {
        "actual_fork": test_fork,
        "direct_fork": predicted_forks["direct"],
        **{
            f"composed_{kind}": predicted_forks[f"composed_{kind}"]
            for kind in TRANSITION_KINDS
        },
    }
    for name, fork_values in variants.items():
        features, records, labels = risk_features(
            np, rows, test_indices, fork_values
        )
        if records != test_records or not np.array_equal(labels, test_labels):
            raise ValueError("risk feature variants lost record alignment")
        probabilities[name] = _probabilities(fork_model, features).tolist()
    return probabilities, test_records


def _summarize_one_step(
    np: Any,
    collected: dict[str, dict[str, list[Any]]],
    *,
    sketch_width: int,
) -> dict[str, Any]:
    output = {}
    for kind, values in collected.items():
        actual = np.concatenate(values["actual"])
        predicted = np.concatenate(values["predicted"])
        persistence = np.concatenate(values["persistence"])
        boundaries = np.asarray(values["boundary"])
        output[kind] = {
            "overall": vector_metrics(np, actual, predicted),
            "by_family": _state_family_metrics(
                np, actual, predicted, sketch_width=sketch_width
            ),
            "persistence_baseline": vector_metrics(np, actual, persistence),
            "persistence_by_family": _state_family_metrics(
                np, actual, persistence, sketch_width=sketch_width
            ),
            "by_boundary": [
                {
                    "boundary": boundary,
                    "model": vector_metrics(
                        np,
                        actual[boundaries == boundary],
                        predicted[boundaries == boundary],
                    ),
                    "persistence": vector_metrics(
                        np,
                        actual[boundaries == boundary],
                        persistence[boundaries == boundary],
                    ),
                }
                for boundary in range(31)
            ],
        }
    return output


def _summarize_final_state(
    np: Any,
    collected: dict[str, dict[str, list[Any]]],
    *,
    sketch_width: int,
) -> dict[str, Any]:
    output = {}
    for kind, values in collected.items():
        actual = np.concatenate(values["actual"])
        predicted = np.concatenate(values["predicted"])
        persistence = np.concatenate(values["persistence"])
        sources = np.asarray(values["source_layer"])
        output[kind] = {
            "overall": vector_metrics(np, actual, predicted),
            "by_family": _state_family_metrics(
                np, actual, predicted, sketch_width=sketch_width
            ),
            "persistence_baseline": vector_metrics(np, actual, persistence),
            "persistence_by_family": _state_family_metrics(
                np, actual, persistence, sketch_width=sketch_width
            ),
            "by_source_layer": [
                {
                    "source_layer": source,
                    "model": vector_metrics(
                        np, actual[sources == source], predicted[sources == source]
                    ),
                    "persistence": vector_metrics(
                        np,
                        actual[sources == source],
                        persistence[sources == source],
                    ),
                }
                for source in range(32)
            ],
        }
    return output


def _risk_metrics_by_fold(
    records: list[dict[str, Any]], probabilities: dict[str, list[float]]
) -> list[dict[str, Any]]:
    folds = sorted({int(record["fold"]) for record in records})
    result = []
    for fold in folds:
        indices = [
            index
            for index, record in enumerate(records)
            if int(record["fold"]) == fold
        ]
        current_rows = [records[index] for index in indices]
        result.append(
            {
                "fold": fold,
                "metrics": {
                    name: binary_metrics(
                        current_rows,
                        "operational_silent_failure",
                        [values[index] for index in indices],
                    )
                    for name, values in probabilities.items()
                },
            }
        )
    return result


def analyze_prediction(
    np: Any,
    atlas_dirs: list[Path],
    *,
    sketch_width: int,
    ridge_alpha: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import sklearn

    loaded = load_prediction_data(np, atlas_dirs, sketch_width)
    rows = loaded["rows"]
    contexts = [row["context"] for row in rows]
    folds = trajectory_folds(contexts)
    all_indices = set(range(len(rows)))
    one_step = {
        kind: {"actual": [], "predicted": [], "persistence": [], "boundary": []}
        for kind in TRANSITION_KINDS
    }
    final_state = {
        kind: {
            "actual": [],
            "predicted": [],
            "persistence": [],
            "source_layer": [],
        }
        for kind in TRANSITION_KINDS
    }
    fork_predictions = {
        "direct": [],
        "local_actual_state": [],
        "history_actual_state": [],
        "source_context_actual_state": [],
        **{f"composed_{kind}": [] for kind in TRANSITION_KINDS},
    }
    fork_actual = []
    risk_probabilities = {
        "local_only": [],
        "actual_fork": [],
        "direct_fork": [],
        **{f"composed_{kind}": [] for kind in TRANSITION_KINDS},
    }
    risk_records = []

    for fold_index, test_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(test_indices))
        models, fold_steps = _fit_transitions(
            np,
            rows,
            train_indices,
            test_indices,
            ridge_alpha=ridge_alpha,
        )
        for kind in TRANSITION_KINDS:
            for field in one_step[kind]:
                one_step[kind][field].extend(fold_steps[kind][field])
        composed = _compose_test_contexts(np, rows, test_indices, models)
        test_path = stack_context_rows(np, rows, test_indices, "path").reshape(
            len(test_indices), 32, 32, -1
        )
        test_actual_final = test_path[:, :, 31].reshape(-1, 3 * sketch_width)
        source_layers = np.arange(32)
        test_initial_state = test_path[:, source_layers, source_layers].reshape(
            -1, 3 * sketch_width
        )
        for kind in TRANSITION_KINDS:
            final_state[kind]["actual"].append(test_actual_final)
            final_state[kind]["predicted"].append(composed[kind])
            final_state[kind]["persistence"].append(test_initial_state)
            final_state[kind]["source_layer"].extend(
                np.tile(source_layers, len(test_indices)).tolist()
            )

        train_actual_final = stack_context_rows(
            np, rows, train_indices, "path"
        ).reshape(len(train_indices), 32, 32, -1)[:, :, 31].reshape(
            -1, 3 * sketch_width
        )
        train_fork = stack_context_rows(np, rows, train_indices, "fork")
        test_fork = stack_context_rows(np, rows, test_indices, "fork")
        endpoint_models = _fit_endpoint_models(
            np,
            rows,
            train_indices,
            train_actual_final,
            train_fork,
            ridge_alpha=ridge_alpha,
        )
        predicted_forks = _predict_forks(
            np,
            rows,
            test_indices,
            test_actual_final,
            composed,
            endpoint_models,
        )
        fork_actual.append(test_fork)
        for name, values in predicted_forks.items():
            fork_predictions[name].append(values)

        fold_probabilities, fold_records = _risk_fold(
            np,
            rows,
            train_indices,
            test_indices,
            train_fork,
            test_fork,
            predicted_forks,
        )
        for name, values in fold_probabilities.items():
            risk_probabilities[name].extend(values)
        risk_records.extend({**record, "fold": fold_index} for record in fold_records)

    if any(len(values) != len(risk_records) for values in risk_probabilities.values()):
        raise ValueError("risk predictions do not align across folds")
    for index, record in enumerate(risk_records):
        for name, values in risk_probabilities.items():
            record[f"probability_{name}"] = values[index]

    actual_fork = np.concatenate(fork_actual)
    fork_output = {
        name: _fork_metrics(
            np,
            actual_fork,
            np.concatenate(values),
            sketch_width=sketch_width,
        )
        for name, values in fork_predictions.items()
    }
    risk_metrics = {
        name: binary_metrics(
            risk_records, "operational_silent_failure", values
        )
        for name, values in risk_probabilities.items()
    }
    output = {
        "schema_version": 1,
        "analysis": "development-only cross-fitted interface composition test",
        "status": (
            "hypothesis-generating; all maps are fit without named holdout data and "
            "all reported predictions are from whole-trajectory held-out folds"
        ),
        "question": (
            "Can independently fit local language-block transformations compose to "
            "the policy-monitor fork and rank terminal failures missed by SAFE?"
        ),
        "population": {
            "planned_contexts": sum(len(run["context_ids"]) for run in loaded["runs"]),
            "complete_contexts": len(rows),
            "unresolved_contexts": len(loaded["errors"]),
            "trajectory_folds": len(folds),
            "layer_interventions": len(rows) * 32,
            "eligible_terminal_interventions": len(risk_records),
            "silent_failures": sum(
                int(record["operational_silent_failure"])
                for record in risk_records
            ),
        },
        "design": _design_record(sketch_width, ridge_alpha),
        "implementation": {
            "base_repository_revision": loaded["runs"][0]["code"]["revision"],
            "git_metadata": (
                "analysis files were staged on the CPU worker without a Git "
                "directory; exact source files are identified by SHA-256"
            ),
            "driver_sha256": file_sha256(
                Path(__file__).with_name(
                    "analyze_language_interface_prediction.py"
                )
            ),
            "evaluation_sha256": file_sha256(Path(__file__)),
            "data_sha256": file_sha256(
                Path(__file__).with_name("language_interface_prediction_data.py")
            ),
            "mechanics_sha256": file_sha256(
                Path(__file__).with_name("language_interface_prediction.py")
            ),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "logistic_regression": {
                "C": 1.0,
                "class_weight": None,
                "max_iter": 2_000,
                "penalty": "l2",
                "solver": "lbfgs",
            },
        },
        "interpretation_boundary": (
            "The traced cache state contains differential current-token post-rotary "
            "entries. Earlier prompt/action-token cache entries remain an exact replay "
            "precondition and are not represented in this cross-context predictor."
        ),
        "sources": [
            {
                "path": str(path.resolve()),
                "run_sha256": file_sha256(path / "run.json"),
            }
            for path in atlas_dirs
        ],
        "unresolved": loaded["errors"],
        "one_step": _summarize_one_step(
            np, one_step, sketch_width=sketch_width
        ),
        "recursive_final_state": _summarize_final_state(
            np, final_state, sketch_width=sketch_width
        ),
        "fork": fork_output,
        "residual_risk": {
            "metrics": risk_metrics,
            "by_fold": _risk_metrics_by_fold(risk_records, risk_probabilities),
            "trajectory_cluster_bootstrap": clustered_bootstrap(
                risk_records,
                "operational_silent_failure",
                risk_probabilities,
                samples=bootstrap_samples,
                seed=seed,
                baseline="local_only",
            ),
        },
    }
    return output, risk_records


def _design_record(sketch_width: int, ridge_alpha: float) -> dict[str, Any]:
    return {
        "sketch": {
            "width_per_tensor_family": sketch_width,
            "state_dimensions": sketch_width * 3,
            "seeds": SKETCH_SEEDS,
            "cache_layer_seed": "family seed + 1009 * (zero-based layer + 1)",
            "cache_state": (
                "sum of independently sketched layer-owned differential entries "
                "from the fault output through the current boundary"
            ),
            "fit_to_data_or_outcomes": False,
            "role": (
                "computational screen only; this analysis does not claim the "
                "sketch is a sufficient or uniquely meaningful interface"
            ),
        },
        "folds": (
            "Within each task, trajectories are sorted by episode index; fold k "
            "holds out the kth trajectory and all of its contexts and faults."
        ),
        "transition_models": {
            "linear": "zero-preserving standardized ridge map of the signed state",
            "identity_linear": (
                "the signed state plus a zero-preserving standardized ridge map "
                "of its one-boundary correction"
            ),
            "state_conditioned": (
                "ridge map of perturbation direction, magnitude, clean boundary "
                "state, and action-token position; output is rescaled by input norm"
            ),
            "identity_state_conditioned": (
                "the signed state plus a ridge-predicted correction conditioned on "
                "perturbation direction, magnitude, clean boundary state, and "
                "action-token position"
            ),
            "ridge_alpha": ridge_alpha,
        },
        "fork_tests": {
            "local_actual_state": (
                "actual current-token final cut, clean final cut, and token position"
            ),
            "history_actual_state": (
                "the local description plus injection state and source layer; "
                "improvement diagnoses information absent from the local cut"
            ),
            "source_context_actual_state": (
                "the history description plus the clean state at the fault's "
                "source boundary"
            ),
            "direct": (
                "injection state and graph/context descriptors without composition"
            ),
            "composed": (
                "recursive held-out local-transition prediction passed through the "
                "same fork map used for the actual final cut"
            ),
        },
        "risk": (
            "A frozen-form L2 logistic model is fit on actual immediate fork state "
            "plus exact pre-fault simulator state, task, and token position. Raw "
            "simulator coordinates occupy separate task blocks, so coordinates from "
            "different LIBERO object models are never equated. The same model "
            "evaluates actual, direct, and composed fork states."
        ),
    }
