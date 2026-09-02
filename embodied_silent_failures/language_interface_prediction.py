from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


SKETCH_SEEDS = {
    "residual": 2026090101,
    "key": 2026090102,
    "value": 2026090103,
    "final_residual": 2026090104,
    "safe_feature": 2026090105,
    "action_logits": 2026090106,
}


def balanced_signed_sketch(
    np: Any,
    values: Any,
    *,
    width: int,
    seed: int,
) -> Any:
    """Apply an outcome-independent sparse signed projection to the last axis."""
    matrix = np.asarray(values, dtype=np.float32)
    dimensions = int(matrix.shape[-1])
    if width <= 0 or dimensions % width:
        raise ValueError(
            f"sketch width {width} must divide input dimension {dimensions}"
        )

    # Weinberger et al., ICML 2009, doi:10.1145/1553374.1553516, use a signed
    # hash projection to preserve geometry without fitting to labels. We use
    # the same signed bucket sum with equally sized random buckets so bucket
    # occupancy itself cannot vary between tensor families or experiments.
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dimensions)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), dimensions)
    signed = matrix[..., permutation] * signs
    return signed.reshape(matrix.shape[:-1] + (width, dimensions // width)).sum(
        axis=-1,
        dtype=np.float32,
    )


def trajectory_folds(contexts: list[dict[str, Any]]) -> list[list[int]]:
    """Hold out one trajectory per task in each mechanically balanced fold."""
    by_task: dict[int, set[int]] = defaultdict(set)
    for context in contexts:
        by_task[int(context["task_id"])].add(int(context["episode_index"]))
    counts = {len(episodes) for episodes in by_task.values()}
    if len(counts) != 1:
        raise ValueError("every task must contribute the same number of trajectories")
    fold_count = counts.pop()
    if fold_count < 2:
        raise ValueError("trajectory-held-out evaluation requires at least two folds")

    assignment = {
        (task, episode): rank
        for task, episodes in sorted(by_task.items())
        for rank, episode in enumerate(sorted(episodes))
    }
    folds = [
        [
            index
            for index, context in enumerate(contexts)
            if assignment[(int(context["task_id"]), int(context["episode_index"]))]
            == fold
        ]
        for fold in range(fold_count)
    ]
    if sorted(index for fold in folds for index in fold) != list(range(len(contexts))):
        raise ValueError("trajectory folds do not partition the contexts")
    return folds


def one_hot(np: Any, values: Any, classes: int) -> Any:
    indices = np.asarray(values, dtype=int)
    if (indices < 0).any() or (indices >= classes).any():
        raise ValueError("one-hot index is outside the declared class range")
    result = np.zeros((len(indices), classes), dtype=np.float64)
    result[np.arange(len(indices)), indices] = 1.0
    return result


def fit_ridge(
    np: Any,
    features: Any,
    targets: Any,
    *,
    alpha: float,
    fit_intercept: bool,
) -> dict[str, Any]:
    """Fit one standardized multi-output ridge map with a fixed penalty."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("ridge inputs must be aligned matrices")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must contain finite observations")
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")

    x_mean = x.mean(axis=0) if fit_intercept else np.zeros(x.shape[1])
    y_mean = y.mean(axis=0) if fit_intercept else np.zeros(y.shape[1])
    centered_x = x - x_mean
    centered_y = y - y_mean
    x_scale = np.sqrt(np.mean(centered_x * centered_x, axis=0))
    y_scale = np.sqrt(np.mean(centered_y * centered_y, axis=0))
    x_scale[x_scale < 1e-12] = 1.0
    y_scale[y_scale < 1e-12] = 1.0
    scaled_x = centered_x / x_scale
    scaled_y = centered_y / y_scale
    gram = scaled_x.T @ scaled_x
    gram.flat[:: gram.shape[0] + 1] += alpha
    coefficients = np.linalg.solve(gram, scaled_x.T @ scaled_y)
    return {
        "alpha": float(alpha),
        "fit_intercept": bool(fit_intercept),
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "coefficients": coefficients,
    }


def predict_ridge(np: Any, model: dict[str, Any], features: Any) -> Any:
    x = np.asarray(features, dtype=np.float64)
    scaled = (x - model["x_mean"]) / model["x_scale"]
    return (
        scaled @ model["coefficients"] * model["y_scale"] + model["y_mean"]
    ).astype(np.float32)


def _homogeneous_features(
    np: Any,
    state: Any,
    clean_state: Any,
    action_token: Any,
) -> tuple[Any, Any]:
    values = np.asarray(state, dtype=np.float64)
    clean = np.asarray(clean_state, dtype=np.float64)
    tokens = np.asarray(action_token, dtype=int)
    norms = np.linalg.norm(values, axis=1)
    if (norms <= 0).any():
        raise ValueError("homogeneous transition inputs must be nonzero")
    features = np.concatenate(
        (
            values / norms[:, None],
            np.log1p(norms)[:, None],
            clean,
            one_hot(np, tokens, 7),
        ),
        axis=1,
    )
    return features, norms


def fit_transition(
    np: Any,
    state: Any,
    clean_state: Any,
    action_token: Any,
    target: Any,
    *,
    alpha: float,
    kind: str,
) -> dict[str, Any]:
    values = np.asarray(state, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    identity_update = kind.startswith("identity_")
    fit_targets = targets - values if identity_update else targets
    if kind in ("linear", "identity_linear"):
        ridge = fit_ridge(
            np, values, fit_targets, alpha=alpha, fit_intercept=False
        )
    elif kind in ("state_conditioned", "identity_state_conditioned"):
        features, norms = _homogeneous_features(
            np, values, clean_state, action_token
        )
        ridge = fit_ridge(
            np,
            features,
            fit_targets / norms[:, None],
            alpha=alpha,
            fit_intercept=True,
        )
    else:
        raise ValueError(f"unknown transition kind: {kind}")
    return {"kind": kind, "ridge": ridge}


def predict_transition(
    np: Any,
    model: dict[str, Any],
    state: Any,
    clean_state: Any,
    action_token: Any,
) -> Any:
    values = np.asarray(state, dtype=np.float64)
    if model["kind"] in ("linear", "identity_linear"):
        predicted = predict_ridge(np, model["ridge"], values)
        if model["kind"].startswith("identity_"):
            return (predicted + values).astype(np.float32)
        return predicted
    features, norms = _homogeneous_features(
        np, values, clean_state, action_token
    )
    normalized = predict_ridge(np, model["ridge"], features)
    predicted = normalized * norms[:, None]
    if model["kind"].startswith("identity_"):
        return (predicted + values).astype(np.float32)
    return predicted


def compose_path(
    np: Any,
    models: list[dict[str, Any]],
    initial_state: Any,
    clean_path: Any,
    action_token: int,
    *,
    source_layer: int,
) -> Any:
    current = np.asarray(initial_state, dtype=np.float32)[None, :]
    clean = np.asarray(clean_path, dtype=np.float32)
    if len(models) != 31 or clean.shape[0] != 32:
        raise ValueError("composition requires all 31 transitions and 32 clean cuts")
    for boundary in range(source_layer, 31):
        current = predict_transition(
            np,
            models[boundary],
            current,
            clean[boundary][None, :],
            np.asarray([action_token]),
        )
    return current[0]


def vector_metrics(np: Any, actual: Any, predicted: Any) -> dict[str, Any]:
    truth = np.asarray(actual, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or not len(truth):
        raise ValueError("vector metrics require aligned nonempty matrices")
    error = estimate - truth
    truth_energy = float(np.sum(truth * truth))
    error_energy = float(np.sum(error * error))
    truth_norm = np.linalg.norm(truth, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    denominator = truth_norm * estimate_norm
    cosine = np.divide(
        np.sum(truth * estimate, axis=1),
        denominator,
        out=np.full(len(truth), np.nan),
        where=denominator > 0,
    )
    norm_ratio = np.divide(
        estimate_norm,
        truth_norm,
        out=np.full(len(truth), np.nan),
        where=truth_norm > 0,
    )
    finite_cosine = cosine[np.isfinite(cosine)]
    finite_ratio = norm_ratio[np.isfinite(norm_ratio)]
    return {
        "rows": len(truth),
        "dimensions": int(truth.shape[1]),
        "normalized_mse_against_zero": (
            error_energy / truth_energy if truth_energy > 0 else None
        ),
        "relative_rmse_against_zero": (
            math.sqrt(error_energy / truth_energy) if truth_energy > 0 else None
        ),
        "median_cosine": (
            float(np.median(finite_cosine)) if len(finite_cosine) else None
        ),
        "median_predicted_to_actual_norm": (
            float(np.median(finite_ratio)) if len(finite_ratio) else None
        ),
    }
