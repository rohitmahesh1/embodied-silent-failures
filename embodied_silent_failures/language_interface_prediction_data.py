from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_silent_failures.language_interface_prediction import (
    SKETCH_SEEDS,
    balanced_signed_sketch,
    one_hot,
)
from embodied_silent_failures.provenance import file_sha256, load_json


TRANSITION_KINDS = (
    "linear",
    "identity_linear",
    "state_conditioned",
    "identity_state_conditioned",
)


def fork_slices(width: int) -> dict[str, tuple[int, int]]:
    return {
        "final_residual": (0, width),
        "safe_feature": (width, 2 * width),
        "action_logits": (2 * width, 3 * width),
        "executed_command": (3 * width, 3 * width + 7),
        "safe_score_shift": (3 * width + 7, 3 * width + 8),
    }


def _sketch(np: Any, values: Any, family: str, width: int) -> Any:
    return balanced_signed_sketch(
        np, values, width=width, seed=SKETCH_SEEDS[family]
    )


def _cache_row_sketch(
    np: Any,
    values: Any,
    *,
    family: str,
    layer: int,
    width: int,
) -> Any:
    # A boundary cache cut is a growing tuple of layer-owned entries. Giving
    # each layer an independent signed map before summation is a sparse sketch
    # of that mechanically ordered tuple, rather than treating different
    # layers' cache coordinates as interchangeable.
    return balanced_signed_sketch(
        np,
        values,
        width=width,
        seed=SKETCH_SEEDS[family] + 1009 * (layer + 1),
    )


def cumulative_cache_state(
    np: Any,
    row_sketches: Any,
    sources: Any,
    boundaries: Any,
    *,
    width: int,
) -> Any:
    """Accumulate every differential cache entry live at each boundary cut."""
    sketches = np.asarray(row_sketches, dtype=np.float32)
    source_values = np.asarray(sources, dtype=int)
    boundary_values = np.asarray(boundaries, dtype=int)
    expected = {
        (source, boundary)
        for source in range(31)
        for boundary in range(source + 1, 32)
    }
    observed = set(zip(source_values.tolist(), boundary_values.tolist(), strict=True))
    if observed != expected or len(sketches) != len(expected):
        raise ValueError("cache path does not cover every downstream cut coordinate")
    result = np.zeros((32, 32, width), dtype=np.float32)
    index = {
        (source, boundary): row
        for row, source, boundary in zip(
            sketches, source_values, boundary_values, strict=True
        )
    }
    for source in range(31):
        running = np.zeros(width, dtype=np.float32)
        for boundary in range(source + 1, 32):
            running = running + index[(source, boundary)]
            result[source, boundary] = running
    return result


def _sketch_cache_path(
    np: Any,
    values: Any,
    sources: Any,
    boundaries: Any,
    *,
    family: str,
    width: int,
) -> Any:
    source_values = np.asarray(sources, dtype=int)
    boundary_values = np.asarray(boundaries, dtype=int)
    matrix = np.asarray(values, dtype=np.float32).reshape(-1, 4096)
    sketches = np.empty((len(matrix), width), dtype=np.float32)
    for layer in range(1, 32):
        selected = boundary_values == layer
        sketches[selected] = _cache_row_sketch(
            np,
            matrix[selected],
            family=family,
            layer=layer,
            width=width,
        )
    return cumulative_cache_state(
        np,
        sketches,
        source_values,
        boundary_values,
        width=width,
    )


def path_state(np: Any, archive: Any, width: int) -> Any:
    residual = np.zeros((32, 32, width), dtype=np.float32)
    residual_rows = _sketch(
        np, archive["residual_path_delta"], "residual", width
    )
    for row, source, boundary in zip(
        residual_rows,
        archive["residual_path_source_layer"].astype(int),
        archive["residual_path_boundary_layer"].astype(int),
        strict=True,
    ):
        residual[source, boundary] = row
    sources = archive["cache_path_source_layer"].astype(int)
    boundaries = archive["cache_path_boundary_layer"].astype(int)
    keys = _sketch_cache_path(
        np,
        archive["cache_key_path_delta"],
        sources,
        boundaries,
        family="key",
        width=width,
    )
    values = _sketch_cache_path(
        np,
        archive["cache_value_path_delta"],
        sources,
        boundaries,
        family="value",
        width=width,
    )
    return np.concatenate((residual, keys, values), axis=-1)


def clean_state(np: Any, archive: Any, width: int) -> Any:
    clean_keys = np.asarray(archive["clean_keys"], dtype=np.float32).reshape(32, 4096)
    clean_values = np.asarray(archive["clean_values"], dtype=np.float32).reshape(
        32, 4096
    )
    key_rows = np.stack(
        [
            _cache_row_sketch(
                np, clean_keys[layer], family="key", layer=layer, width=width
            )
            for layer in range(32)
        ]
    ).cumsum(axis=0)
    value_rows = np.stack(
        [
            _cache_row_sketch(
                np, clean_values[layer], family="value", layer=layer, width=width
            )
            for layer in range(32)
        ]
    ).cumsum(axis=0)
    return np.concatenate(
        (
            _sketch(np, archive["clean_residual"], "residual", width),
            key_rows,
            value_rows,
        ),
        axis=-1,
    )


def fork_state(np: Any, archive: Any, width: int) -> Any:
    score_shift = np.asarray(
        archive["score_change_from_control_at_fault"], dtype=np.float32
    )[:, None]
    return np.concatenate(
        (
            _sketch(
                np, archive["final_residual_delta"], "final_residual", width
            ),
            _sketch(np, archive["safe_feature_delta"], "safe_feature", width),
            _sketch(
                np,
                archive["action_logit_delta"].reshape(32, -1),
                "action_logits",
                width,
            ),
            np.asarray(archive["command_delta"], dtype=np.float32),
            score_shift,
        ),
        axis=-1,
    )


def load_prediction_data(
    np: Any, atlas_dirs: list[Path], width: int
) -> dict[str, Any]:
    required = {
        "residual_path_source_layer",
        "residual_path_boundary_layer",
        "residual_path_delta",
        "cache_path_source_layer",
        "cache_path_boundary_layer",
        "cache_key_path_delta",
        "cache_value_path_delta",
        "clean_residual",
        "clean_keys",
        "clean_values",
        "final_residual_delta",
        "safe_feature_delta",
        "action_logit_delta",
        "command_delta",
        "physical_simulator_state",
        "record_id",
        "eligible_causal_outcome",
        "task_failure",
        "operational_silent_failure",
        "score_change_from_control_at_fault",
    }
    runs = [load_json(path / "run.json") for path in atlas_dirs]
    if {run["analysis_split"] for run in runs} != {"development"}:
        raise ValueError("interface prediction only accepts development atlases")
    if len({run["code"]["extractor_sha256"] for run in runs}) != 1:
        raise ValueError("atlas shards were produced by different extractors")

    rows = []
    errors = []
    for atlas_dir, run in zip(atlas_dirs, runs, strict=True):
        for context_id in run["context_ids"]:
            result_dir = atlas_dir / "contexts" / str(context_id)
            complete_path = result_dir / "context.complete.json"
            if not complete_path.is_file():
                error_path = result_dir / "context.error.json"
                errors.append(
                    load_json(error_path)
                    if error_path.is_file()
                    else {"context_id": context_id, "status": "missing"}
                )
                continue
            complete = load_json(complete_path)
            artifact = result_dir / str(complete["artifact"]["name"])
            if file_sha256(artifact) != complete["artifact"]["sha256"]:
                errors.append(
                    {"context_id": context_id, "status": "artifact_hash_mismatch"}
                )
                continue
            with np.load(artifact, allow_pickle=False) as archive:
                missing = sorted(required - set(archive.files))
                if missing:
                    errors.append(
                        {
                            "context_id": context_id,
                            "status": "missing_prediction_arrays",
                            "missing": missing,
                        }
                    )
                    continue
                row = {
                    "context": complete["context"],
                    "path": path_state(np, archive, width),
                    "clean": clean_state(np, archive, width),
                    "fork": fork_state(np, archive, width),
                    "physical": np.asarray(
                        archive["physical_simulator_state"], dtype=np.float32
                    ),
                    "record_id": archive["record_id"].astype(str),
                    "eligible": archive["eligible_causal_outcome"].astype(int),
                    "task_failure": archive["task_failure"].astype(int),
                    "silent_failure": archive["operational_silent_failure"].astype(
                        int
                    ),
                }
                if not all(
                    np.isfinite(row[name]).all()
                    for name in ("path", "clean", "fork", "physical")
                ):
                    errors.append(
                        {"context_id": context_id, "status": "nonfinite_arrays"}
                    )
                    continue
                rows.append(row)
    rows.sort(key=lambda value: str(value["context"]["context_id"]))
    if not rows:
        raise ValueError("no complete atlas contexts were available")
    return {"runs": runs, "rows": rows, "errors": errors}


def boundary_rows(
    np: Any,
    rows: list[dict[str, Any]],
    indices: list[int],
    boundary: int,
) -> tuple[Any, Any, Any, Any]:
    states = []
    targets = []
    clean = []
    tokens = []
    for index in indices:
        row = rows[index]
        count = boundary + 1
        states.append(row["path"][:count, boundary])
        targets.append(row["path"][:count, boundary + 1])
        clean.append(np.repeat(row["clean"][None, boundary], count, axis=0))
        tokens.extend([int(row["context"]["action_token_position"])] * count)
    return (
        np.concatenate(states),
        np.concatenate(clean),
        np.asarray(tokens, dtype=int),
        np.concatenate(targets),
    )


def endpoint_features(
    np: Any,
    rows: list[dict[str, Any]],
    context_indices: list[int],
    state: Any,
    *,
    kind: str,
) -> Any:
    result = []
    position = 0
    for index in context_indices:
        row = rows[index]
        token = int(row["context"]["action_token_position"])
        token_rows = one_hot(np, np.full(32, token), 7)
        source_rows = one_hot(np, np.arange(32), 32)
        current = np.asarray(state[position : position + 32], dtype=np.float64)
        clean_final = np.repeat(row["clean"][31][None, :], 32, axis=0)
        injection = row["path"][np.arange(32), np.arange(32)]
        clean_source = row["clean"][np.arange(32)]
        if kind == "local":
            values = (current, clean_final, token_rows)
        elif kind == "history":
            values = (
                current,
                clean_final,
                token_rows,
                injection,
                source_rows,
            )
        elif kind == "source_context":
            values = (
                current,
                clean_final,
                token_rows,
                injection,
                clean_source,
                source_rows,
            )
        elif kind == "direct":
            values = (
                injection,
                clean_source,
                clean_final,
                token_rows,
                source_rows,
            )
        else:
            raise ValueError(f"unknown endpoint feature kind: {kind}")
        result.append(np.concatenate(values, axis=1))
        position += 32
    return np.concatenate(result)


def stack_context_rows(
    np: Any, rows: list[dict[str, Any]], indices: list[int], name: str
) -> Any:
    return np.concatenate([rows[index][name] for index in indices])


def task_block_physical_state(
    np: Any,
    state: Any,
    *,
    task_id: int,
    coordinate_width: int,
    task_count: int = 10,
) -> Any:
    """Preserve raw state coordinates without equating different task models."""
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if not 0 <= task_id < task_count:
        raise ValueError("task ID is outside the declared LIBERO task range")
    if len(values) > coordinate_width:
        raise ValueError("raw state exceeds the declared per-task coordinate width")
    result = np.zeros(task_count * coordinate_width, dtype=np.float32)
    start = task_id * coordinate_width
    result[start : start + len(values)] = values
    return result


def risk_features(
    np: Any,
    rows: list[dict[str, Any]],
    indices: list[int],
    fork: Any,
) -> tuple[Any, list[dict[str, Any]], Any]:
    features = []
    records = []
    labels = []
    position = 0
    physical_width = max(len(row["physical"].reshape(-1)) for row in rows)
    for index in indices:
        row = rows[index]
        context = row["context"]
        token_rows = one_hot(
            np, np.full(32, int(context["action_token_position"])), 7
        )
        task_rows = one_hot(np, np.full(32, int(context["task_id"])), 10)
        physical_vector = task_block_physical_state(
            np,
            row["physical"],
            task_id=int(context["task_id"]),
            coordinate_width=physical_width,
        )
        physical = np.repeat(physical_vector[None, :], 32, axis=0)
        values = np.concatenate(
            (
                fork[position : position + 32],
                physical,
                token_rows,
                task_rows,
            ),
            axis=1,
        )
        for layer in range(32):
            if int(row["eligible"][layer]) != 1:
                continue
            features.append(values[layer])
            labels.append(int(row["silent_failure"][layer]))
            records.append(
                {
                    "context_id": context["context_id"],
                    "task_id": int(context["task_id"]),
                    "episode_index": int(context["episode_index"]),
                    "phase": context["phase"],
                    "action_token_position": int(
                        context["action_token_position"]
                    ),
                    "source_layer": layer,
                    "record_id": str(row["record_id"][layer]),
                    "operational_silent_failure": bool(labels[-1]),
                }
            )
        position += 32
    return np.asarray(features), records, np.asarray(labels, dtype=int)


def local_risk_features(
    np: Any,
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    fork_width: int,
) -> tuple[Any, list[dict[str, Any]], Any]:
    base = []
    for index in indices:
        row = rows[index]
        injection = row["path"][np.arange(32), np.arange(32)]
        source = one_hot(np, np.arange(32), 32)
        base.extend(
            np.concatenate((injection[layer], source[layer])).tolist()
            for layer in range(32)
            if int(row["eligible"][layer]) == 1
        )
    blank_fork = np.zeros((len(indices) * 32, fork_width), dtype=np.float32)
    context_features, records, labels = risk_features(
        np, rows, indices, blank_fork
    )
    return (
        np.concatenate(
            (np.asarray(base), context_features[:, fork_width:]), axis=1
        ),
        records,
        labels,
    )
