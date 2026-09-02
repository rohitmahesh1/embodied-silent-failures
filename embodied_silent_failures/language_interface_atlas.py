from __future__ import annotations

from typing import Any

from embodied_silent_failures.analyze_language_campaign import analysis_row


LANGUAGE_BLOCK_COUNT = 32
ACTION_TOKEN_COUNT = 7


def bfloat16_words_to_float32(np: Any, value: Any) -> Any:
    """Decode exact bfloat16 bit patterns without requiring PyTorch."""
    words = np.asarray(value)
    if words.dtype.kind != "i" or words.dtype.itemsize != 2:
        raise ValueError(f"expected signed 16-bit bfloat16 words, got {words.dtype}")
    bits = words.astype(np.uint16, copy=False).astype(np.uint32) << 16
    return bits.view(np.float32)


def _row_index(archive: Any, prefix: str) -> dict[tuple[int, int, int], int]:
    owners = archive[f"{prefix}_row_intervention"].astype(int)
    layers = archive[f"{prefix}_row_layer"].astype(int)
    tokens = archive[f"{prefix}_row_token"].astype(int)
    keys = list(zip(owners.tolist(), layers.tolist(), tokens.tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{prefix} contains duplicate sparse coordinates")
    return {key: index for index, key in enumerate(keys)}


def _fault_rows(
    archive: Any,
    name: str,
    index: dict[tuple[int, int, int], int],
    keys: list[tuple[int, int, int]],
) -> Any:
    missing = [key for key in keys if key not in index]
    if missing:
        raise ValueError(f"{name} is missing sparse coordinates {missing[:5]}")
    return archive[name][[index[key] for key in keys]]


def _optional_bool(np: Any, values: list[Any]) -> Any:
    return np.asarray(
        [-1 if value is None else int(bool(value)) for value in values],
        dtype=np.int8,
    )


def _optional_float(np: Any, values: list[Any]) -> Any:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float32,
    )


def score_arrays(np: Any, score_document: dict[str, Any], context_id: str) -> dict[str, Any]:
    primary_alpha = format(float(score_document["monitor"]["primary_alpha"]), "g")
    records = sorted(
        (
            analysis_row(record, primary_alpha)
            for record in score_document["records"]
            if str(record["context_id"]) == context_id
        ),
        key=lambda value: int(value["layer_index"]),
    )
    if [int(value["layer_index"]) for value in records] != list(
        range(LANGUAGE_BLOCK_COUNT)
    ):
        raise ValueError(f"scores do not cover all language blocks for {context_id}")
    return {
        "record_id": np.asarray([value["record_id"] for value in records]),
        "eligible_causal_outcome": _optional_bool(
            np, [value["eligible_causal_outcome"] for value in records]
        ),
        "command_changed": _optional_bool(
            np, [value["command_changed"] for value in records]
        ),
        "task_failure": _optional_bool(
            np, [value["task_failure"] for value in records]
        ),
        "safe_alarm_at_fault": _optional_bool(
            np, [value["safe_alarm_at_fault"] for value in records]
        ),
        "safe_alarm_within_10": _optional_bool(
            np, [value["safe_alarm_within_10"] for value in records]
        ),
        "safe_alarm_post_fault_any": _optional_bool(
            np, [value["safe_alarm_post_fault_any"] for value in records]
        ),
        "operational_silent_failure": _optional_bool(
            np, [value["operational_silent_failure"] for value in records]
        ),
        "score_at_fault": _optional_float(
            np, [value["score_at_fault"] for value in records]
        ),
        "control_score_at_fault": _optional_float(
            np, [value["control_score_at_fault"] for value in records]
        ),
        "score_change_from_control_at_fault": _optional_float(
            np, [value["score_change_from_control_at_fault"] for value in records]
        ),
    }


def context_arrays(
    np: Any,
    archive: Any,
    local: dict[str, Any],
    score_document: dict[str, Any],
    safe_features: dict[str, Any],
) -> dict[str, Any]:
    """Reduce one exact trace archive to mechanically declared interface cuts."""
    context = local["context"]
    context_id = str(context["context_id"])
    token = int(context["action_token_position"])
    if not 0 <= token < ACTION_TOKEN_COUNT:
        raise ValueError(f"invalid action-token position for {context_id}: {token}")

    fault_layers = archive["fault_layer"].astype(int)
    if sorted(fault_layers.tolist()) != list(range(LANGUAGE_BLOCK_COUNT)):
        raise ValueError(f"fault archive does not cover every block for {context_id}")
    owner_by_layer = {layer: owner for owner, layer in enumerate(fault_layers)}

    residual_index = _row_index(archive, "fault_residuals")
    key_index = _row_index(archive, "fault_attention_cache_keys")
    value_index = _row_index(archive, "fault_attention_cache_values")

    clean_residual_words = archive["clean_residuals"]
    clean_key_words = archive["clean_attention_cache_keys"]
    clean_value_words = archive["clean_attention_cache_values"]
    source_residual_words = archive["source_residuals"]

    injection_keys = [
        (owner_by_layer[layer], layer, token)
        for layer in range(LANGUAGE_BLOCK_COUNT)
    ]
    injection_words = _fault_rows(
        archive, "fault_residuals", residual_index, injection_keys
    )
    source_words = source_residual_words[:, token]
    if not np.array_equal(injection_words, source_words):
        raise ValueError(
            f"fault injection rows do not reproduce the archived source for {context_id}"
        )

    immediate_keys = [
        (owner_by_layer[layer], layer + 1, token)
        for layer in range(LANGUAGE_BLOCK_COUNT - 1)
    ]
    immediate_residual_words = _fault_rows(
        archive, "fault_residuals", residual_index, immediate_keys
    )
    immediate_key_words = _fault_rows(
        archive, "fault_attention_cache_keys", key_index, immediate_keys
    )
    immediate_value_words = _fault_rows(
        archive, "fault_attention_cache_values", value_index, immediate_keys
    )

    residual_path_coordinates = [
        (source_layer, boundary_layer)
        for source_layer in range(LANGUAGE_BLOCK_COUNT)
        for boundary_layer in range(source_layer, LANGUAGE_BLOCK_COUNT)
    ]
    residual_path_keys = [
        (owner_by_layer[source], boundary, token)
        for source, boundary in residual_path_coordinates
    ]
    residual_path_words = _fault_rows(
        archive, "fault_residuals", residual_index, residual_path_keys
    )
    cache_path_coordinates = [
        (source_layer, boundary_layer)
        for source_layer in range(LANGUAGE_BLOCK_COUNT - 1)
        for boundary_layer in range(source_layer + 1, LANGUAGE_BLOCK_COUNT)
    ]
    cache_path_keys = [
        (owner_by_layer[source], boundary, token)
        for source, boundary in cache_path_coordinates
    ]
    cache_key_path_words = _fault_rows(
        archive, "fault_attention_cache_keys", key_index, cache_path_keys
    )
    cache_value_path_words = _fault_rows(
        archive, "fault_attention_cache_values", value_index, cache_path_keys
    )

    final_residual_words = np.broadcast_to(
        clean_residual_words[LANGUAGE_BLOCK_COUNT - 1],
        (LANGUAGE_BLOCK_COUNT,) + clean_residual_words.shape[1:],
    ).copy()
    for layer in range(LANGUAGE_BLOCK_COUNT):
        keys = [
            (owner_by_layer[layer], LANGUAGE_BLOCK_COUNT - 1, current_token)
            for current_token in range(token, ACTION_TOKEN_COUNT)
        ]
        final_residual_words[layer, token:] = _fault_rows(
            archive, "fault_residuals", residual_index, keys
        )

    clean_residual = bfloat16_words_to_float32(np, clean_residual_words[:, token])
    clean_keys = bfloat16_words_to_float32(np, clean_key_words[:, token])
    clean_values = bfloat16_words_to_float32(np, clean_value_words[:, token])
    clean_final_residual = bfloat16_words_to_float32(
        np, clean_residual_words[LANGUAGE_BLOCK_COUNT - 1, -1]
    )
    clean_logits = np.asarray(archive["clean_action_logits"][0], dtype=np.float32)

    clean_safe = np.asarray(safe_features["clean"], dtype=np.float32)
    fault_safe = np.asarray(safe_features["fault"], dtype=np.float32)
    if clean_safe.ndim != 2 or clean_safe.shape[0] != ACTION_TOKEN_COUNT:
        raise ValueError(f"clean SAFE feature has the wrong shape for {context_id}")
    if fault_safe.shape != (LANGUAGE_BLOCK_COUNT,) + clean_safe.shape:
        raise ValueError(f"faulted SAFE features have the wrong shape for {context_id}")

    local_records = sorted(
        local["interventions"], key=lambda value: int(value.get("layer_index", -1))
    )
    if len(local_records) != LANGUAGE_BLOCK_COUNT or any(
        value.get("status") != "complete" for value in local_records
    ):
        raise ValueError(f"local intervention records are incomplete for {context_id}")
    command_delta = np.asarray(
        [
            np.asarray(value["faulted_executed_command"], dtype=np.float64)
            - np.asarray(value["clean_executed_command"], dtype=np.float64)
            for value in local_records
        ],
        dtype=np.float64,
    )

    arrays = {
        "injection_residual_delta": bfloat16_words_to_float32(np, injection_words)
        - clean_residual,
        "immediate_residual_delta": bfloat16_words_to_float32(
            np, immediate_residual_words
        )
        - clean_residual[1:],
        "immediate_key_delta": bfloat16_words_to_float32(np, immediate_key_words)
        - clean_keys[1:],
        "immediate_value_delta": bfloat16_words_to_float32(
            np, immediate_value_words
        )
        - clean_values[1:],
        "residual_path_source_layer": np.asarray(
            [source for source, _boundary in residual_path_coordinates],
            dtype=np.int8,
        ),
        "residual_path_boundary_layer": np.asarray(
            [boundary for _source, boundary in residual_path_coordinates],
            dtype=np.int8,
        ),
        "residual_path_delta": bfloat16_words_to_float32(
            np, residual_path_words
        )
        - clean_residual[
            [boundary for _source, boundary in residual_path_coordinates]
        ],
        "cache_path_source_layer": np.asarray(
            [source for source, _boundary in cache_path_coordinates], dtype=np.int8
        ),
        "cache_path_boundary_layer": np.asarray(
            [boundary for _source, boundary in cache_path_coordinates], dtype=np.int8
        ),
        "cache_key_path_delta": bfloat16_words_to_float32(
            np, cache_key_path_words
        )
        - clean_keys[[boundary for _source, boundary in cache_path_coordinates]],
        "cache_value_path_delta": bfloat16_words_to_float32(
            np, cache_value_path_words
        )
        - clean_values[[boundary for _source, boundary in cache_path_coordinates]],
        "final_residual_delta": bfloat16_words_to_float32(
            np, final_residual_words[:, -1]
        )
        - clean_final_residual[None, :],
        "safe_feature_delta": fault_safe[:, -1] - clean_safe[None, -1],
        "action_logit_delta": np.asarray(
            archive["fault_action_logits"], dtype=np.float32
        )
        - clean_logits[None, :, :],
        "command_delta": command_delta,
        "clean_residual": clean_residual,
        "clean_keys": clean_keys,
        "clean_values": clean_values,
        "clean_final_residual": clean_final_residual,
        "clean_safe_feature": clean_safe[-1],
    }
    arrays.update(score_arrays(np, score_document, context_id))
    return arrays
