from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_silent_failures.language_campaign import (
    ACTION_TOKEN_COUNT,
    LANGUAGE_BLOCK_COUNT,
)


HIDDEN_SIZE = 4096
ATTENTION_HEADS = 32
ATTENTION_HEAD_SIZE = 128
ACTION_VOCABULARY_SIZE = 256

_ARCHIVE_ARRAYS = {
    "fault_layer",
    "clean_residuals",
    "source_residuals",
    "clean_block_inputs",
    "clean_post_attention_residuals",
    "clean_attention_cache_keys",
    "clean_attention_cache_values",
    "clean_action_logits",
    "clean_sequence_token_ids",
    "fault_residuals",
    "fault_residuals_row_intervention",
    "fault_residuals_row_layer",
    "fault_residuals_row_token",
    "fault_post_attention_residuals",
    "fault_post_attention_residuals_row_intervention",
    "fault_post_attention_residuals_row_layer",
    "fault_post_attention_residuals_row_token",
    "fault_attention_cache_keys",
    "fault_attention_cache_keys_row_intervention",
    "fault_attention_cache_keys_row_layer",
    "fault_attention_cache_keys_row_token",
    "fault_attention_cache_values",
    "fault_attention_cache_values_row_intervention",
    "fault_attention_cache_values_row_layer",
    "fault_attention_cache_values_row_token",
    "fault_action_logits",
    "fault_sequence_token_ids",
    "clean_decoder_position_ids_values",
    "clean_decoder_position_ids_offsets",
    "clean_decoder_position_ids_shapes",
    "clean_decoder_position_ids_call_indices",
    "clean_decoder_cache_position_values",
    "clean_decoder_cache_position_offsets",
    "clean_decoder_cache_position_shapes",
    "clean_decoder_cache_position_call_indices",
    "clean_decoder_attention_mask_present",
    "clean_prompt_cache_keys",
    "clean_prompt_cache_values",
    "clean_initial_language_input",
}


def load_context_arrays(np: Any, path: Path) -> dict[str, Any]:
    """Load one context once so compressed arrays are not reopened inside JVPs."""
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_ARCHIVE_ARRAYS - set(archive.files))
        if missing:
            raise ValueError(f"context archive omits required arrays: {missing}")
        result = {name: archive[name] for name in _ARCHIVE_ARRAYS}
        for suffix in (
            "values",
            "offsets",
            "shapes",
            "call_indices",
        ):
            name = f"clean_decoder_attention_mask_{suffix}"
            if name in archive.files:
                result[name] = archive[name]
    return result


def decode_bfloat16(torch: Any, words: Any, device: Any) -> Any:
    """Restore exact BF16 words written by language_interface_archive.py."""
    tensor = torch.from_numpy(words.copy()).view(torch.bfloat16)
    return tensor.to(device)


def ragged_call(np: Any, arrays: dict[str, Any], name: str, call: int) -> Any:
    calls = arrays[f"clean_{name}_call_indices"]
    rows = np.flatnonzero(calls == call)
    if len(rows) != 1:
        raise ValueError(f"expected one {name} record for call {call}")
    row = int(rows[0])
    offsets = arrays[f"clean_{name}_offsets"]
    values = arrays[f"clean_{name}_values"][offsets[row] : offsets[row + 1]]
    shape = tuple(int(value) for value in arrays[f"clean_{name}_shapes"][row])
    return values.reshape(shape)


def sparse_rows(np: Any, arrays: dict[str, Any], name: str) -> dict[tuple[int, int, int], int]:
    keys = zip(
        arrays[f"fault_{name}_row_intervention"].tolist(),
        arrays[f"fault_{name}_row_layer"].tolist(),
        arrays[f"fault_{name}_row_token"].tolist(),
        strict=True,
    )
    result = {
        (int(intervention), int(layer), int(token)): row
        for row, (intervention, layer, token) in enumerate(keys)
    }
    if len(result) != len(arrays[f"fault_{name}"]):
        raise ValueError(f"fault {name} coordinates are duplicated")
    return result


def sparse_value(
    arrays: dict[str, Any],
    indices: dict[tuple[int, int, int], int],
    name: str,
    intervention: int,
    layer: int,
    token: int,
) -> Any:
    key = (intervention, layer, token)
    if key not in indices:
        raise ValueError(f"fault {name} has no row for coordinate {key}")
    return arrays[f"fault_{name}"][indices[key]]


def output_names(source_layer: int) -> list[dict[str, Any]]:
    if not 0 <= source_layer < LANGUAGE_BLOCK_COUNT:
        raise ValueError("source layer is outside the OpenVLA language backbone")
    names = []
    for layer in range(source_layer + 1, LANGUAGE_BLOCK_COUNT):
        names.extend(
            (
                {"family": "post_attention_residual", "layer_index": layer},
                {"family": "post_block_residual", "layer_index": layer},
                {"family": "current_token_key", "layer_index": layer},
                {"family": "current_token_value", "layer_index": layer},
            )
        )
    names.extend(
        (
            {"family": "selected_token_final_feature", "layer_index": 31},
            {"family": "selected_token_action_logits", "layer_index": None},
        )
    )
    return names


def approximation_metrics(np: Any, actual: Any, predicted: Any) -> dict[str, Any]:
    truth = np.asarray(actual, dtype=np.float64).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if truth.shape != estimate.shape:
        raise ValueError("approximation vectors have different shapes")
    finite = bool(np.isfinite(truth).all() and np.isfinite(estimate).all())
    if not finite:
        return {
            "finite": False,
            "actual_l2": None,
            "predicted_l2": None,
            "error_l2": None,
            "normalized_error": None,
            "cosine": None,
            "norm_ratio": None,
        }
    actual_l2 = float(np.linalg.norm(truth))
    predicted_l2 = float(np.linalg.norm(estimate))
    error_l2 = float(np.linalg.norm(estimate - truth))
    denominator = actual_l2 * predicted_l2
    return {
        "finite": True,
        "actual_l2": actual_l2,
        "predicted_l2": predicted_l2,
        "error_l2": error_l2,
        "normalized_error": error_l2 / actual_l2 if actual_l2 else None,
        "cosine": (
            float(np.dot(truth, estimate) / denominator) if denominator else None
        ),
        "norm_ratio": predicted_l2 / actual_l2 if actual_l2 else None,
    }


def reconstruction_metrics(torch: Any, actual: Any, replayed: Any) -> dict[str, Any]:
    difference = replayed.float() - actual.float()
    return {
        "exact_equal": bool(torch.equal(actual, replayed)),
        "maximum_absolute_error": float(difference.abs().max()),
        "error_l2": float(torch.linalg.vector_norm(difference)),
    }


def perturbation_summary(torch: Any, value: Any) -> dict[str, Any]:
    flattened = value.float().reshape(-1)
    return {
        "l2": float(torch.linalg.vector_norm(flattened)),
        "maximum_absolute_value": float(flattened.abs().max()),
        "changed_element_count": int(torch.count_nonzero(flattened)),
        "element_count": int(flattened.numel()),
        "finite": bool(torch.isfinite(flattened).all()),
    }


def decoder_modules(model: Any) -> tuple[Any, Any, Any]:
    language_model = model.language_model
    layers = language_model.model.layers
    if len(layers) != LANGUAGE_BLOCK_COUNT:
        raise ValueError(
            f"pinned OpenVLA has {len(layers)} language blocks, expected 32"
        )
    return layers, language_model.model.norm, language_model.lm_head


def context_tensors(
    np: Any,
    torch: Any,
    arrays: dict[str, Any],
    token: int,
    device: Any,
) -> dict[str, Any]:
    if not 0 <= token < ACTION_TOKEN_COUNT:
        raise ValueError("action-token position must be between zero and six")
    prompt_keys = decode_bfloat16(
        torch, arrays["clean_prompt_cache_keys"], device
    )
    prompt_values = decode_bfloat16(
        torch, arrays["clean_prompt_cache_values"], device
    )
    initial_language_input = (
        decode_bfloat16(torch, arrays["clean_initial_language_input"], device)
        if token == 0
        else None
    )
    current_keys = decode_bfloat16(
        torch, arrays["clean_attention_cache_keys"], device
    )
    current_values = decode_bfloat16(
        torch, arrays["clean_attention_cache_values"], device
    )

    keys = []
    values = []
    for layer in range(LANGUAGE_BLOCK_COUNT):
        key = prompt_keys[layer]
        value = prompt_values[layer]
        if token == 0:
            # The prompt cache was captured after generation call zero. Removing
            # its last entry reconstructs the cache immediately before the final
            # prompt position whose logits produce action token zero.
            key = key[..., :-1, :]
            value = value[..., :-1, :]
        elif token > 1:
            key = torch.cat(
                (
                    key,
                    current_keys[layer, 1:token]
                    .permute(1, 0, 2)
                    .unsqueeze(0),
                ),
                dim=-2,
            )
            value = torch.cat(
                (
                    value,
                    current_values[layer, 1:token]
                    .permute(1, 0, 2)
                    .unsqueeze(0),
                ),
                dim=-2,
            )
        keys.append(key)
        values.append(value)

    full_position_ids = torch.from_numpy(
        ragged_call(np, arrays, "decoder_position_ids", token).copy()
    ).to(device)
    full_cache_position = torch.from_numpy(
        ragged_call(np, arrays, "decoder_cache_position", token).copy()
    ).to(device)
    position_ids = full_position_ids[:, -1:]
    cache_position = full_cache_position[-1:]

    full_attention_mask = None
    if bool(arrays["clean_decoder_attention_mask_present"][token]):
        full_attention_mask = torch.from_numpy(
            ragged_call(np, arrays, "decoder_attention_mask", token).copy()
        ).to(device)
    attention_mask = full_attention_mask
    if attention_mask is not None:
        if attention_mask.ndim >= 3:
            attention_mask = attention_mask[..., -1:, :]
    return {
        "action_token_position": token,
        "prefix_keys": tuple(keys),
        "prefix_values": tuple(values),
        "prompt_keys": tuple(prompt_keys),
        "prompt_values": tuple(prompt_values),
        "position_ids": position_ids,
        "cache_position": cache_position,
        "attention_mask": attention_mask,
        "full_position_ids": full_position_ids,
        "full_cache_position": full_cache_position,
        "full_attention_mask": full_attention_mask,
        "initial_language_input": initial_language_input,
        "clean_current_keys": current_keys,
        "clean_current_values": current_values,
    }


def clean_full_prompt_states(
    torch: Any, layers: Any, context: dict[str, Any]
) -> tuple[Any, ...]:
    """Replay call zero with its original full-sequence tensor shapes."""
    from transformers.cache_utils import DynamicCache

    hidden = context["initial_language_input"]
    cache = DynamicCache()
    states = []
    with torch.no_grad():
        for layer in layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=context["full_attention_mask"],
                position_ids=context["full_position_ids"],
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=context["full_cache_position"],
            )[0]
            states.append(hidden.detach())
    return tuple(states)


def decoder_path(
    torch: Any,
    layers: Any,
    final_norm: Any,
    lm_head: Any,
    context: dict[str, Any],
    source_layer: int,
) -> Any:
    from transformers.cache_utils import DynamicCache

    full_prompt_states = context.get("full_prompt_states")
    use_full_prompt = context["action_token_position"] == 0
    if use_full_prompt and full_prompt_states is None:
        raise ValueError("token-zero replay requires full prompt states")

    def path(hidden: Any) -> tuple[Any, ...]:
        cache = DynamicCache()
        if use_full_prompt:
            source_state = full_prompt_states[source_layer]
            if tuple(hidden.shape) != (1, 1, HIDDEN_SIZE):
                raise ValueError("token-zero source vector has an unexpected shape")
            hidden = torch.cat((source_state[:, :-1, :], hidden), dim=1)
            cache.key_cache = list(context["prompt_keys"][: source_layer + 1])
            cache.value_cache = list(context["prompt_values"][: source_layer + 1])
            position_ids = context["full_position_ids"]
            cache_position = context["full_cache_position"]
            attention_mask = context["full_attention_mask"]
        else:
            cache.key_cache = list(context["prefix_keys"])
            cache.value_cache = list(context["prefix_values"])
            position_ids = context["position_ids"]
            cache_position = context["cache_position"]
            attention_mask = context["attention_mask"]
        cache._seen_tokens = int(cache.key_cache[0].shape[-2])
        outputs = []
        for layer_index in range(source_layer + 1, LANGUAGE_BLOCK_COUNT):
            layer = layers[layer_index]

            # Transformers 4.40.1, modeling_llama.py::LlamaDecoderLayer.forward,
            # defines these two residual sublayers in this exact order. Keeping
            # their existing module calls preserves the pinned implementation;
            # the split only exposes the already-recorded attention/MLP cut.
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention, _weights, _cache = layer.self_attn(
                hidden_states=normalized,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
            )
            post_attention = residual + attention
            hidden = post_attention + layer.mlp(
                layer.post_attention_layernorm(post_attention)
            )
            outputs.extend(
                (
                    post_attention[:, -1:, :],
                    hidden[:, -1:, :],
                    cache.key_cache[layer_index][..., -1:, :],
                    cache.value_cache[layer_index][..., -1:, :],
                )
            )
        feature = final_norm(hidden)
        logits = lm_head(feature)[..., -ACTION_VOCABULARY_SIZE:]
        return (*outputs, feature[:, -1:, :], logits[:, -1:, :])

    return path


def expected_path(
    torch: Any,
    arrays: dict[str, Any],
    indices: dict[str, dict[tuple[int, int, int], int]],
    source_layer: int,
    token: int,
    intervention: int,
    *,
    condition: str,
    final_norm: Any,
    device: Any,
) -> tuple[Any, ...]:
    if condition not in {"clean", "fault"}:
        raise ValueError("expected path condition must be clean or fault")
    outputs = []
    for layer in range(source_layer + 1, LANGUAGE_BLOCK_COUNT):
        if condition == "clean":
            post_attention = arrays["clean_post_attention_residuals"][layer, token]
            residual = arrays["clean_residuals"][layer, token]
            key = arrays["clean_attention_cache_keys"][layer, token]
            value = arrays["clean_attention_cache_values"][layer, token]
        else:
            post_attention = sparse_value(
                arrays,
                indices["post_attention_residuals"],
                "post_attention_residuals",
                intervention,
                layer,
                token,
            )
            residual = sparse_value(
                arrays,
                indices["residuals"],
                "residuals",
                intervention,
                layer,
                token,
            )
            key = sparse_value(
                arrays,
                indices["attention_cache_keys"],
                "attention_cache_keys",
                intervention,
                layer,
                token,
            )
            value = sparse_value(
                arrays,
                indices["attention_cache_values"],
                "attention_cache_values",
                intervention,
                layer,
                token,
            )
        outputs.extend(
            (
                decode_bfloat16(torch, post_attention, device).reshape(1, 1, -1),
                decode_bfloat16(torch, residual, device).reshape(1, 1, -1),
                decode_bfloat16(torch, key, device).reshape(
                    1, ATTENTION_HEADS, 1, ATTENTION_HEAD_SIZE
                ),
                decode_bfloat16(torch, value, device).reshape(
                    1, ATTENTION_HEADS, 1, ATTENTION_HEAD_SIZE
                ),
            )
        )

    final_words = (
        arrays["clean_residuals"][31, token]
        if condition == "clean"
        else (
            arrays["source_residuals"][31, token]
            if source_layer == 31
            else sparse_value(
                arrays,
                indices["residuals"],
                "residuals",
                intervention,
                31,
                token,
            )
        )
    )
    final_residual = decode_bfloat16(torch, final_words, device).reshape(1, 1, -1)
    with torch.no_grad():
        feature = final_norm(final_residual)
    logits = torch.from_numpy(
        (
            arrays["clean_action_logits"][0, token]
            if condition == "clean"
            else arrays["fault_action_logits"][intervention, token]
        ).copy()
    ).to(device).reshape(1, 1, -1)
    return (*outputs, feature, logits)


def analyze_intervention(
    np: Any,
    torch: Any,
    model: Any,
    arrays: dict[str, Any],
    context: dict[str, Any],
    indices: dict[str, dict[tuple[int, int, int], int]],
    source_layer: int,
    token: int,
) -> dict[str, Any]:
    layers, final_norm, lm_head = decoder_modules(model)
    interventions = np.flatnonzero(arrays["fault_layer"] == source_layer)
    if len(interventions) != 1:
        raise ValueError(f"archive has {len(interventions)} records for layer {source_layer}")
    intervention = int(interventions[0])
    device = next(model.parameters()).device
    clean_source = decode_bfloat16(
        torch, arrays["clean_residuals"][source_layer, token], device
    ).reshape(1, 1, HIDDEN_SIZE)
    fault_source = decode_bfloat16(
        torch, arrays["source_residuals"][source_layer, token], device
    ).reshape(1, 1, HIDDEN_SIZE)
    source_reconstruction = (
        reconstruction_metrics(
            torch,
            clean_source,
            context["full_prompt_states"][source_layer][:, -1:, :],
        )
        if token == 0
        else reconstruction_metrics(torch, clean_source, clean_source)
    )
    path = decoder_path(
        torch, layers, final_norm, lm_head, context, source_layer
    )
    names = output_names(source_layer)
    clean_expected = expected_path(
        torch,
        arrays,
        indices,
        source_layer,
        token,
        intervention,
        condition="clean",
        final_norm=final_norm,
        device=device,
    )
    fault_expected = expected_path(
        torch,
        arrays,
        indices,
        source_layer,
        token,
        intervention,
        condition="fault",
        final_norm=final_norm,
        device=device,
    )

    # Pearlmutter's JVP computes the chain-rule product through the existing
    # network at the clean state. It is a first-order prediction of the finite
    # t-1 replacement, not a fitted transition model or an exact replay.
    primal, tangent = torch.autograd.functional.jvp(
        path,
        clean_source,
        fault_source - clean_source,
        create_graph=False,
        strict=True,
    )
    with torch.no_grad():
        fault_replay = path(fault_source)

    records = []
    reconstruction_valid = source_reconstruction["exact_equal"]
    for name, clean, fault, replayed_clean, replayed_fault, estimate in zip(
        names,
        clean_expected,
        fault_expected,
        primal,
        fault_replay,
        tangent,
        strict=True,
    ):
        clean_check = reconstruction_metrics(torch, clean, replayed_clean)
        fault_check = reconstruction_metrics(torch, fault, replayed_fault)
        reconstruction_valid = bool(
            reconstruction_valid
            and clean_check["exact_equal"]
            and fault_check["exact_equal"]
        )
        actual_delta = (fault.float() - clean.float()).detach().cpu().numpy()
        predicted_delta = estimate.float().detach().cpu().numpy()
        records.append(
            {
                **name,
                "clean_reconstruction": clean_check,
                "fault_reconstruction": fault_check,
                "first_order": approximation_metrics(
                    np, actual_delta, predicted_delta
                ),
            }
        )

    clean_logits = primal[-1].float()
    fault_logits = fault_expected[-1].float()
    predicted_logits = clean_logits + tangent[-1].float()
    vocabulary_start = int(lm_head.out_features) - ACTION_VOCABULARY_SIZE
    clean_argmax = vocabulary_start + int(clean_logits.reshape(-1).argmax())
    fault_argmax = vocabulary_start + int(fault_logits.reshape(-1).argmax())
    predicted_token = vocabulary_start + int(predicted_logits.reshape(-1).argmax())
    clean_token = int(arrays["clean_sequence_token_ids"][0, -ACTION_TOKEN_COUNT + token])
    fault_token = int(
        arrays["fault_sequence_token_ids"][intervention, -ACTION_TOKEN_COUNT + token]
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "source_layer": source_layer,
        "action_token_position": token,
        "path_blocks": LANGUAGE_BLOCK_COUNT - source_layer - 1,
        "factorization": (
            "attention residual update, MLP residual update, final norm, and LM head"
        ),
        "method": (
            "reverse-over-reverse Jacobian-vector product at the clean execution"
        ),
        "reconstruction_valid": reconstruction_valid,
        "source_reconstruction": source_reconstruction,
        "source_perturbation": perturbation_summary(
            torch, fault_source - clean_source
        ),
        "selected_token": {
            "clean_token_id": clean_token,
            "fault_token_id": fault_token,
            "predicted_token_id": predicted_token,
            "clean_logits_argmax_id": clean_argmax,
            "fault_logits_argmax_id": fault_argmax,
            "clean_archive_consistent": clean_argmax == clean_token,
            "fault_archive_consistent": fault_argmax == fault_token,
            "fault_changed_token": fault_token != clean_token,
            "prediction_matches_fault": predicted_token == fault_token,
        },
        "outputs": records,
    }
