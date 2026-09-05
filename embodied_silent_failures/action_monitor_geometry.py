from __future__ import annotations

import math
from typing import Any


ACTION_TOKEN_COUNT = 7
ACTION_VOCABULARY_SIZE = 256
SAFE_TOKEN_INDEX = 6


def _logsumexp(np: Any, values: Any) -> Any:
    maximum = np.max(values, axis=-1, keepdims=True)
    return np.squeeze(maximum, axis=-1) + np.log(
        np.exp(values - maximum).sum(axis=-1)
    )


def conditional_js_divergence(np: Any, clean: Any, faulted: Any) -> Any:
    """Jensen-Shannon divergence within OpenVLA's 256 action tokens."""
    clean = np.asarray(clean, dtype=np.float64)
    faulted = np.asarray(faulted, dtype=np.float64)
    if clean.shape != faulted.shape or clean.shape[-1] != ACTION_VOCABULARY_SIZE:
        raise ValueError("paired action logits must have matching [..., 256] shape")
    clean_log = clean - _logsumexp(np, clean)[..., None]
    faulted_log = faulted - _logsumexp(np, faulted)[..., None]
    mixture_log = np.logaddexp(clean_log, faulted_log) - math.log(2.0)
    clean_probability = np.exp(clean_log)
    faulted_probability = np.exp(faulted_log)
    return 0.5 * (
        (clean_probability * (clean_log - mixture_log)).sum(axis=-1)
        + (faulted_probability * (faulted_log - mixture_log)).sum(axis=-1)
    )


def _choice_margin(np: Any, logits: Any, choices: Any) -> Any:
    values = np.asarray(logits, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.int64)
    selected = np.take_along_axis(values, choices[..., None], axis=-1)[..., 0]
    alternatives = values.copy()
    np.put_along_axis(alternatives, choices[..., None], -np.inf, axis=-1)
    return selected - alternatives.max(axis=-1)


def action_monitor_arrays(
    np: Any,
    *,
    clean_logits: Any,
    faulted_logits: Any,
    clean_tokens: Any,
    faulted_tokens: Any,
    clean_raw_action: Any,
    faulted_raw_action: Any,
    clean_command: Any,
    faulted_command: Any,
    clean_full_log_normalizer: Any,
    faulted_full_log_normalizer: Any,
) -> dict[str, Any]:
    clean_logits = np.asarray(clean_logits, dtype=np.float64)
    faulted_logits = np.asarray(faulted_logits, dtype=np.float64)
    expected = (clean_logits.shape[0], ACTION_TOKEN_COUNT, ACTION_VOCABULARY_SIZE)
    if clean_logits.shape != expected or faulted_logits.shape != expected:
        raise ValueError("action logits must have [step, 7, 256] shape")

    clean_tokens = np.asarray(clean_tokens)
    faulted_tokens = np.asarray(faulted_tokens)
    if clean_tokens.shape != expected[:2] or faulted_tokens.shape != expected[:2]:
        raise ValueError("action tokens must have [step, 7] shape")

    clean_choices = clean_logits.argmax(axis=-1)
    faulted_choices = faulted_logits.argmax(axis=-1)
    clean_margin = _choice_margin(np, clean_logits, clean_choices)
    faulted_clean_choice_margin = _choice_margin(
        np, faulted_logits, clean_choices
    )
    delta = faulted_logits - clean_logits
    scale = 0.5 * (
        np.linalg.norm(clean_logits, axis=-1)
        + np.linalg.norm(faulted_logits, axis=-1)
    )
    action_log_mass_clean = (
        _logsumexp(np, clean_logits)
        - np.asarray(clean_full_log_normalizer, dtype=np.float64)
    )
    action_log_mass_faulted = (
        _logsumexp(np, faulted_logits)
        - np.asarray(faulted_full_log_normalizer, dtype=np.float64)
    )
    return {
        "conditional_action_js": conditional_js_divergence(
            np, clean_logits, faulted_logits
        ).astype(np.float32),
        "action_logit_l2": np.linalg.norm(delta, axis=-1).astype(np.float32),
        "action_logit_symmetric_normalized_l2": (
            np.linalg.norm(delta, axis=-1)
            / np.maximum(scale, np.finfo(np.float64).eps)
        ).astype(np.float32),
        "clean_choice_margin": clean_margin.astype(np.float32),
        "faulted_clean_choice_margin": faulted_clean_choice_margin.astype(
            np.float32
        ),
        "clean_choice_margin_erosion": (
            clean_margin - faulted_clean_choice_margin
        ).astype(np.float32),
        "action_argmax_changed": (clean_choices != faulted_choices),
        "generated_action_token_changed": (clean_tokens != faulted_tokens),
        "action_log_mass_delta": (
            action_log_mass_faulted - action_log_mass_clean
        ).astype(np.float32),
        "raw_action_l2": np.linalg.norm(
            np.asarray(faulted_raw_action, dtype=np.float64)
            - np.asarray(clean_raw_action, dtype=np.float64),
            axis=-1,
        ).astype(np.float32),
        "executed_command_l2": np.linalg.norm(
            np.asarray(faulted_command, dtype=np.float64)
            - np.asarray(clean_command, dtype=np.float64),
            axis=-1,
        ).astype(np.float32),
    }


def summarize_action_monitor_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    js = np.asarray(arrays["conditional_action_js"])
    l2 = np.asarray(arrays["action_logit_l2"])
    erosion = np.asarray(arrays["clean_choice_margin_erosion"])
    argmax_changed = np.asarray(arrays["action_argmax_changed"])
    tokens_changed = np.asarray(arrays["generated_action_token_changed"])
    command_l2 = np.asarray(arrays["executed_command_l2"])
    if js.ndim != 2 or js.shape[1] != ACTION_TOKEN_COUNT:
        raise ValueError("action comparison arrays must have [step, 7] shape")
    selected_js = js[:, SAFE_TOKEN_INDEX]
    return {
        "window_steps": int(js.shape[0]),
        "same_feature_action_js_at_fault": float(selected_js[0]),
        "same_feature_action_js_sum": float(selected_js.sum()),
        "same_feature_action_logit_l2_at_fault": float(
            l2[0, SAFE_TOKEN_INDEX]
        ),
        "same_feature_clean_choice_margin_erosion_at_fault": float(
            erosion[0, SAFE_TOKEN_INDEX]
        ),
        "same_feature_action_argmax_changed_at_fault": bool(
            argmax_changed[0, SAFE_TOKEN_INDEX]
        ),
        "full_action_mean_js_at_fault": float(js[0].mean()),
        "full_action_js_sum": float(js.sum()),
        "generated_action_token_change_fraction_at_fault": float(
            tokens_changed[0].mean()
        ),
        "executed_command_l2_at_fault": float(command_l2[0]),
        "executed_command_l2_energy": float(
            math.sqrt(float((command_l2 * command_l2).sum()))
        ),
    }


def bfloat16_words_to_float32(np: Any, words: Any) -> Any:
    values = np.asarray(words, dtype=np.uint16)
    return (values.astype(np.uint32) << 16).view(np.float32)

