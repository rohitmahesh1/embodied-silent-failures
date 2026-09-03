from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from embodied_silent_failures.language_fault import (
    LanguageBlockInjector,
    LanguageInferenceTrace,
    tensor_change,
)
from embodied_silent_failures.openvla_runtime import Runtime


@dataclass(frozen=True)
class GenerationLogitTrace:
    sequence_token_ids: Any
    action_token_logits: Any
    top_token_ids: Any
    top_token_logits: Any
    log_normalizer: Any
    entropy: Any
    vocabulary_size: int
    action_token_start: int


@dataclass(frozen=True)
class PolicyDecision:
    raw_action: Any
    command: Any
    action_tokens: tuple[int, ...]
    hidden_states: Any
    generation_logits: GenerationLogitTrace
    trace: LanguageInferenceTrace | None
    inference_seconds: float


def extract_hidden_states(runtime: Runtime, generated: Any) -> Any:
    # SAFE OpenVLA 300dce26 returns one hidden-state tuple per generated action
    # token. SAFE's public loader consumes the final layer at the final sequence
    # position, matching openvla_rollout.py::_extract_hidden_states.
    values = [token_states[-1][0, -1, :] for token_states in generated["hidden_states"]]
    result = runtime.torch.stack(values, dim=0).detach().cpu()
    if result.ndim != 2 or result.shape[0] != 7:
        raise ValueError(f"unexpected OpenVLA hidden-state shape: {tuple(result.shape)}")
    return result


def generation_logit_trace(runtime: Runtime, generated: Any) -> GenerationLogitTrace:
    # SAFE OpenVLA 300dce26, modeling_prismatic.py::predict_action, decodes
    # actions from the final 256 vocabulary entries. Keep that complete action
    # vocabulary plus the global top tokens and exact normalization summaries;
    # this avoids archiving the unrelated full language vocabulary per fault.
    values = [value[0].detach().to(runtime.torch.float32) for value in generated["logits"]]
    logits = runtime.torch.stack(values, dim=0)
    if logits.ndim != 2 or logits.shape[0] != 7 or logits.shape[1] < 256:
        raise ValueError(f"unexpected OpenVLA logit shape: {tuple(logits.shape)}")
    top_count = min(32, int(logits.shape[1]))
    top_logits, top_ids = runtime.torch.topk(logits, k=top_count, dim=-1)
    log_normalizer = runtime.torch.logsumexp(logits, dim=-1)
    probabilities = runtime.torch.softmax(logits, dim=-1)
    log_probabilities = runtime.torch.log_softmax(logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    vocabulary_size = int(logits.shape[1])
    return GenerationLogitTrace(
        sequence_token_ids=generated["sequences"][0].detach().cpu(),
        action_token_logits=logits[:, -256:].detach().cpu(),
        top_token_ids=top_ids.detach().cpu(),
        top_token_logits=top_logits.detach().cpu(),
        log_normalizer=log_normalizer.detach().cpu(),
        entropy=entropy.detach().cpu(),
        vocabulary_size=vocabulary_size,
        action_token_start=vocabulary_size - 256,
    )


def policy_decision(
    runtime: Runtime,
    policy_config: Any,
    model: Any,
    processor: Any,
    observation: dict[str, Any],
    task_description: str,
    *,
    injector: LanguageBlockInjector | None = None,
    action_token_position: int | None = None,
    replacement_layer: int | None = None,
    sources: dict[int, Any] | None = None,
    cache_replacement_layers: frozenset[int] | None = None,
    cache_sources: dict[int, dict[str, Any]] | None = None,
    capture_internal_state: bool = False,
    capture_context_state: bool = False,
) -> PolicyDecision:
    resize_size = runtime.get_image_resize_size(policy_config)
    image = runtime.get_libero_image(observation, resize_size)
    state = runtime.np.concatenate(
        (
            observation["robot0_eef_pos"],
            runtime.quat2axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    policy_observation = {"full_image": image, "state": state}
    if injector is not None and action_token_position is None:
        raise ValueError("language-block tracing requires an action-token position")
    trace_context = (
        injector.inference(
            int(action_token_position),
            replacement_layer=replacement_layer,
            sources=sources,
            cache_replacement_layers=cache_replacement_layers,
            cache_sources=cache_sources,
            capture_internal_state=capture_internal_state,
            capture_context_state=capture_context_state,
        )
        if injector is not None
        else nullcontext()
    )

    runtime.torch.cuda.synchronize()
    started = time.perf_counter()
    with runtime.torch.inference_mode(), trace_context:
        raw_actions, generated = runtime.get_action(
            policy_config,
            model,
            policy_observation,
            task_description,
            processor=processor,
            n_samples=1,
        )
    runtime.torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    raw_action = runtime.np.asarray(raw_actions).copy()
    if raw_action.shape != (7,):
        raise ValueError(f"unexpected OpenVLA action shape: {raw_action.shape}")
    command = runtime.normalize_gripper_action(raw_action.copy(), binarize=True)
    command = runtime.invert_gripper_action(command)
    action_tokens = generated["sequences"][:, -7:].detach().cpu().reshape(-1)
    return PolicyDecision(
        raw_action=raw_action,
        command=runtime.np.asarray(command).copy(),
        action_tokens=tuple(int(value) for value in action_tokens.tolist()),
        hidden_states=extract_hidden_states(runtime, generated),
        generation_logits=generation_logit_trace(runtime, generated),
        trace=injector.last_trace if injector is not None else None,
        inference_seconds=inference_seconds,
    )


def array_change(np: Any, reference: Any, value: Any) -> dict[str, Any]:
    reference_array = np.asarray(reference)
    value_array = np.asarray(value)
    if reference_array.shape != value_array.shape:
        raise ValueError("cannot compare policy outputs with different shapes")
    difference = value_array.astype(float) - reference_array.astype(float)
    reference_l2 = float(np.linalg.norm(reference_array.astype(float)))
    difference_l2 = float(np.linalg.norm(difference))
    return {
        "difference_l2": difference_l2,
        "normalized_difference_l2": difference_l2
        / max(reference_l2, np.finfo(float).eps),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "changed_element_count": int(np.count_nonzero(difference)),
        "exact_equal": bool(np.array_equal(reference_array, value_array)),
        "finite": bool(np.isfinite(difference).all()),
    }


def _cache_precondition(
    runtime: Runtime,
    clean: LanguageInferenceTrace,
    faulted: LanguageInferenceTrace,
    *,
    layer_index: int,
    token_position: int,
) -> dict[str, Any]:
    coordinates = [
        (layer, token)
        for token in range(token_position)
        for layer in range(32)
    ]
    coordinates.extend(
        (layer, token_position) for layer in range(layer_index + 1)
    )
    result = {}
    for kind in ("key", "value"):
        first_difference = None
        exact_coordinates = 0
        for layer, token in coordinates:
            exact = runtime.torch.equal(
                clean.cache_values_by_call[layer][kind][token],
                faulted.cache_values_by_call[layer][kind][token],
            )
            if exact:
                exact_coordinates += 1
            elif first_difference is None:
                first_difference = {
                    "layer_index": layer,
                    "action_token_position": token,
                }
        result[kind] = {
            "compared_coordinates": len(coordinates),
            "exact_coordinates": exact_coordinates,
            "all_coordinates_exact": exact_coordinates == len(coordinates),
            "first_difference": first_difference,
        }
    return {
        "scope": (
            "all earlier action-token cache writes and the selected call through "
            "the faulted block; these entries precede the output replacement"
        ),
        **result,
    }


def intervention_record(
    runtime: Runtime,
    *,
    site: dict[str, Any],
    source: LanguageInferenceTrace,
    clean: PolicyDecision,
    faulted: PolicyDecision,
) -> dict[str, Any]:
    layer_index = int(site["layer_index"])
    if clean.trace is None or faulted.trace is None:
        raise ValueError("local intervention is missing a language-block trace")
    propagation = []
    for downstream_layer in range(32):
        propagation.append(
            {
                "layer_index": downstream_layer,
                **tensor_change(
                    runtime.torch,
                    clean.trace.block_values[downstream_layer],
                    faulted.trace.block_values[downstream_layer],
                ),
            }
        )
    return {
        "status": "complete",
        "site_id": site["site_id"],
        "layer_index": layer_index,
        "action_token_position": int(site["action_token_position"]),
        "identity": site["identity"],
        "injection": tensor_change(
            runtime.torch,
            clean.trace.block_values[layer_index],
            source.block_values[layer_index],
        ),
        "cache_precondition": _cache_precondition(
            runtime,
            clean.trace,
            faulted.trace,
            layer_index=layer_index,
            token_position=int(site["action_token_position"]),
        ),
        "propagation": propagation,
        "safe_feature": tensor_change(
            runtime.torch, clean.hidden_states, faulted.hidden_states
        ),
        "raw_action": array_change(runtime.np, clean.raw_action, faulted.raw_action),
        "executed_command": array_change(
            runtime.np, clean.command, faulted.command
        ),
        "clean_action_tokens": list(clean.action_tokens),
        "faulted_action_tokens": list(faulted.action_tokens),
        "action_tokens_exact_equal": clean.action_tokens == faulted.action_tokens,
        "clean_raw_action": clean.raw_action.tolist(),
        "faulted_raw_action": faulted.raw_action.tolist(),
        "clean_executed_command": clean.command.tolist(),
        "faulted_executed_command": faulted.command.tolist(),
        "clean_hook_calls": clean.trace.call_counts,
        "faulted_hook_calls": faulted.trace.call_counts,
        "clean_hook_anomalies": list(clean.trace.anomalies),
        "faulted_hook_anomalies": list(faulted.trace.anomalies),
        "faulted_inference_seconds": faulted.inference_seconds,
    }
