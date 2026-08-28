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
class PolicyDecision:
    raw_action: Any
    command: Any
    action_tokens: tuple[int, ...]
    hidden_states: Any
    trace: LanguageInferenceTrace | None
    inference_seconds: float


def _hidden_states(runtime: Runtime, generated: Any) -> Any:
    # SAFE OpenVLA 300dce26 returns one hidden-state tuple per generated action
    # token. SAFE's public loader consumes the final layer at the final sequence
    # position, matching openvla_rollout.py::_extract_hidden_states.
    values = [token_states[-1][0, -1, :] for token_states in generated["hidden_states"]]
    result = runtime.torch.stack(values, dim=0).detach().cpu()
    if result.ndim != 2 or result.shape[0] != 7:
        raise ValueError(f"unexpected OpenVLA hidden-state shape: {tuple(result.shape)}")
    return result


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
        hidden_states=_hidden_states(runtime, generated),
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
