from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from embodied_silent_failures.language_policy import (
    PolicyDecision,
    generation_logit_trace,
)
from embodied_silent_failures.openvla_runtime import Runtime
from embodied_silent_failures.temporal_fault import TemporalProcessor, decode_action_tokens


def atlas_policy_decision(
    runtime: Runtime,
    policy_config: Any,
    model: Any,
    processor: Any,
    observation: dict[str, Any],
    task_description: str,
    *,
    policy_step: int,
    adapter: Any = None,
) -> PolicyDecision:
    """Run one OpenVLA decision through an optional graph-site adapter."""
    source_observation = observation
    if adapter is not None:
        source_observation = adapter.boundary(
            "libero.current_observation", observation, policy_step=policy_step
        )
    resize_size = runtime.get_image_resize_size(policy_config)
    image = runtime.get_libero_image(source_observation, resize_size)
    if adapter is not None:
        image = adapter.boundary(
            "libero.current_image", image, policy_step=policy_step
        )
        image = adapter.boundary(
            "policy.selected_image", image, policy_step=policy_step
        )
    state = runtime.np.concatenate(
        (
            source_observation["robot0_eef_pos"],
            runtime.quat2axisangle(source_observation["robot0_eef_quat"]),
            source_observation["robot0_gripper_qpos"],
        )
    )
    policy_observation = {"full_image": image, "state": state}
    selected_processor = (
        TemporalProcessor(processor, adapter) if adapter is not None else processor
    )
    inference = adapter.inference(policy_step) if adapter is not None else nullcontext()

    runtime.torch.cuda.synchronize()
    started = time.perf_counter()
    with runtime.torch.inference_mode(), inference:
        raw_actions, generated = runtime.get_action(
            policy_config,
            model,
            policy_observation,
            task_description,
            processor=selected_processor,
            n_samples=1,
        )
        # evidence_graph/rollout.py::policy_outputs records this compact view at
        # the OpenVLA policy-call boundary. Reconstruct the same view from SAFE
        # OpenVLA 300dce26 so table ports map to live values without renaming them.
        policy_call = {
            "sequences": generated["sequences"],
            "final_layer_states": [
                token_states[-1] for token_states in generated["hidden_states"]
            ],
        }
        if adapter is not None:
            policy_call = adapter.boundary("openvla.policy_call", policy_call)
        action_tokens = policy_call["sequences"][:, -model.get_action_dim(
            policy_config.unnorm_key
        ) :]
        if adapter is not None:
            action_tokens = adapter.boundary("openvla.action_tokens", action_tokens)
            if getattr(adapter, "requires_action_redecode", lambda: False)():
                raw_actions = decode_action_tokens(
                    model, action_tokens, policy_config.unnorm_key, runtime.np
                )
    runtime.torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    raw_action = runtime.np.asarray(raw_actions).copy()
    if adapter is not None:
        raw_action = adapter.boundary(
            "openvla.raw_action", raw_action, policy_step=policy_step
        )
    if raw_action.shape != (7,):
        raise ValueError(f"unexpected OpenVLA action shape: {raw_action.shape}")
    command = runtime.normalize_gripper_action(raw_action.copy(), binarize=True)
    command = runtime.invert_gripper_action(command)
    if adapter is not None:
        command = adapter.boundary(
            "libero.executed_command", command, policy_step=policy_step
        )

    # SAFE b6036abe, data.openvla.load_rollouts and process_tensor_idx_rel,
    # stacks one final-layer vector per action token and selects the seventh for
    # SAFE-MLP. Keep both declared boundaries independently injectable.
    hidden_states = runtime.torch.stack(
        [value[0, -1, :] for value in policy_call["final_layer_states"]], dim=0
    ).detach().cpu()
    if adapter is not None:
        hidden_states = adapter.boundary(
            "safe.final_layer_action_features",
            hidden_states,
            policy_step=policy_step,
        )
        monitor_input = adapter.boundary(
            "safe.monitor_input", hidden_states[-1], policy_step=policy_step
        )
        if tuple(monitor_input.shape) != tuple(hidden_states[-1].shape):
            raise ValueError("SAFE monitor input adapter changed the feature schema")
        hidden_states = hidden_states.clone()
        hidden_states[-1] = monitor_input
    sequence_tokens = action_tokens.detach().cpu().reshape(-1)
    logits = generation_logit_trace(runtime, model, generated)
    logits = replace(
        logits,
        sequence_token_ids=policy_call["sequences"][0].detach().cpu(),
    )
    return PolicyDecision(
        raw_action=runtime.np.asarray(raw_action).copy(),
        command=runtime.np.asarray(command).copy(),
        action_tokens=tuple(int(value) for value in sequence_tokens.tolist()),
        hidden_states=hidden_states.detach().cpu(),
        generation_logits=logits,
        trace=None,
        inference_seconds=inference_seconds,
    )
