from __future__ import annotations

import time
import types
from typing import Any

from embodied_silent_failures.pi05_contract import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    DIFFUSION_STEPS,
    PROTOCOL_VERSION,
)


PARITY_ATOL = 1e-6
REQUEST_KEY = "evidence_request"


def sample_actions_with_safe_evidence(
    self: Any,
    rng: Any,
    observation: Any,
    *,
    num_steps: int = DIFFUSION_STEPS,
    noise: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run pi0.5 sampling while retaining SAFE's published feature."""
    import einops
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask

    del rng
    observation = model_module.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    expected_noise_shape = (
        batch_size,
        self.action_horizon,
        self.action_dim,
    )
    if noise.shape != expected_noise_shape:
        raise ValueError(
            f"sampling noise has shape {noise.shape}, expected {expected_noise_shape}"
        )

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    # SAFE openpi commit 9c99ed5, src/openpi/models/pi0.py::Pi0.sample_actions,
    # defines pre_velocity as the action-expert suffix features immediately before
    # action_out_proj, retained for every diffusion step and action-horizon position.
    # This adaptation keeps that tap while preserving pi0.5's adaRMS condition from
    # Physical Intelligence openpi commit 15a9616 in the same function and file.
    feature_width = self.action_out_proj.in_features
    pre_velocity = jnp.zeros(
        (batch_size, num_steps, self.action_horizon, feature_width),
        dtype=jnp.float32,
    )

    def step(carry: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
        x_t, diffusion_time, features, index = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            jnp.broadcast_to(diffusion_time, batch_size),
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        if prefix_out is not None:
            raise ValueError(
                "pi0.5 suffix inference unexpectedly returned prefix output"
            )
        current = suffix_out[:, -self.action_horizon :]
        features = features.at[:, index].set(current)
        velocity = self.action_out_proj(current)
        return x_t + dt * velocity, diffusion_time + dt, features, index + 1

    def cond(carry: tuple[Any, Any, Any, Any]) -> Any:
        _, diffusion_time, _, _ = carry
        return diffusion_time >= -dt / 2

    actions, _, pre_velocity, completed_steps = jax.lax.while_loop(
        cond,
        step,
        (noise, 1.0, pre_velocity, 0),
    )
    return actions, {
        "pre_velocity": pre_velocity,
        "completed_diffusion_steps": completed_steps,
    }


def create_evidence_policy(trained_policy: Any) -> Any:
    """Wrap a pinned OpenPI policy without modifying its source checkout."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi.models import model as model_module
    from openpi.shared import nnx_utils
    from openpi_client import base_policy

    if trained_policy._is_pytorch_model:  # noqa: SLF001 - pinned OpenPI contract
        raise TypeError("pi0.5 SAFE evidence collection requires the JAX checkpoint")
    model = trained_policy._model  # noqa: SLF001 - pinned OpenPI contract
    if not getattr(model, "pi05", False):
        raise ValueError("loaded policy is not a pi0.5 model")
    if model.action_horizon != ACTION_HORIZON or model.action_dim != ACTION_DIMENSION:
        raise ValueError(
            "pi0.5 model dimensions disagree with the frozen LIBERO contract: "
            f"horizon={model.action_horizon}, action_dim={model.action_dim}"
        )

    instrumented_method = types.MethodType(sample_actions_with_safe_evidence, model)
    instrumented_sampler = nnx_utils.module_jit(
        instrumented_method, static_argnames=("num_steps",)
    )
    reference_sampler = nnx_utils.module_jit(
        model.sample_actions, static_argnames=("num_steps",)
    )

    class EvidencePolicy(base_policy.BasePolicy):
        def __init__(self) -> None:
            # Physical Intelligence openpi commit 15a9616,
            # src/openpi/policies/policy.py::Policy.infer, applies this exact transform,
            # batching, sampling, batch removal, and output-transform order. These private
            # members are therefore part of the explicitly pinned adapter boundary.
            self._input_transform = trained_policy._input_transform
            self._output_transform = trained_policy._output_transform
            self._sample_kwargs = dict(trained_policy._sample_kwargs)
            self._metadata = {
                **trained_policy.metadata,
                "evidence_protocol_version": PROTOCOL_VERSION,
                "evidence_name": "safe_pre_velocity",
            }

        def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
            request_obs = dict(obs)
            request = request_obs.pop(REQUEST_KEY, None)
            if not isinstance(request, dict):
                raise ValueError(f"request must include an object at {REQUEST_KEY!r}")
            decision_id = request.get("decision_id")
            noise_seed = request.get("noise_seed")
            compare_reference = request.get("compare_reference", False)
            if type(decision_id) is not int or decision_id < 0:
                raise ValueError("decision_id must be a non-negative integer")
            if type(noise_seed) is not int or not 0 <= noise_seed < 2**32:
                raise ValueError("noise_seed must be a 32-bit unsigned integer")
            if type(compare_reference) is not bool:
                raise ValueError("compare_reference must be a boolean")

            inputs = jax.tree.map(lambda value: value, request_obs)
            inputs = self._input_transform(inputs)
            inputs = jax.tree.map(
                lambda value: jnp.asarray(value)[np.newaxis, ...], inputs
            )
            observation = model_module.Observation.from_dict(inputs)
            noise_key = jax.random.key(noise_seed)
            noise = jax.random.normal(
                noise_key,
                (1, model.action_horizon, model.action_dim),
            )
            configured_sample_kwargs = dict(self._sample_kwargs)
            configured_steps = configured_sample_kwargs.pop(
                "num_steps", DIFFUSION_STEPS
            )
            if configured_steps != DIFFUSION_STEPS or configured_sample_kwargs:
                raise ValueError(
                    "pi0.5 policy sampling arguments disagree with the frozen "
                    "evidence contract: "
                    f"num_steps={configured_steps}, extra={configured_sample_kwargs}"
                )
            sample_kwargs = {"num_steps": DIFFUSION_STEPS, "noise": noise}

            started = time.monotonic()
            actions, evidence = instrumented_sampler(
                noise_key, observation, **sample_kwargs
            )
            actions.block_until_ready()
            inference_ms = (time.monotonic() - started) * 1000

            raw_actions = np.asarray(jax.device_get(actions[0]))
            host_outputs = {
                "state": np.asarray(jax.device_get(inputs["state"][0])),
                "actions": raw_actions,
            }
            outputs = self._output_transform(host_outputs)
            outputs["raw_actions"] = raw_actions
            outputs["evidence"] = {
                "pre_velocity": np.asarray(jax.device_get(evidence["pre_velocity"][0])),
                "sampling_noise": np.asarray(jax.device_get(noise[0])),
                "completed_diffusion_steps": int(
                    np.asarray(jax.device_get(evidence["completed_diffusion_steps"]))
                ),
                "noise_seed": noise_seed,
                "decision_id": decision_id,
                "protocol_version": PROTOCOL_VERSION,
            }
            outputs["policy_timing"] = {"infer_ms": inference_ms}

            if compare_reference:
                reference = reference_sampler(noise_key, observation, **sample_kwargs)
                reference.block_until_ready()
                maximum_error = float(
                    np.asarray(jax.device_get(jnp.max(jnp.abs(reference - actions))))
                )
                outputs["reference_comparison"] = {
                    "maximum_absolute_raw_action_error": maximum_error,
                    "atol": PARITY_ATOL,
                    "passed": maximum_error <= PARITY_ATOL,
                }
            return outputs

        def reset(self) -> None:
            return None

        @property
        def metadata(self) -> dict[str, Any]:
            return self._metadata

    return EvidencePolicy()
