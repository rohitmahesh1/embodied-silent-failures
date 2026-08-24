from __future__ import annotations

import contextlib
import hashlib
import io
import json
import time
import types
from typing import Any

from embodied_silent_failures.pi0_fast_contract import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    ACTION_TOKEN_START,
    ACTION_TOKEN_STOP,
    FEATURE_DIMENSION,
    FEATURE_SOURCE_DTYPE,
    FEATURE_TRANSPORT_ENCODING,
    MAX_DECODING_STEPS,
    PALIGEMMA_EOS_TOKEN,
    PARITY_ACTION_ATOL,
    PROTOCOL_VERSION,
    SAFE_OPENPI_PARENT_REVISION,
)


REQUEST_KEY = "evidence_request"
FAST_FALLBACK_PREFIX = "Error decoding tokens:"
INSTRUMENTED_STATIC_ARGUMENTS = (
    "max_decoding_steps",
    "temperature",
    "n_action_samples",
)


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    records = []
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        records.append(
            {
                "type": f"{type(current).__module__}.{type(current).__qualname__}",
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    return records


def decode_with_diagnostics(
    decode: Any, state: Any, tokens: Any
) -> tuple[Any | None, dict[str, Any]]:
    """Run one decode while preserving the pinned FAST fallback signal."""
    import numpy as np

    output = io.StringIO()
    try:
        # physical-intelligence/fast revision ec4d7aa, in
        # processing_action_tokenizer.py::UniversalActionProcessor.decode, reports
        # malformed DCT coefficients with print() before substituting normalized
        # zero actions. Capturing that exact output makes the fallback observable
        # without changing the pinned dependency or depending on server stdout.
        with contextlib.redirect_stdout(output):
            actions = np.asarray(decode(state, tokens))
    except Exception as error:
        return None, {
            "status": "raised",
            "captured_stdout": output.getvalue(),
            "exception_chain": _exception_chain(error),
        }

    captured = output.getvalue()
    fallback = any(
        line.startswith(FAST_FALLBACK_PREFIX) for line in captured.splitlines()
    )
    return actions, {
        "status": "tokenizer_fallback" if fallback else "decoded",
        "captured_stdout": captured,
        "action_shape": list(actions.shape),
        "actions_sha256": _array_sha256(actions),
        "policy_visible_actions_all_zero": bool(np.all(actions == 0)),
    }


def token_record(tokens: Any, decode_step: int) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(tokens, dtype=np.int32)
    decoded = values[:decode_step]
    return {
        "decode_step": decode_step,
        "decoded_token_ids": decoded.tolist(),
        "decoded_tokens_sha256": _array_sha256(decoded),
        "full_buffer_sha256": _array_sha256(values),
    }


def decode_audit_classification(
    decoded_tokens_exact: bool,
    instrumented_status: str,
    reference_status: str,
) -> str:
    if not decoded_tokens_exact:
        return "sampler_tokens_differ"
    if instrumented_status == reference_status == "tokenizer_fallback":
        return "same_tokens_same_tokenizer_fallback"
    if instrumented_status == "tokenizer_fallback" and reference_status == "decoded":
        return "instrumented_only_tokenizer_fallback"
    if instrumented_status == "raised" or reference_status == "raised":
        return "decode_raised"
    return "same_tokens_different_decode_status"


def evidence_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {
        **(metadata or {}),
        "evidence_protocol_version": PROTOCOL_VERSION,
        "evidence_names": ["encoded", "pre_logits", "action_token_logits"],
    }


def reference_sample_action_tokens(
    self: Any,
    rng: Any,
    observation: Any,
    *,
    max_decoding_steps: int = MAX_DECODING_STEPS,
    temperature: float = 0.0,
) -> Any:
    """Run the uninstrumented parent implementation for the parity canary."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_module
    from openpi.models.pi0_fast import (
        PALIGEMMA_EOS_TOKEN,
        left_to_right_align,
        make_attn_mask,
        put_along_last_axis,
    )

    # Physical Intelligence openpi commit 29068dd,
    # src/openpi/models/pi0_fast.py::Pi0FAST.sample_actions, is the direct parent
    # of SAFE's instrumentation commit 109b414. This is that parent's greedy
    # decoding path, retained independently so the first policy decision can
    # prove that feature collection leaves generated tokens and actions intact.
    observation = model_module.preprocess_observation(
        None,
        observation,
        train=False,
        image_keys=list(observation.images.keys()),
    )
    embeddings, input_mask, ar_mask = self.embed_inputs(observation)
    attention_mask = make_attn_mask(input_mask, ar_mask)
    embeddings, input_mask, attention_mask = left_to_right_align(
        embeddings, input_mask, attention_mask
    )
    prefill_size = embeddings.shape[1]
    prefill_len = jnp.sum(input_mask, axis=-1)
    prefix_start = prefill_size - prefill_len
    attention_mask = jnp.pad(
        attention_mask, ((0, 0), (0, 0), (0, max_decoding_steps))
    )
    positions = jnp.cumsum(input_mask, axis=-1) - 1
    logits, cache, _ = self.PaliGemma.llm(
        embedded_prefix=embeddings,
        mask=attention_mask,
        positions=positions,
        decode=True,
    )
    last_logit = logits[:, -1:]
    tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

    def step(carry: tuple[Any, Any, Any, Any, Any]):
        current_logit, current_tokens, current_cache, _, index = carry
        token = (
            jax.random.categorical(rng, current_logit / temperature, axis=-1)
            if temperature > 0
            else jnp.argmax(current_logit, axis=-1)
        )
        current_tokens = put_along_last_axis(
            current_tokens,
            jnp.broadcast_to(index, (token.shape[0], 1)),
            token,
        )
        all_eos = jnp.all(jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1))
        token_embedding = self.PaliGemma.llm(token, embed_only=True)
        token_positions = prefill_len[:, None] + index + 1
        token_mask = jnp.logical_and(
            jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
            >= prefix_start[:, None, None],
            jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
            < jnp.broadcast_to(
                prefill_size + index + 1, (prefix_start.shape[0], 1, 1)
            ),
        )
        current_logit, current_cache, _ = self.PaliGemma.llm(
            embedded_prefix=token_embedding,
            mask=token_mask,
            positions=token_positions,
            decode=True,
            kv_cache=current_cache,
        )
        return (
            current_logit,
            current_tokens,
            current_cache,
            all_eos,
            index + 1,
        )

    def cond(carry: tuple[Any, Any, Any, Any, Any]):
        return (~carry[3]) & (carry[4] < max_decoding_steps)

    _, tokens, _, _, _ = jax.lax.while_loop(
        cond, step, (last_logit, tokens, cache, False, 0)
    )
    return tokens


def parity_record(
    tokens: Any,
    decode_step: int,
    actions: Any,
    reference_tokens: Any,
    reference_actions: Any,
) -> dict[str, Any]:
    """Compare only policy-visible output, leaving unused padding diagnostic."""
    import numpy as np

    tokens = np.asarray(tokens)
    reference_tokens = np.asarray(reference_tokens)
    actions = np.asarray(actions)
    reference_actions = np.asarray(reference_actions)
    eos = np.flatnonzero(reference_tokens == PALIGEMMA_EOS_TOKEN)
    reference_decode_step = int(eos[0] + 1) if eos.size else len(reference_tokens)
    decoded_length_exact = decode_step == reference_decode_step
    decoded_tokens_exact = decoded_length_exact and bool(
        np.array_equal(tokens[:decode_step], reference_tokens[:reference_decode_step])
    )
    maximum_action_error = float(np.max(np.abs(reference_actions - actions)))
    actions_close = bool(
        np.allclose(
            reference_actions,
            actions,
            rtol=0.0,
            atol=PARITY_ACTION_ATOL,
            equal_nan=False,
        )
    )
    return {
        "decoded_length_exact": decoded_length_exact,
        "decoded_tokens_exact": decoded_tokens_exact,
        "unused_padding_exact": bool(np.array_equal(tokens, reference_tokens)),
        "instrumented_decoded_tokens": decode_step,
        "reference_decoded_tokens": reference_decode_step,
        "maximum_absolute_action_error": maximum_action_error,
        "action_atol": PARITY_ACTION_ATOL,
        "passed": decoded_tokens_exact and actions_close,
    }


def create_evidence_policy(
    model: Any,
    input_transforms: list[Any],
    output_transforms: list[Any],
    metadata: dict[str, Any] | None,
) -> Any:
    """Serve SAFE's pinned pi0-FAST features through an audited response."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi import transforms as transform_module
    from openpi.models import model as model_module
    from openpi.shared import nnx_utils
    from openpi_client import base_policy

    if model.action_horizon != ACTION_HORIZON or model.action_dim != ACTION_DIMENSION:
        raise ValueError(
            "pi0-FAST dimensions disagree with the frozen LIBERO contract: "
            f"horizon={model.action_horizon}, action_dim={model.action_dim}"
        )
    instrumented = nnx_utils.module_jit(
        model.sample_actions,
        static_argnames=INSTRUMENTED_STATIC_ARGUMENTS,
    )
    reference_method = types.MethodType(reference_sample_action_tokens, model)
    reference = nnx_utils.module_jit(
        reference_method,
        static_argnames=("max_decoding_steps", "temperature"),
    )

    class EvidencePolicy(base_policy.BasePolicy):
        def __init__(self) -> None:
            # SAFE openpi commit 9c99ed5,
            # src/openpi/policies/policy_config.py::create_trained_policy, composes
            # these exact model input and output transforms around the checkpoint.
            self._input_transform = transform_module.compose(input_transforms)
            self._output_transform = transform_module.compose(output_transforms)
            self._metadata = evidence_metadata(metadata)

        def _decode(self, state: Any, tokens: Any) -> Any:
            return self._output_transform(
                {
                    "state": np.asarray(jax.device_get(state)),
                    "actions": np.asarray(jax.device_get(tokens)),
                }
            )["actions"]

        def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
            request_obs = dict(obs)
            request = request_obs.pop(REQUEST_KEY, None)
            if not isinstance(request, dict):
                raise ValueError(f"request must include an object at {REQUEST_KEY!r}")
            decision_id = request.get("decision_id")
            compare_reference = request.get("compare_reference", False)
            audit_malformed_decode = request.get("audit_malformed_decode", False)
            record_decode_fallback = request.get("record_decode_fallback", False)
            if type(decision_id) is not int or decision_id < 0:
                raise ValueError("decision_id must be a non-negative integer")
            if type(compare_reference) is not bool:
                raise ValueError("compare_reference must be a boolean")
            if type(audit_malformed_decode) is not bool:
                raise ValueError("audit_malformed_decode must be a boolean")
            if type(record_decode_fallback) is not bool:
                raise ValueError("record_decode_fallback must be a boolean")
            if audit_malformed_decode and record_decode_fallback:
                raise ValueError(
                    "decode attribution and fallback continuation are separate modes"
                )

            inputs = jax.tree.map(lambda value: value, request_obs)
            inputs = self._input_transform(inputs)
            inputs = jax.tree.map(
                lambda value: jnp.asarray(value)[np.newaxis, ...], inputs
            )
            observation = model_module.Observation.from_dict(inputs)
            key = jax.random.key(0)

            started = time.monotonic()
            tokens, auxiliary = instrumented(
                key,
                observation,
                max_decoding_steps=MAX_DECODING_STEPS,
                temperature=0.0,
                n_action_samples=1,
            )
            tokens.block_until_ready()
            inference_ms = (time.monotonic() - started) * 1000

            decode_step = int(np.asarray(jax.device_get(auxiliary["decode_step"])))
            if not 1 <= decode_step <= MAX_DECODING_STEPS:
                raise ValueError(f"pi0-FAST decoded {decode_step} tokens")
            host_tokens = np.asarray(jax.device_get(tokens[0]), dtype=np.int32)
            if audit_malformed_decode or record_decode_fallback:
                actions, instrumented_decode = decode_with_diagnostics(
                    self._decode, inputs["state"][0], host_tokens
                )
            else:
                actions = np.asarray(self._decode(inputs["state"][0], host_tokens))
                instrumented_decode = None

            if (
                audit_malformed_decode
                and instrumented_decode is not None
                and instrumented_decode["status"] != "decoded"
            ):
                reference_tokens = reference(
                    key,
                    observation,
                    max_decoding_steps=MAX_DECODING_STEPS,
                    temperature=0.0,
                )
                reference_tokens.block_until_ready()
                host_reference = np.asarray(
                    jax.device_get(reference_tokens[0]), dtype=np.int32
                )
                eos = np.flatnonzero(host_reference == PALIGEMMA_EOS_TOKEN)
                reference_decode_step = (
                    int(eos[0] + 1) if eos.size else len(host_reference)
                )
                _reference_actions, reference_decode = decode_with_diagnostics(
                    self._decode, inputs["state"][0], host_reference
                )
                decoded_tokens_exact = bool(
                    decode_step == reference_decode_step
                    and np.array_equal(
                        host_tokens[:decode_step],
                        host_reference[:reference_decode_step],
                    )
                )
                return {
                    "decode_audit": {
                        "schema_version": 1,
                        "protocol_version": PROTOCOL_VERSION,
                        "decision_id": decision_id,
                        "instrumented": {
                            "tokens": token_record(host_tokens, decode_step),
                            "decode": instrumented_decode,
                        },
                        "reference": {
                            "source_revision": SAFE_OPENPI_PARENT_REVISION,
                            "tokens": token_record(
                                host_reference, reference_decode_step
                            ),
                            "decode": reference_decode,
                        },
                        "comparison": {
                            "decoded_length_exact": (
                                decode_step == reference_decode_step
                            ),
                            "decoded_tokens_exact": decoded_tokens_exact,
                            "full_buffer_exact": bool(
                                np.array_equal(host_tokens, host_reference)
                            ),
                        },
                        "classification": decode_audit_classification(
                            decoded_tokens_exact,
                            instrumented_decode["status"],
                            reference_decode["status"],
                        ),
                    },
                    "policy_timing": {"infer_ms": inference_ms},
                }
            if actions is None:
                raise RuntimeError(
                    "pi0-FAST decode raised instead of returning its documented "
                    f"fallback: {instrumented_decode}"
                )
            source_values = {
                "encoded_bfloat16_bits": auxiliary[
                    "encoded"
                ][0, :decode_step],
                "pre_logits_bfloat16_bits": auxiliary[
                    "pre_logits"
                ][0, :decode_step],
                "action_token_logits_bfloat16_bits": auxiliary[
                    "logits"
                ][0, :decode_step, ACTION_TOKEN_START:ACTION_TOKEN_STOP],
            }
            source_dtypes = {
                name: str(value.dtype) for name, value in source_values.items()
            }
            if set(source_dtypes.values()) != {FEATURE_SOURCE_DTYPE}:
                raise ValueError(
                    "pi0-FAST feature dtypes disagree with its pinned bfloat16 "
                    f"model configuration: {source_dtypes}"
                )
            feature_bits = {
                name: np.asarray(
                    jax.device_get(
                        jax.lax.bitcast_convert_type(value, jnp.uint16)
                    ),
                    dtype=np.uint16,
                )
                for name, value in source_values.items()
            }
            expected = (decode_step, FEATURE_DIMENSION)
            for name, value in feature_bits.items():
                if value.shape != expected:
                    raise ValueError(
                        f"pi0-FAST {name} has shape {value.shape}, expected {expected}"
                    )

            result = {
                "actions": actions,
                "raw_action_tokens": host_tokens[:decode_step],
                "evidence": {
                    **feature_bits,
                    "action_token_start": ACTION_TOKEN_START,
                    "action_token_stop": ACTION_TOKEN_STOP,
                    "decoded_tokens": decode_step,
                    "decision_id": decision_id,
                    "protocol_version": PROTOCOL_VERSION,
                    "source_dtypes": source_dtypes,
                    "transport_encoding": FEATURE_TRANSPORT_ENCODING,
                },
                "policy_timing": {"infer_ms": inference_ms},
            }
            if record_decode_fallback:
                result["decode_diagnostic"] = {
                    "schema_version": 1,
                    "protocol_version": PROTOCOL_VERSION,
                    "decision_id": decision_id,
                    **instrumented_decode,
                }
            if compare_reference:
                reference_tokens = reference(
                    key,
                    observation,
                    max_decoding_steps=MAX_DECODING_STEPS,
                    temperature=0.0,
                )
                reference_tokens.block_until_ready()
                host_reference = np.asarray(
                    jax.device_get(reference_tokens[0]), dtype=np.int32
                )
                reference_actions, reference_decode = decode_with_diagnostics(
                    self._decode, inputs["state"][0], host_reference
                )
                if reference_actions is None:
                    raise RuntimeError(
                        "pi0-FAST parent decode raised during parity comparison: "
                        f"{reference_decode}"
                    )
                result["reference_comparison"] = parity_record(
                    host_tokens,
                    decode_step,
                    actions,
                    host_reference,
                    reference_actions,
                )
            return result

        def reset(self) -> None:
            return None

        @property
        def metadata(self) -> dict[str, Any]:
            return self._metadata

    return EvidencePolicy()
