from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping


_PORT_TOKEN = re.compile(r"\.([^\.\[]+)|\[(\d+)\]")
_ACTION_SEQUENCE_BOUNDARIES = {
    "openvla.action_tokens",
    "openvla.policy_call",
}
_INFERENCE_BOUNDARIES = _ACTION_SEQUENCE_BOUNDARIES | {
    "openvla.processor_output",
}


@dataclass(frozen=True)
class TemporalReplacementSpec:
    site_id: str
    identity: dict[str, Any]
    policy_step: int
    source_policy_step: int
    mode: str = "prior_value"

    def __post_init__(self) -> None:
        if not self.site_id:
            raise ValueError("temporal replacement site ID must be nonempty")
        if self.identity.get("kind") not in {
            "module_output",
            "declared_runtime_boundary",
        }:
            raise ValueError("temporal replacement has an unsupported site identity")
        if self.policy_step < 0:
            raise ValueError("temporal replacement policy step must be non-negative")
        if self.mode == "prior_value":
            if self.source_policy_step != self.policy_step - 1:
                raise ValueError("prior-value replacement requires source step t-1")
        elif self.mode == "current_value_canary":
            if self.source_policy_step != self.policy_step:
                raise ValueError("current-value canary must source its target step")
        else:
            raise ValueError(f"unsupported temporal replacement mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _port_tokens(port: str) -> list[str | int]:
    if not port.startswith("value"):
        raise ValueError(f"output port does not start at value: {port}")
    position = len("value")
    tokens: list[str | int] = []
    while position < len(port):
        match = _PORT_TOKEN.match(port, position)
        if match is None:
            raise ValueError(f"unsupported output port syntax: {port}")
        name, index = match.groups()
        tokens.append(name if name is not None else int(index))
        position = match.end()
    return tokens


def value_at_port(value: Any, port: str) -> Any:
    current = value
    for token in _port_tokens(port):
        current = current[token]
    return current


def replace_at_port(value: Any, port: str, replacement: Any) -> Any:
    """Return the same container shape with one recorder-compatible leaf replaced."""

    def replace(current: Any, tokens: list[str | int]) -> Any:
        if not tokens:
            return replacement
        token, remaining = tokens[0], tokens[1:]
        child = current[token]
        changed = replace(child, remaining)
        if isinstance(current, tuple):
            values = list(current)
            values[int(token)] = changed
            if hasattr(current, "_fields"):
                return type(current)(*values)
            return tuple(values)
        if isinstance(current, list):
            values = list(current)
            values[int(token)] = changed
            return values
        if isinstance(current, Mapping):
            values = copy.copy(current)
            values[token] = changed
            return values
        raise TypeError(
            f"cannot replace child {token!r} in output container {type(current)}"
        )

    return replace(value, _port_tokens(port))


def _plain_schema(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        **(
            {"shape": [int(size) for size in shape]}
            if shape is not None
            else {}
        ),
        **(
            {"dtype": str(value.dtype)}
            if getattr(value, "dtype", None) is not None
            else {}
        ),
        **(
            {"device": str(value.device)}
            if getattr(value, "device", None) is not None
            else {}
        ),
    }


def _tensor_sha256(torch: Any, value: Any) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _comparison(torch: Any, prior: Any, current: Any) -> dict[str, Any]:
    if tuple(prior.shape) != tuple(current.shape) or prior.dtype != current.dtype:
        raise ValueError("prior and current temporal values have different schemas")
    if prior.device != current.device:
        raise ValueError("prior and current temporal values are on different devices")
    prior_float = prior.detach().to(torch.float32)
    current_float = current.detach().to(torch.float32)
    difference = current_float - prior_float
    prior_norm = float(torch.linalg.vector_norm(prior_float).item())
    current_norm = float(torch.linalg.vector_norm(current_float).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    denominator = max(current_norm, torch.finfo(torch.float32).eps)
    exact_equal = bool(torch.equal(prior, current))
    finite_prior = bool(torch.isfinite(prior_float).all())
    finite_current = bool(torch.isfinite(current_float).all())
    finite_difference = bool(torch.isfinite(difference).all())
    cosine = None
    if prior_norm > 0 and current_norm > 0 and finite_prior and finite_current:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                prior_float.reshape(1, -1), current_float.reshape(1, -1)
            ).item()
        )
    return {
        "schema": _plain_schema(current),
        "element_count": int(current.numel()),
        "prior_sha256": _tensor_sha256(torch, prior),
        "current_sha256": _tensor_sha256(torch, current),
        "exact_equal": exact_equal,
        "prior_finite": finite_prior,
        "current_finite": finite_current,
        "difference_finite": finite_difference,
        "prior_l2": prior_norm,
        "current_l2": current_norm,
        "difference_l2": difference_norm,
        "normalized_difference_l2": difference_norm / denominator,
        "maximum_absolute_difference": (
            float(torch.max(torch.abs(difference)).item())
            if difference.numel()
            else 0.0
        ),
        "cosine_similarity": cosine,
        "changed_element_count": int(torch.count_nonzero(prior != current).item()),
    }


def _numpy_comparison(np: Any, prior: Any, current: Any) -> dict[str, Any]:
    prior_array = np.asarray(prior)
    current_array = np.asarray(current)
    if prior_array.shape != current_array.shape or prior_array.dtype != current_array.dtype:
        raise ValueError("prior and current temporal arrays have different schemas")
    prior_float = prior_array.astype(np.float64)
    current_float = current_array.astype(np.float64)
    difference = current_float - prior_float
    prior_norm = float(np.linalg.norm(prior_float))
    current_norm = float(np.linalg.norm(current_float))
    difference_norm = float(np.linalg.norm(difference))
    denominator = max(current_norm, np.finfo(np.float64).eps)
    cosine = None
    if prior_norm > 0 and current_norm > 0:
        cosine = float(
            np.dot(prior_float.reshape(-1), current_float.reshape(-1))
            / (prior_norm * current_norm)
        )
    digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return {
        "schema": _plain_schema(current_array),
        "element_count": int(current_array.size),
        "prior_sha256": digest(prior_array),
        "current_sha256": digest(current_array),
        "exact_equal": bool(np.array_equal(prior_array, current_array)),
        "prior_finite": bool(np.isfinite(prior_float).all()),
        "current_finite": bool(np.isfinite(current_float).all()),
        "difference_finite": bool(np.isfinite(difference).all()),
        "prior_l2": prior_norm,
        "current_l2": current_norm,
        "difference_l2": difference_norm,
        "normalized_difference_l2": difference_norm / denominator,
        "maximum_absolute_difference": (
            float(np.max(np.abs(difference))) if difference.size else 0.0
        ),
        "cosine_similarity": cosine,
        "changed_element_count": int(np.count_nonzero(prior_array != current_array)),
    }


def _clone(torch: Any, np: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"temporal replacement requires numeric data, got {array.dtype}")
    return array.copy()


class TemporalReplacementInjector:
    """Replace one runtime value with its preceding policy-step value."""

    def __init__(self, torch: Any, np: Any, spec: TemporalReplacementSpec) -> None:
        self._torch = torch
        self._np = np
        self.spec = spec
        self._handle: Any = None
        self._trial_seed: int | None = None
        self._policy_step: int | None = None
        self._module_call_index = 0
        self._boundary_calls: dict[tuple[int, str], int] = {}
        self._source: Any = None
        self._record: dict[str, Any] | None = None
        self._observer: Any = None

    @property
    def record(self) -> dict[str, Any] | None:
        return self._record

    @property
    def is_temporal_replacement(self) -> bool:
        return True

    def set_observer(self, observer: Any) -> None:
        self._observer = observer

    def install(self, model: Any) -> None:
        if self._handle is not None:
            raise RuntimeError("temporal replacement hook is already installed")
        identity = self.spec.identity
        if identity["kind"] != "module_output":
            return
        module_path = str(identity["module_path"])
        if not module_path.startswith("policy"):
            raise ValueError(f"module site is outside the policy root: {module_path}")
        relative = module_path.removeprefix("policy").removeprefix(".")
        modules = dict(model.named_modules())
        if relative not in modules:
            raise KeyError(f"policy has no traced module path {module_path}")
        self._handle = modules[relative].register_forward_hook(self._module_hook)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._source = None

    def begin_trial(self, trial_seed: int) -> None:
        if trial_seed < 0:
            raise ValueError("trial seed must be non-negative")
        self._trial_seed = trial_seed
        self._policy_step = None
        self._module_call_index = 0
        self._boundary_calls = {}
        self._source = None
        self._record = None

    @contextmanager
    def inference(self, policy_step: int) -> Iterator[None]:
        if self._trial_seed is None:
            raise RuntimeError("begin_trial must be called before inference")
        if self._policy_step is not None:
            raise RuntimeError("temporal replacement contexts cannot be nested")
        self._policy_step = policy_step
        self._module_call_index = 0
        self._boundary_calls = {}
        completed = False
        try:
            yield
            completed = True
        finally:
            identity = self.spec.identity
            expected_inside_context = bool(
                identity["kind"] == "module_output"
                or identity.get("event_name") in _INFERENCE_BOUNDARIES
            )
            if (
                completed
                and expected_inside_context
                and policy_step == self.spec.policy_step
                and self._record is None
            ):
                raise RuntimeError("the requested temporal replacement site was not reached")
            self._policy_step = None

    def boundary(
        self, event_name: str, output: Any, *, policy_step: int | None = None
    ) -> Any:
        identity = self.spec.identity
        if identity["kind"] != "declared_runtime_boundary":
            return output
        active_step = self._policy_step if policy_step is None else policy_step
        if active_step is None:
            raise RuntimeError("temporal boundary ran outside a policy step")
        call_key = (active_step, event_name)
        call_index = self._boundary_calls.get(call_key, 0)
        self._boundary_calls[call_key] = call_index + 1
        if event_name != identity["event_name"]:
            return output
        if call_index != int(identity["event_call_index"]):
            return output
        return self._observe_or_replace(output, policy_step=policy_step)

    def requires_action_redecode(self) -> bool:
        identity = self.spec.identity
        return bool(
            self._policy_step == self.spec.policy_step
            and identity["kind"] == "declared_runtime_boundary"
            and identity["event_name"] in _ACTION_SEQUENCE_BOUNDARIES
        )

    def require_injected(self) -> dict[str, Any]:
        if self._record is None:
            raise RuntimeError(
                f"rollout ended before temporal replacement step {self.spec.policy_step}"
            )
        return self._record

    def _module_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        call_index = self._module_call_index
        self._module_call_index += 1
        if call_index != int(self.spec.identity["module_call_index"]):
            return output
        return self._observe_or_replace(output)

    def _observe_or_replace(
        self, output: Any, *, policy_step: int | None = None
    ) -> Any:
        active_step = self._policy_step if policy_step is None else policy_step
        if active_step not in {
            self.spec.source_policy_step,
            self.spec.policy_step,
        }:
            return output
        port = str(self.spec.identity["output_port"])
        current = value_at_port(output, port)
        if active_step == self.spec.source_policy_step:
            self._source = _clone(self._torch, self._np, current)
            if self.spec.mode == "prior_value":
                return output
        if active_step != self.spec.policy_step:
            return output
        source = (
            _clone(self._torch, self._np, current)
            if self.spec.mode == "current_value_canary"
            else self._source
        )
        if source is None:
            raise RuntimeError("temporal replacement has no captured source value")
        comparison = (
            _comparison(self._torch, source, current)
            if isinstance(current, self._torch.Tensor)
            else _numpy_comparison(self._np, source, current)
        )
        replacement = _clone(self._torch, self._np, source)
        self._record = {
            "kind": "single_temporal_value_replacement",
            "operator": "replace x_t with x_(t-1)",
            **self.spec.to_dict(),
            "trial_seed": self._trial_seed,
            "comparison": comparison,
        }
        if self._observer is not None:
            self._observer(current, replacement, self._record)
        return replace_at_port(output, port, replacement)


class TemporalProcessor:
    """Apply the declared processor-output adapter after the pinned `.to` call."""

    def __init__(self, processor: Any, injector: TemporalReplacementInjector) -> None:
        self._processor = processor
        self._injector = injector

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _TemporalProcessorOutput(self._processor(*args, **kwargs), self._injector)


class _TemporalProcessorOutput:
    def __init__(self, value: Any, injector: TemporalReplacementInjector) -> None:
        self._value = value
        self._injector = injector

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        value = self._value.to(*args, **kwargs)
        return self._injector.boundary("openvla.processor_output", value)


def decode_action_tokens(model: Any, action_tokens: Any, unnorm_key: str, np: Any) -> Any:
    """Reapply the pinned OpenVLA token-to-action conversion after an adapter."""
    # OpenVLA 300dce26, modeling_prismatic.OpenVLAForActionPrediction.predict_action:
    # map the final action token IDs to bin centers, clip out-of-range IDs, then
    # unnormalize with the checkpoint's q01/q99 statistics and optional mask.
    token_ids = action_tokens.detach().cpu().numpy()
    discretized = model.vocab_size - token_ids
    discretized = np.clip(
        discretized - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )
    normalized = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    mask = mask[None].repeat(normalized.shape[0], axis=0)
    actions = np.where(
        mask,
        0.5 * (normalized + 1) * (high - low) + low,
        normalized,
    )
    return actions[0] if actions.shape[0] == 1 else actions


def finite_number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None
