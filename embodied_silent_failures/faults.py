import hashlib
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator


FAULT_SITES = ("decoder_layer", "final_hidden", "action_logits")


@dataclass(frozen=True)
class FaultSpec:
    site: str
    layer: int | None
    policy_step: int
    generation_step: int
    bit_index: int | None
    seed: int
    feature_index: int | None = None

    def __post_init__(self) -> None:
        if self.site not in FAULT_SITES:
            raise ValueError(f"unsupported fault site: {self.site}")
        if self.site == "decoder_layer" and self.layer is None:
            raise ValueError("decoder_layer faults require a layer")
        if self.site != "decoder_layer" and self.layer is not None:
            raise ValueError(f"{self.site} faults do not take a layer")
        if self.layer is not None and self.layer < 0:
            raise ValueError("fault layer must be non-negative")
        if self.policy_step < 0:
            raise ValueError("fault policy step must be non-negative")
        if not 0 <= self.generation_step < 7:
            raise ValueError("fault generation step must be between 0 and 6")
        if self.bit_index is not None and self.bit_index < 0:
            raise ValueError("fault bit index must be non-negative")
        if self.feature_index is not None and self.feature_index < 0:
            raise ValueError("fault feature index must be non-negative")
        if self.seed < 0:
            raise ValueError("fault seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        evidence_relation = {
            "decoder_layer": "upstream_of_monitor_tap",
            "final_hidden": "exact_monitor_input_when_targeting_monitored_token",
            "action_logits": "post_tap_with_autoregressive_feedback",
        }[self.site]
        return {
            "kind": "single_transient_activation_bit_flip",
            "evidence_relation": evidence_relation,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FaultSpec":
        fields = (
            "site",
            "layer",
            "policy_step",
            "generation_step",
            "bit_index",
            "seed",
            "feature_index",
        )
        missing = [field for field in fields[:6] if field not in value]
        if missing:
            raise ValueError(f"fault specification is missing {', '.join(missing)}")
        return cls(**{field: value.get(field) for field in fields})


def _event_seed(spec: FaultSpec, trial_seed: int) -> int:
    value = (
        f"{spec.seed}:{trial_seed}:{spec.site}:{spec.layer}:"
        f"{spec.policy_step}:{spec.generation_step}"
    )
    digest = hashlib.sha256(value.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _floating_layout(torch: Any, dtype: Any) -> tuple[Any, int, int, int]:
    layouts = {
        torch.bfloat16: (torch.int16, 16, 7, 8),
        torch.float16: (torch.int16, 16, 10, 5),
        torch.float32: (torch.int32, 32, 23, 8),
    }
    try:
        return layouts[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported activation dtype: {dtype}") from error


def _bit_class(bit_index: int, mantissa_bits: int, exponent_bits: int) -> str:
    if bit_index < mantissa_bits:
        return "mantissa"
    if bit_index < mantissa_bits + exponent_bits:
        return "exponent"
    return "sign"


def _indices_for_flat_index(shape: tuple[int, ...], flat_index: int) -> list[int]:
    indices = []
    remaining = flat_index
    for size in reversed(shape):
        indices.append(remaining % size)
        remaining //= size
    return list(reversed(indices))


def _active_token_flat_index(
    shape: tuple[int, ...], feature_index: int
) -> int:
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(
            "fault injection expects a batch-one [batch, sequence, feature] tensor"
        )
    sequence_length, feature_count = shape[1], shape[2]
    if sequence_length <= 0 or feature_count <= 0:
        raise ValueError("fault injection received an empty sequence or feature axis")
    if not 0 <= feature_index < feature_count:
        raise IndexError(f"feature index {feature_index} is outside [0, {feature_count})")
    return (sequence_length - 1) * feature_count + feature_index


def _replace_first_tensor(torch: Any, output: Any, replacement: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (replacement, *output[1:])
    raise TypeError(f"fault hook received unsupported module output: {type(output)}")


class TransientActivationFault:
    """Apply one deterministic native-format bit flip during one policy inference."""

    def __init__(self, torch: Any, spec: FaultSpec) -> None:
        self._torch = torch
        self.spec = spec
        self._handle: Any = None
        self._trial_seed: int | None = None
        self._policy_step: int | None = None
        self._generation_step = 0
        self._record: dict[str, Any] | None = None

    @property
    def record(self) -> dict[str, Any] | None:
        return self._record

    def install(self, model: Any) -> None:
        if self._handle is not None:
            raise RuntimeError("fault hook is already installed")

        if self.spec.site == "decoder_layer":
            layers = model.language_model.model.layers
            if self.spec.layer is None or self.spec.layer >= len(layers):
                raise IndexError(
                    f"OpenVLA has {len(layers)} decoder layers, requested {self.spec.layer}"
                )
            module = layers[self.spec.layer]
        elif self.spec.site == "final_hidden":
            module = model.language_model.model.norm
        else:
            module = model.language_model.lm_head

        self._handle = module.register_forward_hook(self._hook)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def begin_trial(self, trial_seed: int) -> None:
        if trial_seed < 0:
            raise ValueError("trial seed must be non-negative")
        self._trial_seed = trial_seed
        self._policy_step = None
        self._generation_step = 0
        self._record = None

    @contextmanager
    def inference(self, policy_step: int) -> Iterator[None]:
        if self._trial_seed is None:
            raise RuntimeError("begin_trial must be called before inference")
        if self._policy_step is not None:
            raise RuntimeError("fault inference contexts cannot be nested")

        self._policy_step = policy_step
        self._generation_step = 0
        completed = False
        try:
            yield
            completed = True
        finally:
            if (
                completed
                and policy_step == self.spec.policy_step
                and self._record is None
            ):
                raise RuntimeError(
                    "the requested fault was not injected; the generation step or hook "
                    "does not match the model execution"
                )
            self._policy_step = None

    def require_injected(self) -> dict[str, Any]:
        if self._record is None:
            raise RuntimeError(
                f"rollout ended before fault policy step {self.spec.policy_step}"
            )
        return self._record

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        generation_step = self._generation_step
        self._generation_step += 1

        if self._policy_step != self.spec.policy_step:
            return output
        if generation_step != self.spec.generation_step:
            return output
        if self._record is not None:
            raise RuntimeError("fault was injected more than once in one trial")
        if self._trial_seed is None:
            raise RuntimeError("fault hook ran outside a trial")

        tensor = output if isinstance(output, self._torch.Tensor) else output[0]
        faulted, record = self._flip(tensor, self._trial_seed)
        self._record = record
        return _replace_first_tensor(self._torch, output, faulted)

    def _flip(self, tensor: Any, trial_seed: int) -> tuple[Any, dict[str, Any]]:
        integer_dtype, width, mantissa_bits, exponent_bits = _floating_layout(
            self._torch, tensor.dtype
        )
        if tensor.numel() == 0:
            raise ValueError("cannot inject a fault into an empty tensor")
        if self.spec.bit_index is not None and self.spec.bit_index >= width:
            raise ValueError(
                f"bit index {self.spec.bit_index} is invalid for {tensor.dtype}"
            )

        rng = random.Random(_event_seed(self.spec, trial_seed))
        shape = tuple(tensor.shape)
        feature_index = (
            self.spec.feature_index
            if self.spec.feature_index is not None
            else rng.randrange(shape[-1])
        )
        if not 0 <= feature_index < shape[-1]:
            raise IndexError(
                f"feature index {feature_index} is outside [0, {shape[-1]})"
            )
        flat_index = _active_token_flat_index(shape, feature_index)
        bit_index = (
            self.spec.bit_index
            if self.spec.bit_index is not None
            else rng.randrange(width)
        )

        faulted = tensor.clone()
        integer_view = faulted.view(integer_dtype).reshape(-1)
        before_signed = int(integer_view[flat_index].item())
        mask_unsigned = 1 << bit_index
        mask_signed = (
            mask_unsigned
            if bit_index < width - 1
            else mask_unsigned - (1 << width)
        )
        mask = self._torch.tensor(
            mask_signed, dtype=integer_dtype, device=tensor.device
        )
        integer_view[flat_index] = self._torch.bitwise_xor(
            integer_view[flat_index], mask
        )
        after_signed = int(integer_view[flat_index].item())

        value_view = faulted.reshape(-1)
        before_value = float(tensor.reshape(-1)[flat_index].item())
        after_value = float(value_view[flat_index].item())
        unsigned_mask = (1 << width) - 1
        record = {
            **self.spec.to_dict(),
            "trial_seed": trial_seed,
            "tensor_scope": "final_sequence_token",
            "tensor_shape": list(shape),
            "tensor_dtype": str(tensor.dtype),
            "sequence_index": shape[1] - 1,
            "feature_index": feature_index,
            "feature_selection": (
                "exact" if self.spec.feature_index is not None else "seeded_uniform"
            ),
            "flat_index": flat_index,
            "indices": _indices_for_flat_index(shape, flat_index),
            "actual_bit_index": bit_index,
            "bit_class": _bit_class(bit_index, mantissa_bits, exponent_bits),
            "before_bits": f"0x{before_signed & unsigned_mask:0{width // 4}x}",
            "after_bits": f"0x{after_signed & unsigned_mask:0{width // 4}x}",
            "before_value": before_value,
            "after_value": after_value,
            "before_finite": bool(self._torch.isfinite(tensor.reshape(-1)[flat_index])),
            "after_finite": bool(self._torch.isfinite(value_view[flat_index])),
        }
        return faulted, record
