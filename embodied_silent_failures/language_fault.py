from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class LanguageInferenceTrace:
    action_token_position: int
    block_values: dict[int, Any]
    block_values_by_call: dict[int, dict[int, Any]]
    attention_values_by_call: dict[int, dict[str, dict[int, Any]]]
    sequence_lengths_by_call: dict[int, dict[int, int]]
    call_counts: dict[int, int]
    anomalies: tuple[str, ...]


class LanguageBlockInjector:
    """Observe all block residuals and optionally replace one final token vector."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._handles: list[Any] = []
        self._active = False
        self._token_position = 0
        self._replacement_layer: int | None = None
        self._sources: Mapping[int, Any] = {}
        self._call_counts: dict[int, int] = {}
        self._block_values: dict[int, Any] = {}
        self._block_values_by_call: dict[int, dict[int, Any]] = {}
        self._attention_values_by_call: dict[int, dict[str, dict[int, Any]]] = {}
        self._sequence_lengths_by_call: dict[int, dict[int, int]] = {}
        self._last_trace: LanguageInferenceTrace | None = None

    @property
    def last_trace(self) -> LanguageInferenceTrace | None:
        return self._last_trace

    def install(self, model: Any) -> None:
        if self._handles:
            raise RuntimeError("language-block hooks are already installed")
        modules = dict(model.named_modules())
        for layer_index in range(32):
            # The exact path is the one observed for all 32 residual blocks in
            # the pinned temporal-site table, not a semantic layer alias.
            path = f"language_model.model.layers.{layer_index}"
            if path not in modules:
                raise KeyError(f"policy has no traced language block {path}")
            for suffix in ("k_proj", "v_proj"):
                projection_path = f"{path}.self_attn.{suffix}"
                if projection_path not in modules:
                    raise KeyError(
                        f"policy has no traced attention projection {projection_path}"
                    )
        for layer_index in range(32):
            path = f"language_model.model.layers.{layer_index}"
            self._handles.append(
                modules[path].register_forward_hook(self._hook(layer_index))
            )
            for kind, suffix in (("key", "k_proj"), ("value", "v_proj")):
                projection_path = f"{path}.self_attn.{suffix}"
                self._handles.append(
                    modules[projection_path].register_forward_hook(
                        self._attention_hook(layer_index, kind)
                    )
                )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        self._sources = {}

    @contextmanager
    def inference(
        self,
        action_token_position: int,
        *,
        replacement_layer: int | None = None,
        sources: Mapping[int, Any] | None = None,
    ) -> Iterator[None]:
        if self._active:
            raise RuntimeError("language-block inference contexts cannot be nested")
        if not 0 <= action_token_position < 7:
            raise ValueError("action-token position must be between zero and six")
        if replacement_layer is not None and not 0 <= replacement_layer < 32:
            raise ValueError("replacement layer must be between zero and 31")
        if replacement_layer is not None and replacement_layer not in (sources or {}):
            raise ValueError("replacement layer has no prior-decision source vector")
        self._active = True
        self._token_position = action_token_position
        self._replacement_layer = replacement_layer
        self._sources = sources or {}
        self._call_counts = {index: 0 for index in range(32)}
        self._block_values = {}
        self._block_values_by_call = {index: {} for index in range(32)}
        self._attention_values_by_call = {
            index: {"key": {}, "value": {}} for index in range(32)
        }
        self._sequence_lengths_by_call = {index: {} for index in range(32)}
        self._last_trace = None
        try:
            yield
        finally:
            anomalies = []
            for layer_index in range(32):
                calls = self._call_counts[layer_index]
                if calls <= action_token_position:
                    anomalies.append(
                        f"layer {layer_index} observed {calls} calls before token "
                        f"position {action_token_position}"
                    )
                if layer_index not in self._block_values:
                    anomalies.append(f"layer {layer_index} has no captured token vector")
                for kind in ("key", "value"):
                    projection_calls = len(
                        self._attention_values_by_call[layer_index][kind]
                    )
                    if projection_calls != calls:
                        anomalies.append(
                            f"layer {layer_index} observed {projection_calls} {kind} "
                            f"projection calls for {calls} block calls"
                        )
            self._last_trace = LanguageInferenceTrace(
                action_token_position=action_token_position,
                block_values=self._block_values,
                block_values_by_call=self._block_values_by_call,
                attention_values_by_call=self._attention_values_by_call,
                sequence_lengths_by_call=self._sequence_lengths_by_call,
                call_counts=self._call_counts,
                anomalies=tuple(anomalies),
            )
            self._active = False
            self._sources = {}

    def _hook(self, layer_index: int):
        def observe(_module: Any, _inputs: Any, output: Any) -> Any:
            if not self._active:
                return output
            call_index = self._call_counts[layer_index]
            self._call_counts[layer_index] = call_index + 1
            if not isinstance(output, tuple) or not output:
                raise TypeError("OpenVLA language block did not return a tuple")
            hidden = output[0]
            if not isinstance(hidden, self._torch.Tensor) or hidden.ndim != 3:
                raise TypeError("OpenVLA language block has an unexpected output")

            # vla-safe/openvla 300dce26, modeling_prismatic.py::predict_action,
            # delegates seven-token decoding to Transformers `generate`. The
            # recorded block port is `value[0]`; selecting its final sequence
            # position matches the next-token representation on call zero and
            # the sole representation on calls one through six.
            final_token = hidden[:, -1:, :]
            self._sequence_lengths_by_call[layer_index][call_index] = int(
                hidden.shape[1]
            )
            if (
                call_index == self._token_position
                and layer_index == self._replacement_layer
            ):
                source = self._sources[layer_index]
                if (
                    tuple(source.shape) != tuple(final_token.shape)
                    or source.dtype != final_token.dtype
                    or source.device != final_token.device
                ):
                    raise ValueError("prior and current action-token vectors differ in schema")
                changed_hidden = hidden.clone()
                changed_hidden[:, -1:, :] = source
                values = list(output)
                values[0] = changed_hidden
                output = tuple(values)
                final_token = changed_hidden[:, -1:, :]
            captured = final_token.detach().clone()
            self._block_values_by_call[layer_index][call_index] = captured
            if call_index == self._token_position:
                self._block_values[layer_index] = captured
            return output

        return observe

    def _attention_hook(self, layer_index: int, kind: str):
        def observe(_module: Any, _inputs: Any, output: Any) -> Any:
            if not self._active:
                return output
            if not isinstance(output, self._torch.Tensor) or output.ndim != 3:
                raise TypeError("OpenVLA attention projection has an unexpected output")
            call_index = len(self._attention_values_by_call[layer_index][kind])
            # OpenVLA 300dce26 instantiates the language backbone from the
            # pinned Transformers Llama blocks. These named k_proj/v_proj
            # outputs are the mechanically observed inputs to the attention
            # cache-write path; keys are still rotated downstream, so this does
            # not claim to be a dump of the complete cache.
            self._attention_values_by_call[layer_index][kind][call_index] = (
                output[:, -1:, :].detach().clone()
            )
            return output

        return observe


def tensor_change(torch: Any, reference: Any, value: Any) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(value.shape):
        raise ValueError("cannot compare action-token vectors with different shapes")
    reference_float = reference.detach().to(torch.float32)
    value_float = value.detach().to(torch.float32)
    difference = value_float - reference_float
    reference_l2 = float(torch.linalg.vector_norm(reference_float).item())
    difference_l2 = float(torch.linalg.vector_norm(difference).item())
    return {
        "difference_l2": difference_l2,
        "normalized_difference_l2": difference_l2
        / max(reference_l2, torch.finfo(torch.float32).eps),
        "maximum_absolute_difference": float(torch.max(torch.abs(difference)).item()),
        "changed_element_count": int(torch.count_nonzero(difference).item()),
        "exact_equal": bool(torch.equal(reference, value)),
        "finite": bool(torch.isfinite(difference).all()),
    }
