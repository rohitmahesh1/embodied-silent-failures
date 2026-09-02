from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class LanguageInferenceTrace:
    action_token_position: int
    block_values: dict[int, Any]
    block_values_by_call: dict[int, dict[int, Any]]
    cache_values_by_call: dict[int, dict[str, dict[int, Any]]]
    sequence_lengths_by_call: dict[int, dict[int, int]]
    call_counts: dict[int, int]
    anomalies: tuple[str, ...]
    block_inputs_by_call: dict[int, dict[int, Any]] = field(default_factory=dict)
    post_attention_residuals_by_call: dict[int, dict[int, Any]] = field(
        default_factory=dict
    )
    attention_queries_by_call: dict[int, dict[int, Any]] = field(
        default_factory=dict
    )
    model_input_ids_by_call: dict[int, Any] = field(default_factory=dict)
    model_attention_masks_by_call: dict[int, Any] = field(default_factory=dict)
    decoder_position_ids_by_call: dict[int, Any] = field(default_factory=dict)
    decoder_cache_positions_by_call: dict[int, Any] = field(default_factory=dict)
    decoder_attention_masks_by_call: dict[int, Any] = field(default_factory=dict)
    decoder_attention_mask_present_by_call: dict[int, bool] = field(
        default_factory=dict
    )
    prompt_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    prompt_cache_format: str | None = None
    initial_pixel_values: Any | None = None
    initial_language_input: Any | None = None


class LanguageBlockInjector:
    """Observe all block residuals and optionally replace one final token vector."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._handles: list[Any] = []
        self._active = False
        self._token_position = 0
        self._replacement_layer: int | None = None
        self._sources: Mapping[int, Any] = {}
        self._cache_replacement_layers: frozenset[int] = frozenset()
        self._cache_sources: Mapping[int, Mapping[str, Any]] = {}
        self._call_counts: dict[int, int] = {}
        self._block_values: dict[int, Any] = {}
        self._block_values_by_call: dict[int, dict[int, Any]] = {}
        self._cache_values_by_call: dict[int, dict[str, dict[int, Any]]] = {}
        self._sequence_lengths_by_call: dict[int, dict[int, int]] = {}
        self._block_inputs_by_call: dict[int, dict[int, Any]] = {}
        self._post_attention_residuals_by_call: dict[int, dict[int, Any]] = {}
        self._attention_queries_by_call: dict[int, dict[int, Any]] = {}
        self._model_input_ids_by_call: dict[int, Any] = {}
        self._model_attention_masks_by_call: dict[int, Any] = {}
        self._decoder_position_ids_by_call: dict[int, Any] = {}
        self._decoder_cache_positions_by_call: dict[int, Any] = {}
        self._decoder_attention_masks_by_call: dict[int, Any] = {}
        self._decoder_attention_mask_present_by_call: dict[int, bool] = {}
        self._prompt_cache: dict[int, dict[str, Any]] = {}
        self._prompt_cache_format: str | None = None
        self._initial_pixel_values: Any | None = None
        self._initial_language_input: Any | None = None
        self._capture_internal_state = False
        self._capture_context_state = False
        self._model_call_count = 0
        self._pending_queries: dict[int, Any] = {}
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
            attention_path = f"{path}.self_attn"
            if attention_path not in modules:
                raise KeyError(
                    f"policy has no traced attention module {attention_path}"
                )
            for suffix in ("post_attention_layernorm", "self_attn.q_proj", "self_attn.rotary_emb"):
                internal_path = f"{path}.{suffix}"
                if internal_path not in modules:
                    raise KeyError(f"policy has no traced internal module {internal_path}")
        self._handles.append(
            model.register_forward_pre_hook(self._model_input_hook, with_kwargs=True)
        )
        for layer_index in range(32):
            path = f"language_model.model.layers.{layer_index}"
            attention = modules[f"{path}.self_attn"]
            self._handles.append(
                modules[path].register_forward_pre_hook(
                    self._block_input_hook(layer_index)
                )
            )
            self._handles.append(
                modules[path].register_forward_hook(self._hook(layer_index))
            )
            self._handles.append(
                modules[f"{path}.post_attention_layernorm"].register_forward_pre_hook(
                    self._post_attention_hook(layer_index)
                )
            )
            self._handles.append(
                modules[f"{path}.self_attn.q_proj"].register_forward_hook(
                    self._query_projection_hook(
                        layer_index,
                        num_heads=int(attention.num_heads),
                        head_dim=int(attention.head_dim),
                    )
                )
            )
            if layer_index == 0:
                self._handles.append(
                    modules[f"{path}.self_attn"].register_forward_pre_hook(
                        self._attention_input_hook, with_kwargs=True
                    )
                )
            self._handles.append(
                modules[f"{path}.self_attn.rotary_emb"].register_forward_hook(
                    self._rotary_hook(layer_index)
                )
            )
            self._handles.append(
                modules[f"{path}.self_attn"].register_forward_hook(
                    self._cache_hook(layer_index)
                )
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        self._sources = {}
        self._cache_replacement_layers = frozenset()
        self._cache_sources = {}

    @contextmanager
    def inference(
        self,
        action_token_position: int,
        *,
        replacement_layer: int | None = None,
        sources: Mapping[int, Any] | None = None,
        cache_replacement_layers: frozenset[int] | None = None,
        cache_sources: Mapping[int, Mapping[str, Any]] | None = None,
        capture_internal_state: bool = False,
        capture_context_state: bool = False,
    ) -> Iterator[None]:
        if self._active:
            raise RuntimeError("language-block inference contexts cannot be nested")
        if not 0 <= action_token_position < 7:
            raise ValueError("action-token position must be between zero and six")
        if replacement_layer is not None and not 0 <= replacement_layer < 32:
            raise ValueError("replacement layer must be between zero and 31")
        if replacement_layer is not None and replacement_layer not in (sources or {}):
            raise ValueError("replacement layer has no prior-decision source vector")
        cache_replacement_layers = cache_replacement_layers or frozenset()
        unknown_cache_layers = sorted(
            layer
            for layer in cache_replacement_layers
            if not 0 <= layer < 32
        )
        if unknown_cache_layers:
            raise ValueError(
                f"cache replacement layers are invalid: {unknown_cache_layers}"
            )
        missing_cache_sources = sorted(
            layer
            for layer in cache_replacement_layers
            if layer not in (cache_sources or {})
        )
        if missing_cache_sources:
            raise ValueError(
                "cache replacement layers have no source entries: "
                f"{missing_cache_sources}"
            )
        self._active = True
        self._token_position = action_token_position
        self._replacement_layer = replacement_layer
        self._sources = sources or {}
        self._cache_replacement_layers = cache_replacement_layers
        self._cache_sources = cache_sources or {}
        self._capture_internal_state = bool(capture_internal_state)
        self._capture_context_state = bool(capture_context_state)
        self._call_counts = {index: 0 for index in range(32)}
        self._block_values = {}
        self._block_values_by_call = {index: {} for index in range(32)}
        self._cache_values_by_call = {
            index: {"key": {}, "value": {}} for index in range(32)
        }
        self._sequence_lengths_by_call = {index: {} for index in range(32)}
        self._block_inputs_by_call = {index: {} for index in range(32)}
        self._post_attention_residuals_by_call = {
            index: {} for index in range(32)
        }
        self._attention_queries_by_call = {index: {} for index in range(32)}
        self._model_input_ids_by_call = {}
        self._model_attention_masks_by_call = {}
        self._decoder_position_ids_by_call = {}
        self._decoder_cache_positions_by_call = {}
        self._decoder_attention_masks_by_call = {}
        self._decoder_attention_mask_present_by_call = {}
        self._prompt_cache = {}
        self._prompt_cache_format = None
        self._initial_pixel_values = None
        self._initial_language_input = None
        self._model_call_count = 0
        self._pending_queries = {}
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
                    cache_calls = len(self._cache_values_by_call[layer_index][kind])
                    if cache_calls != calls:
                        anomalies.append(
                            f"layer {layer_index} observed {cache_calls} exact cache "
                            f"{kind} entries for {calls} block calls"
                        )
                if self._capture_internal_state:
                    for name, values in (
                        ("block input", self._block_inputs_by_call[layer_index]),
                        (
                            "post-attention residual",
                            self._post_attention_residuals_by_call[layer_index],
                        ),
                        (
                            "post-rotary query",
                            self._attention_queries_by_call[layer_index],
                        ),
                    ):
                        if len(values) != calls:
                            anomalies.append(
                                f"layer {layer_index} observed {len(values)} {name} "
                                f"entries for {calls} block calls"
                            )
            if self._model_call_count != 7:
                anomalies.append(
                    f"model observed {self._model_call_count} generation calls"
                )
            if self._capture_internal_state:
                if set(self._decoder_position_ids_by_call) != set(range(7)):
                    anomalies.append("decoder position IDs are incomplete")
                if set(self._decoder_cache_positions_by_call) != set(range(7)):
                    anomalies.append("decoder cache positions are incomplete")
                if set(self._decoder_attention_mask_present_by_call) != set(range(7)):
                    anomalies.append("decoder attention-mask records are incomplete")
            if self._capture_context_state:
                if len(self._prompt_cache) != 32:
                    anomalies.append(
                        f"prompt cache contains {len(self._prompt_cache)} of 32 layers"
                    )
                if self._initial_pixel_values is None:
                    anomalies.append("initial processed pixels were not captured")
                if self._initial_language_input is None:
                    anomalies.append("initial fused language input was not captured")
            self._last_trace = LanguageInferenceTrace(
                action_token_position=action_token_position,
                block_values=self._block_values,
                block_values_by_call=self._block_values_by_call,
                cache_values_by_call=self._cache_values_by_call,
                sequence_lengths_by_call=self._sequence_lengths_by_call,
                call_counts=self._call_counts,
                anomalies=tuple(anomalies),
                block_inputs_by_call=self._block_inputs_by_call,
                post_attention_residuals_by_call=(
                    self._post_attention_residuals_by_call
                ),
                attention_queries_by_call=self._attention_queries_by_call,
                model_input_ids_by_call=self._model_input_ids_by_call,
                model_attention_masks_by_call=self._model_attention_masks_by_call,
                decoder_position_ids_by_call=self._decoder_position_ids_by_call,
                decoder_cache_positions_by_call=(
                    self._decoder_cache_positions_by_call
                ),
                decoder_attention_masks_by_call=(
                    self._decoder_attention_masks_by_call
                ),
                decoder_attention_mask_present_by_call=(
                    self._decoder_attention_mask_present_by_call
                ),
                prompt_cache=self._prompt_cache,
                prompt_cache_format=self._prompt_cache_format,
                initial_pixel_values=self._initial_pixel_values,
                initial_language_input=self._initial_language_input,
            )
            self._active = False
            self._sources = {}
            self._cache_replacement_layers = frozenset()
            self._cache_sources = {}
            self._capture_internal_state = False
            self._capture_context_state = False
            self._pending_queries = {}

    def _attention_input_hook(
        self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if not self._active:
            return
        call_index = self._call_counts[0]
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 1:
            attention_mask = args[1]
        self._decoder_attention_mask_present_by_call[call_index] = isinstance(
            attention_mask, self._torch.Tensor
        )
        if isinstance(attention_mask, self._torch.Tensor):
            self._decoder_attention_masks_by_call[call_index] = (
                attention_mask.detach().clone()
            )
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) > 2:
            position_ids = args[2]
        if isinstance(position_ids, self._torch.Tensor):
            self._decoder_position_ids_by_call[call_index] = (
                position_ids.detach().clone()
            )
        cache_position = kwargs.get("cache_position")
        if cache_position is None and len(args) > 6:
            cache_position = args[6]
        if isinstance(cache_position, self._torch.Tensor):
            self._decoder_cache_positions_by_call[call_index] = (
                cache_position.detach().clone()
            )

    def _model_input_hook(
        self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if not self._active:
            return
        call_index = self._model_call_count
        self._model_call_count += 1

        # vla-safe/openvla 300dce26,
        # modeling_prismatic.py::prepare_inputs_for_generation passes these
        # exact tensors to each of the seven model calls. Call zero contains the
        # complete text/image input; later calls contain one generated token.
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if isinstance(input_ids, self._torch.Tensor):
            self._model_input_ids_by_call[call_index] = input_ids.detach().clone()
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, self._torch.Tensor):
            self._model_attention_masks_by_call[call_index] = (
                attention_mask.detach().clone()
            )

        if self._capture_context_state and call_index == 0:
            pixel_values = kwargs.get("pixel_values")
            if not isinstance(pixel_values, self._torch.Tensor):
                raise TypeError("pinned OpenVLA did not receive tensor pixel values")
            self._initial_pixel_values = pixel_values.detach().clone()

        if self._capture_context_state and call_index == 1:
            cache = kwargs.get("past_key_values")
            if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
                entries = list(zip(cache.key_cache, cache.value_cache, strict=True))
                cache_format = "transformers_dynamic_cache"
            elif isinstance(cache, (list, tuple)):
                entries = list(cache)
                cache_format = "transformers_legacy_tuple"
            else:
                raise TypeError("second OpenVLA call has no recognized prompt cache")
            if len(entries) != 32:
                raise ValueError("prompt cache does not contain all 32 language layers")
            captured = {}
            for layer, entry in enumerate(entries):
                if (
                    not isinstance(entry, (list, tuple))
                    or len(entry) != 2
                    or not all(
                        isinstance(value, self._torch.Tensor) for value in entry
                    )
                ):
                    raise TypeError(
                        f"prompt cache layer {layer} is not an exact key/value pair"
                    )
                captured[layer] = {
                    "key": entry[0].detach().clone(),
                    "value": entry[1].detach().clone(),
                }
            # OpenVLA 300dce26 with Transformers 4.40.1 passes a legacy tuple
            # between generation calls. LlamaModel.forward converts that tuple
            # to DynamicCache internally, then converts it back on return.
            self._prompt_cache = captured
            self._prompt_cache_format = cache_format

    def _block_input_hook(self, layer_index: int):
        def observe(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not self._active:
                return
            if not inputs or not isinstance(inputs[0], self._torch.Tensor):
                raise TypeError("OpenVLA language block has no tensor input")
            hidden = inputs[0]
            call_index = self._call_counts[layer_index]
            if self._capture_internal_state:
                self._block_inputs_by_call[layer_index][call_index] = (
                    hidden[:, -1:, :].detach().clone()
                )
            if (
                self._capture_context_state
                and layer_index == 0
                and call_index == 0
            ):
                self._initial_language_input = hidden.detach().clone()

        return observe

    def _post_attention_hook(self, layer_index: int):
        def observe(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not self._active or not self._capture_internal_state:
                return
            if not inputs or not isinstance(inputs[0], self._torch.Tensor):
                raise TypeError("OpenVLA post-attention norm has no tensor input")
            call_index = self._call_counts[layer_index]
            self._post_attention_residuals_by_call[layer_index][call_index] = (
                inputs[0][:, -1:, :].detach().clone()
            )

        return observe

    def _query_projection_hook(
        self, layer_index: int, *, num_heads: int, head_dim: int
    ):
        def observe(_module: Any, _inputs: Any, output: Any) -> None:
            if not self._active or not self._capture_internal_state:
                return
            if not isinstance(output, self._torch.Tensor) or output.ndim != 3:
                raise TypeError("OpenVLA query projection has an unexpected output")
            batch = int(output.shape[0])
            query = output[:, -1:, :].view(batch, 1, num_heads, head_dim)
            self._pending_queries[layer_index] = query.transpose(1, 2)

        return observe

    def _rotary_hook(self, layer_index: int):
        def observe(_module: Any, _inputs: Any, output: Any) -> None:
            if not self._active or not self._capture_internal_state:
                return
            if (
                not isinstance(output, tuple)
                or len(output) != 2
                or not all(isinstance(value, self._torch.Tensor) for value in output)
            ):
                raise TypeError("OpenVLA rotary embedding has an unexpected output")
            query = self._pending_queries.pop(layer_index, None)
            if query is None:
                raise RuntimeError("rotary embedding ran without a captured query")
            cos, sin = output
            if cos.ndim != 3 or sin.ndim != 3:
                raise TypeError("pinned rotary cosines and sines must be rank three")
            cos = cos[:, -1:, :].unsqueeze(1)
            sin = sin[:, -1:, :].unsqueeze(1)
            half = query.shape[-1] // 2
            rotated_half = self._torch.cat(
                (-query[..., half:], query[..., :half]), dim=-1
            )
            rotated = query * cos + rotated_half * sin
            call_index = self._call_counts[layer_index]
            self._attention_queries_by_call[layer_index][call_index] = (
                rotated.detach().clone()
            )

        return observe

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

    def _cache_hook(self, layer_index: int):
        def observe(_module: Any, _inputs: Any, output: Any) -> Any:
            if not self._active:
                return output
            # Transformers 4.40.1, modeling_llama.py::LlamaFlashAttention2.forward,
            # returns its mutable DynamicCache after rotary encoding and cache
            # update. The final sequence entry is the exact cache write caused by
            # the current generation call in pinned OpenVLA 300dce26.
            if not isinstance(output, tuple) or len(output) < 3:
                raise TypeError("OpenVLA attention did not return its decoding cache")
            cache = output[2]
            if cache is None or not hasattr(cache, "key_cache") or not hasattr(
                cache, "value_cache"
            ):
                raise TypeError("OpenVLA did not use the pinned DynamicCache interface")
            call_index = len(self._cache_values_by_call[layer_index]["key"])
            if len(self._cache_values_by_call[layer_index]["value"]) != call_index:
                raise RuntimeError("cache key and value hooks lost call alignment")

            for kind, attribute in (("key", "key_cache"), ("value", "value_cache")):
                layers = getattr(cache, attribute)
                if len(layers) <= layer_index:
                    raise ValueError(
                        f"decoding cache has no {kind} entry for layer {layer_index}"
                    )
                full_value = layers[layer_index]
                if (
                    not isinstance(full_value, self._torch.Tensor)
                    or full_value.ndim != 4
                ):
                    raise TypeError("OpenVLA decoding cache has an unexpected tensor")
                current = full_value[:, :, -1:, :]
                if (
                    call_index == self._token_position
                    and layer_index in self._cache_replacement_layers
                ):
                    source = self._cache_sources[layer_index].get(kind)
                    if source is None:
                        raise ValueError(
                            f"cache source for layer {layer_index} has no {kind} entry"
                        )
                    if (
                        tuple(source.shape) != tuple(current.shape)
                        or source.dtype != current.dtype
                        or source.device != current.device
                    ):
                        raise ValueError(
                            "faulted and replay cache entries differ in schema"
                        )
                    current.copy_(source)
                self._cache_values_by_call[layer_index][kind][call_index] = (
                    current.detach().clone()
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
