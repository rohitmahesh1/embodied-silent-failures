from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record, write_npz_atomic
from embodied_silent_failures.language_campaign import CACHE_REPLAY_STORAGE
from embodied_silent_failures.language_interface import (
    cache_tensor,
    downstream_coordinates,
    internal_tensor,
    sequence_lengths,
    trace_tensor,
)
from embodied_silent_failures.language_policy import PolicyDecision


def _exact_array(torch: Any, value: Any) -> tuple[Any, dict[str, Any]]:
    value = value.detach().cpu().contiguous()
    torch_dtype = str(value.dtype)
    if value.dtype == torch.bfloat16:
        array = value.view(torch.int16).numpy().copy()
        encoding = "raw torch.bfloat16 words stored as NumPy int16 bit patterns"
    else:
        array = value.numpy().copy()
        encoding = "native NumPy numeric dtype"
    return array, {
        "torch_dtype": torch_dtype,
        "numpy_dtype": array.dtype.str,
        "encoding": encoding,
        "shape": list(array.shape),
    }


def _generation_arrays(torch: Any, decisions: list[PolicyDecision]) -> dict[str, Any]:
    if not decisions:
        return {}
    return {
        "sequence_token_ids": torch.stack(
            [value.generation_logits.sequence_token_ids for value in decisions]
        ).numpy(),
        "action_logits": torch.stack(
            [value.generation_logits.action_token_logits for value in decisions]
        ).numpy(),
        "top_token_ids": torch.stack(
            [value.generation_logits.top_token_ids for value in decisions]
        ).numpy(),
        "top_token_logits": torch.stack(
            [value.generation_logits.top_token_logits for value in decisions]
        ).numpy(),
        "log_normalizer": torch.stack(
            [value.generation_logits.log_normalizer for value in decisions]
        ).numpy(),
        "entropy": torch.stack(
            [value.generation_logits.entropy for value in decisions]
        ).numpy(),
    }


def _ragged_calls(
    torch: Any,
    values: dict[int, Any],
    name: str,
    *,
    require_all: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = sorted(set(range(7)) - set(values))
    if require_all and missing:
        raise ValueError(f"context trace is missing {name} calls {missing}")
    call_indices = sorted(values)
    if not call_indices:
        raise ValueError(f"context trace has no {name} calls")
    calls = [values[index].detach().cpu().contiguous() for index in call_indices]
    ranks = {value.ndim for value in calls}
    if len(ranks) != 1:
        raise ValueError(f"context trace changed {name} rank between calls")
    flattened = torch.cat([value.reshape(-1) for value in calls], dim=0)
    offsets = [0]
    for value in calls:
        offsets.append(offsets[-1] + value.numel())
    encoded, encoding = _exact_array(torch, flattened)
    return {
        f"{name}_values": encoded,
        f"{name}_offsets": offsets,
        f"{name}_shapes": [list(value.shape) for value in calls],
        f"{name}_call_indices": call_indices,
    }, encoding


def _context_state_arrays(
    runtime: Any, decision: PolicyDecision
) -> tuple[dict[str, Any], dict[str, Any]]:
    if decision.trace is None:
        raise ValueError("context archive requires a language trace")
    trace = decision.trace
    if trace.initial_pixel_values is None or trace.initial_language_input is None:
        raise ValueError("context archive is missing processed model inputs")
    if set(trace.prompt_cache) != set(range(32)):
        raise ValueError("context archive is missing the full prompt cache")

    arrays = {}
    encodings = {}
    for name, values in (
        ("model_input_ids", trace.model_input_ids_by_call),
        ("model_attention_mask", trace.model_attention_masks_by_call),
        ("decoder_position_ids", trace.decoder_position_ids_by_call),
        ("decoder_cache_position", trace.decoder_cache_positions_by_call),
    ):
        current, encoding = _ragged_calls(runtime.torch, values, name)
        arrays.update(current)
        encodings[name] = encoding

    arrays["decoder_attention_mask_present"] = [
        bool(trace.decoder_attention_mask_present_by_call.get(index, False))
        for index in range(7)
    ]
    if trace.decoder_attention_masks_by_call:
        current, encoding = _ragged_calls(
            runtime.torch,
            trace.decoder_attention_masks_by_call,
            "decoder_attention_mask",
            require_all=False,
        )
        arrays.update(current)
        encodings["decoder_attention_mask"] = encoding

    for name, value in (
        ("processed_pixel_values", trace.initial_pixel_values),
        ("initial_language_input", trace.initial_language_input),
        (
            "prompt_cache_keys",
            runtime.torch.stack(
                [trace.prompt_cache[layer]["key"] for layer in range(32)]
            ),
        ),
        (
            "prompt_cache_values",
            runtime.torch.stack(
                [trace.prompt_cache[layer]["value"] for layer in range(32)]
            ),
        ),
    ):
        arrays[name], encodings[name] = _exact_array(runtime.torch, value)
    return arrays, encodings


def _sparse_trace_values(
    runtime: Any,
    decisions: list[PolicyDecision],
    start_layers: list[int],
    reader: Any,
    *,
    include_selected_layer: bool,
    references: list[PolicyDecision] | None = None,
) -> tuple[Any | None, dict[str, Any] | None, list[int], list[int], list[int]]:
    rows = []
    owners = []
    layers = []
    tokens = []
    encoding = None
    if references is not None and len(references) != len(decisions):
        raise ValueError("replay references do not align with replay decisions")
    reference_values = references or [None] * len(decisions)
    for owner, (start_layer, decision, reference) in enumerate(
        zip(start_layers, decisions, reference_values, strict=True)
    ):
        trace = reader(runtime.torch, decision)
        coordinates = downstream_coordinates(
            start_layer,
            int(decision.trace.action_token_position),
            include_selected_layer=include_selected_layer,
        )
        selected = trace[
            [layer for layer, _token in coordinates],
            [token for _layer, token in coordinates],
        ]
        if reference is not None:
            reference_trace = reader(runtime.torch, reference)
            reference_selected = reference_trace[
                [layer for layer, _token in coordinates],
                [token for _layer, token in coordinates],
            ]
            exact = (selected == reference_selected).reshape(len(coordinates), -1).all(
                dim=-1
            )
            keep = ~exact
            selected = selected[keep]
            coordinates = [
                coordinate
                for coordinate, retained in zip(
                    coordinates, keep.tolist(), strict=True
                )
                if retained
            ]
        if not coordinates:
            continue
        encoded, current_encoding = _exact_array(runtime.torch, selected)
        encoding = encoding or current_encoding
        if current_encoding["torch_dtype"] != encoding["torch_dtype"]:
            raise ValueError("traced tensor dtype changed within one context")
        rows.append(encoded)
        owners.extend([owner] * len(coordinates))
        layers.extend(layer for layer, _token in coordinates)
        tokens.extend(token for _layer, token in coordinates)
    values = runtime.np.concatenate(rows, axis=0) if rows else None
    if values is not None and encoding is not None:
        encoding = {**encoding, "shape": list(values.shape)}
    return values, encoding, owners, layers, tokens


@dataclass
class InterfaceArchiveBuilder:
    runtime: Any
    source: PolicyDecision
    clean: PolicyDecision
    fault_layers: list[int] = field(default_factory=list)
    faults: list[PolicyDecision] = field(default_factory=list)
    replay_injection_layers: list[int] = field(default_factory=list)
    replay_boundary_layers: list[int] = field(default_factory=list)
    replay_kinds: list[int] = field(default_factory=list)
    replays: list[PolicyDecision] = field(default_factory=list)

    def add_fault(self, layer_index: int, decision: PolicyDecision) -> None:
        self.fault_layers.append(int(layer_index))
        self.faults.append(decision)

    def add_replay(
        self,
        *,
        injection_layer: int,
        boundary_layer: int,
        boundary_kind: str,
        decision: PolicyDecision,
    ) -> None:
        kind_ids = {"immediate": 0, "final": 1}
        if boundary_kind not in kind_ids:
            raise ValueError(f"unknown boundary replay kind: {boundary_kind}")
        self.replay_injection_layers.append(int(injection_layer))
        self.replay_boundary_layers.append(int(boundary_layer))
        self.replay_kinds.append(kind_ids[boundary_kind])
        self.replays.append(decision)

    def write(self, path: Path) -> dict[str, Any]:
        torch = self.runtime.torch
        np = self.runtime.np
        arrays: dict[str, Any] = {
            "fault_layer": np.asarray(self.fault_layers, dtype=np.int16),
            "replay_injection_layer": np.asarray(
                self.replay_injection_layers, dtype=np.int16
            ),
            "replay_boundary_layer": np.asarray(
                self.replay_boundary_layers, dtype=np.int16
            ),
            "replay_kind": np.asarray(self.replay_kinds, dtype=np.uint8),
            "source_block_sequence_lengths": sequence_lengths(np, self.source),
            "clean_block_sequence_lengths": sequence_lengths(np, self.clean),
        }
        if self.faults:
            arrays["fault_block_sequence_lengths"] = np.stack(
                [sequence_lengths(np, decision) for decision in self.faults]
            )
        if self.replays:
            arrays["replay_block_sequence_lengths"] = np.stack(
                [sequence_lengths(np, decision) for decision in self.replays]
            )
        encodings = {}
        row_counts = {}
        fault_by_layer = dict(zip(self.fault_layers, self.faults, strict=True))
        replay_references = [
            fault_by_layer[layer] for layer in self.replay_injection_layers
        ]
        readers = {
            "residuals": (trace_tensor, True, True, "boundary"),
            "block_inputs": (
                lambda module, decision: internal_tensor(
                    module, decision, "block_inputs_by_call"
                ),
                False,
                False,
                "boundary",
            ),
            "post_attention_residuals": (
                lambda module, decision: internal_tensor(
                    module, decision, "post_attention_residuals_by_call"
                ),
                False,
                False,
                "boundary",
            ),
            "attention_queries": (
                lambda module, decision: internal_tensor(
                    module, decision, "attention_queries_by_call"
                ),
                False,
                False,
                "boundary",
            ),
            "attention_cache_keys": (
                lambda module, decision: cache_tensor(module, decision, "key"),
                False,
                False,
                "fault",
            ),
            "attention_cache_values": (
                lambda module, decision: cache_tensor(module, decision, "value"),
                False,
                False,
                "fault",
            ),
        }
        detailed = bool(self.clean.trace and self.clean.trace.prompt_cache)
        if not detailed:
            for name in (
                "block_inputs",
                "post_attention_residuals",
                "attention_queries",
            ):
                readers.pop(name)
        for name, (
            reader,
            fault_includes_boundary,
            replay_includes_boundary,
            replay_start,
        ) in readers.items():
            for prefix, decision in (("source", self.source), ("clean", self.clean)):
                encoded, encoding = _exact_array(torch, reader(torch, decision))
                arrays[f"{prefix}_{name}"] = encoded
                encodings[f"{prefix}_{name}"] = encoding

            fault_values, fault_encoding, *current_fault_indices = _sparse_trace_values(
                self.runtime,
                self.faults,
                self.fault_layers,
                reader,
                include_selected_layer=fault_includes_boundary,
            )
            replay_values, replay_encoding, *current_replay_indices = (
                _sparse_trace_values(
                    self.runtime,
                    self.replays,
                    (
                        self.replay_injection_layers
                        if replay_start == "fault"
                        else self.replay_boundary_layers
                    ),
                    reader,
                    include_selected_layer=replay_includes_boundary,
                    references=replay_references,
                )
            )
            if fault_values is not None:
                arrays[f"fault_{name}"] = fault_values
            if replay_values is not None:
                arrays[f"replay_{name}"] = replay_values
            encodings[f"fault_{name}"] = fault_encoding
            encodings[f"replay_{name}"] = replay_encoding
            row_counts[f"fault_{name}"] = len(current_fault_indices[1])
            row_counts[f"replay_{name}"] = len(current_replay_indices[1])
            if current_fault_indices:
                arrays[f"fault_{name}_row_intervention"] = np.asarray(
                    current_fault_indices[0], dtype=np.int16
                )
                arrays[f"fault_{name}_row_layer"] = np.asarray(
                    current_fault_indices[1], dtype=np.int16
                )
                arrays[f"fault_{name}_row_token"] = np.asarray(
                    current_fault_indices[2], dtype=np.int8
                )
            if current_replay_indices:
                arrays[f"replay_{name}_row_replay"] = np.asarray(
                    current_replay_indices[0], dtype=np.int16
                )
                arrays[f"replay_{name}_row_layer"] = np.asarray(
                    current_replay_indices[1], dtype=np.int16
                )
                arrays[f"replay_{name}_row_token"] = np.asarray(
                    current_replay_indices[2], dtype=np.int8
                )

        for prefix, decisions in (
            ("source", [self.source]),
            ("clean", [self.clean]),
            ("fault", self.faults),
            ("replay", self.replays),
        ):
            for name, value in _generation_arrays(torch, decisions).items():
                arrays[f"{prefix}_{name}"] = value

        context_state_encodings = {}
        if detailed:
            for prefix, decision in (("source", self.source), ("clean", self.clean)):
                current, current_encodings = _context_state_arrays(
                    self.runtime, decision
                )
                arrays.update(
                    {f"{prefix}_{name}": value for name, value in current.items()}
                )
                context_state_encodings[prefix] = current_encodings

        write_npz_atomic(path, np, arrays)
        return {
            "schema_version": 3 if detailed else 2,
            "artifact": artifact_record(path),
            "residual_axes": ["language_block", "action_token_position", "hidden"],
            "trace_encodings": encodings,
            "context_state_encodings": context_state_encodings,
            "fault_records": len(self.faults),
            "boundary_replay_records": len(self.replays),
            "trace_row_counts": row_counts,
            "trace_port_semantics": {
                "residuals": (
                    "post-block output; the selected boundary row is the exact "
                    "intervention or replay value"
                ),
                "block_inputs": (
                    "final sequence position entering each language block; the "
                    "selected fault layer is excluded because replacement occurs "
                    "at that block's output"
                ),
                "post_attention_residuals": (
                    "final sequence position entering post_attention_layernorm, "
                    "which is the exact residual after the attention sublayer"
                ),
                "attention_queries": (
                    "exact final-position query after rotary position encoding, "
                    "organized by language block, generation call, head, and head feature"
                ),
                "attention_cache_keys": (
                    "exact post-rotary cache entry appended on each generation call; "
                    "call zero writes the final prompt position and calls one through "
                    "six write the previously emitted action tokens. Only differential "
                    "fault rows are stored after the clean full prompt cache"
                ),
                "attention_cache_values": (
                    "exact value entry appended for the current token, with the same "
                    "differential-cache scope as the key entry"
                ),
            },
            "boundary_state": (
                "post-block residual plus exact current-token key/value cache entries "
                "from the fault output through the replay boundary"
            ),
            "replay_storage": CACHE_REPLAY_STORAGE,
            "replay_kind_ids": {"0": "immediate", "1": "final"},
            "generation_logits": {
                "action_vocabulary_entries": 256,
                "global_top_entries": 32,
                "normalization": "exact full-vocabulary logsumexp and entropy",
                "sequence_token_ids": (
                    "complete prompt and generated token IDs returned by generate"
                ),
            },
            "context_state": (
                {
                    "model_inputs": (
                        "exact per-call input IDs and masks, initial processed pixels, "
                        "and the complete fused sequence entering language block zero"
                    ),
                    "prompt_cache": (
                        "all 32 key/value layers at the start of generation call one, "
                        "after the prompt pass and before action token zero is consumed"
                    ),
                    "prompt_cache_formats": {
                        "source": self.source.trace.prompt_cache_format,
                        "clean": self.clean.trace.prompt_cache_format,
                    },
                    "attention_conditioning": (
                        "post-rotary current queries plus the prompt cache and every "
                        "subsequent key/value write"
                    ),
                }
                if detailed
                else None
            ),
        }
