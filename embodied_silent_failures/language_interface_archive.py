from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record, write_npz_atomic
from embodied_silent_failures.language_interface import (
    attention_tensor,
    downstream_coordinates,
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


def _sparse_trace_values(
    runtime: Any,
    decisions: list[PolicyDecision],
    start_layers: list[int],
    reader: Any,
    *,
    include_selected_layer: bool,
) -> tuple[Any | None, dict[str, Any] | None, list[int], list[int], list[int]]:
    rows = []
    owners = []
    layers = []
    tokens = []
    encoding = None
    for owner, (start_layer, decision) in enumerate(
        zip(start_layers, decisions, strict=True)
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
        readers = {
            "residuals": (trace_tensor, True, True),
            "attention_key_projections": (
                lambda module, decision: attention_tensor(module, decision, "key"),
                False,
                True,
            ),
            "attention_value_projections": (
                lambda module, decision: attention_tensor(module, decision, "value"),
                False,
                True,
            ),
        }
        for name, (reader, fault_includes_boundary, replay_includes_boundary) in (
            readers.items()
        ):
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
                    self.replay_boundary_layers,
                    reader,
                    include_selected_layer=replay_includes_boundary,
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

        write_npz_atomic(path, np, arrays)
        return {
            "schema_version": 1,
            "artifact": artifact_record(path),
            "residual_axes": ["language_block", "action_token_position", "hidden"],
            "trace_encodings": encodings,
            "fault_records": len(self.faults),
            "boundary_replay_records": len(self.replays),
            "trace_row_counts": row_counts,
            "trace_port_semantics": {
                "residuals": (
                    "post-block output; the selected boundary row is the exact "
                    "intervention or replay value"
                ),
                "attention_key_projections": (
                    "pre-rotary key projection inside a block; a fault excludes "
                    "its selected block because this port runs before the output "
                    "replacement, while a replay includes its boundary to expose "
                    "state omitted by output-only replay"
                ),
                "attention_value_projections": (
                    "value projection inside a block; row inclusion follows the "
                    "same pre-output rule as the key projection"
                ),
            },
            "replay_kind_ids": {"0": "immediate", "1": "final"},
            "generation_logits": {
                "action_vocabulary_entries": 256,
                "global_top_entries": 32,
                "normalization": "exact full-vocabulary logsumexp and entropy",
                "sequence_token_ids": (
                    "complete prompt and generated token IDs returned by generate"
                ),
            },
        }
