from __future__ import annotations

from typing import Any

from embodied_silent_failures.language_policy import PolicyDecision, array_change


LANGUAGE_BLOCK_COUNT = 32
ACTION_TOKEN_COUNT = 7


def trace_tensor(torch: Any, decision: PolicyDecision) -> Any:
    if decision.trace is None:
        raise ValueError("cannot read a decision without a language trace")
    rows = []
    for layer_index in range(LANGUAGE_BLOCK_COUNT):
        calls = decision.trace.block_values_by_call.get(layer_index, {})
        missing = sorted(set(range(ACTION_TOKEN_COUNT)) - set(calls))
        if missing:
            raise ValueError(
                f"language block {layer_index} is missing generation calls {missing}"
            )
        rows.append(
            torch.stack(
                [calls[token][0, 0, :] for token in range(ACTION_TOKEN_COUNT)],
                dim=0,
            )
        )
    return torch.stack(rows, dim=0).detach().cpu().contiguous()


def attention_tensor(torch: Any, decision: PolicyDecision, kind: str) -> Any:
    if decision.trace is None:
        raise ValueError("cannot read a decision without a language trace")
    rows = []
    for layer_index in range(LANGUAGE_BLOCK_COUNT):
        calls = decision.trace.attention_values_by_call.get(layer_index, {}).get(
            kind, {}
        )
        missing = sorted(set(range(ACTION_TOKEN_COUNT)) - set(calls))
        if missing:
            raise ValueError(
                f"language block {layer_index} is missing {kind} projection calls "
                f"{missing}"
            )
        rows.append(
            torch.stack(
                [calls[token][0, 0, :] for token in range(ACTION_TOKEN_COUNT)],
                dim=0,
            )
        )
    return torch.stack(rows, dim=0).detach().cpu().contiguous()


def sequence_lengths(np: Any, decision: PolicyDecision) -> Any:
    if decision.trace is None:
        raise ValueError("cannot read a decision without a language trace")
    values = np.empty((LANGUAGE_BLOCK_COUNT, ACTION_TOKEN_COUNT), dtype=np.int32)
    for layer_index in range(LANGUAGE_BLOCK_COUNT):
        calls = decision.trace.sequence_lengths_by_call.get(layer_index, {})
        missing = sorted(set(range(ACTION_TOKEN_COUNT)) - set(calls))
        if missing:
            raise ValueError(
                f"language block {layer_index} is missing sequence lengths {missing}"
            )
        values[layer_index] = [
            int(calls[token]) for token in range(ACTION_TOKEN_COUNT)
        ]
    return values


def downstream_coordinates(
    layer_index: int,
    token_position: int,
    *,
    include_selected_layer: bool,
) -> list[tuple[int, int]]:
    # The replacement occurs at one block output. On that generation call, the
    # changed output reaches that port and later blocks, but the selected
    # block's internal attention projections have already run. Earlier calls
    # are complete; every block on later calls can receive changed token or
    # attention-cache state.
    coordinates = []
    for token in range(token_position, ACTION_TOKEN_COUNT):
        if token == token_position:
            first_layer = layer_index if include_selected_layer else layer_index + 1
        else:
            first_layer = 0
        coordinates.extend(
            (block, token) for block in range(first_layer, LANGUAGE_BLOCK_COUNT)
        )
    return coordinates


def boundary_replay_targets(
    injection_layer: int, kinds: list[str]
) -> list[tuple[str, int]]:
    targets = []
    if "immediate" in kinds and injection_layer < LANGUAGE_BLOCK_COUNT - 1:
        targets.append(("immediate", injection_layer + 1))
    if "final" in kinds and injection_layer < LANGUAGE_BLOCK_COUNT - 2:
        targets.append(("final", LANGUAGE_BLOCK_COUNT - 1))
    unknown = sorted(set(kinds) - {"immediate", "final"})
    if unknown:
        raise ValueError(f"unknown boundary replay kinds: {unknown}")
    return targets


def trace_repeatability(
    torch: Any, left: PolicyDecision, right: PolicyDecision
) -> dict[str, Any]:
    readers = {
        "residuals": trace_tensor,
        "attention_key_projections": lambda module, decision: attention_tensor(
            module, decision, "key"
        ),
        "attention_value_projections": lambda module, decision: attention_tensor(
            module, decision, "value"
        ),
    }
    result = {}
    for name, reader in readers.items():
        left_value = reader(torch, left)
        right_value = reader(torch, right)
        exact = left_value == right_value
        per_coordinate = exact.reshape(
            LANGUAGE_BLOCK_COUNT, ACTION_TOKEN_COUNT, -1
        ).all(dim=-1)
        result[name] = {
            "all_exact": bool(per_coordinate.all().item()),
            "exact_coordinates": int(per_coordinate.sum().item()),
            "total_coordinates": LANGUAGE_BLOCK_COUNT * ACTION_TOKEN_COUNT,
        }
    return result


def boundary_replay_record(
    runtime: Any,
    *,
    original: PolicyDecision,
    replay: PolicyDecision,
    injection_layer: int,
    boundary_layer: int,
    boundary_kind: str,
) -> dict[str, Any]:
    if original.trace is None or replay.trace is None:
        raise ValueError("boundary replay requires two complete language traces")
    token_position = int(original.trace.action_token_position)
    # Include the boundary for every port here. The residual at that boundary
    # is the value replayed exactly; its key/value projections happened before
    # the output replacement and therefore expose state omitted by an
    # output-only replay.
    coordinates = downstream_coordinates(
        boundary_layer, token_position, include_selected_layer=True
    )

    def compare(getter: Any) -> dict[str, Any]:
        first_difference = None
        maximum_l2 = 0.0
        exact_coordinates = 0
        for layer, token in coordinates:
            left = getter(original.trace, layer, token)
            right = getter(replay.trace, layer, token)
            exact = bool(runtime.torch.equal(left, right))
            if exact:
                exact_coordinates += 1
                continue
            difference = left.detach().to(runtime.torch.float32) - right.detach().to(
                runtime.torch.float32
            )
            l2 = float(runtime.torch.linalg.vector_norm(difference).item())
            maximum_l2 = max(maximum_l2, l2)
            if first_difference is None:
                first_difference = {
                    "layer_index": layer,
                    "action_token_position": token,
                }
        return {
            "compared_coordinates": len(coordinates),
            "exact_coordinates": exact_coordinates,
            "all_coordinates_exact": exact_coordinates == len(coordinates),
            "first_difference": first_difference,
            "maximum_coordinate_difference_l2": maximum_l2,
        }

    residual_comparison = compare(
        lambda trace, layer, token: trace.block_values_by_call[layer][token]
    )
    key_projection_comparison = compare(
        lambda trace, layer, token: trace.attention_values_by_call[layer]["key"][
            token
        ]
    )
    value_projection_comparison = compare(
        lambda trace, layer, token: trace.attention_values_by_call[layer]["value"][
            token
        ]
    )
    logits_difference = (
        original.generation_logits.action_token_logits
        - replay.generation_logits.action_token_logits
    )
    return {
        "status": "complete",
        "injection_layer": injection_layer,
        "boundary_layer": boundary_layer,
        "boundary_kind": boundary_kind,
        "action_token_position": token_position,
        "residual_path": residual_comparison,
        "attention_key_projections": key_projection_comparison,
        "attention_value_projections": value_projection_comparison,
        "action_tokens_exact_equal": original.action_tokens == replay.action_tokens,
        "raw_action": array_change(runtime.np, original.raw_action, replay.raw_action),
        "executed_command": array_change(
            runtime.np, original.command, replay.command
        ),
        "action_logit_difference_l2": float(
            runtime.torch.linalg.vector_norm(logits_difference).item()
        ),
    }
