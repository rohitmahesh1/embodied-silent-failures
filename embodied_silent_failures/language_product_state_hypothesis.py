from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record
from embodied_silent_failures.provenance import load_json


OUTCOMES = (
    "task_failure",
    "operational_silent_failure",
    "monitor_miss_given_failure",
)
STATE_DESCRIPTIONS = ("observation_product", "simulator_product")
FEATURE_STAGES = ("product", "product_and_origin", "product_origin_and_path")
PHASES = ("early", "middle", "late")
TASK_COUNT = 10
TOKEN_COUNT = 7
LAYER_COUNT = 32


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected a serialized boolean, found {value!r}")


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _unpack(np: Any, archive: Any, name: str, index: int) -> Any:
    offsets = archive[f"{name}_offsets"]
    shape = tuple(int(value) for value in archive[f"{name}_shapes"][index])
    start, stop = int(offsets[index]), int(offsets[index + 1])
    return np.asarray(archive[f"{name}_values"][start:stop]).reshape(shape).copy()


def _unpack_state(np: Any, archive: Any, moment: str, index: int) -> Any:
    offsets = archive["numeric_state_offsets"]
    shape = tuple(int(value) for value in archive["numeric_state_shapes"][index])
    start, stop = int(offsets[index]), int(offsets[index + 1])
    return (
        np.asarray(archive[f"numeric_state_{moment}_values"][start:stop])
        .reshape(shape)
        .copy()
    )


def _load_shard(
    np: Any, shard_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(shard_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"product-state shard is not complete: {shard_dir}")
    for record in manifest["artifacts"]:
        path = shard_dir / str(record["name"])
        if artifact_record(path) != record:
            raise ValueError(f"product-state artifact changed: {path}")

    branches = _read_csv(shard_dir / "branches.csv")
    interventions = _read_csv(shard_dir / "interventions.csv")
    state_entries = _read_csv(shard_dir / "state-entries.csv")
    worker_values = {int(row["worker_shard"]) for row in interventions}
    if len(worker_values) != 1:
        raise ValueError(f"shard has ambiguous worker identity: {shard_dir}")
    worker_shard = next(iter(worker_values))

    branch_by_index = {int(row["branch_index"]): row for row in branches}
    if len(branch_by_index) != len(branches):
        raise ValueError(f"branch indices repeat: {shard_dir}")
    controls = {
        str(row["context_id"]): row
        for row in branches
        if row["condition"] == "activation_control"
    }
    if len(controls) != len({row["context_id"] for row in branches}):
        raise ValueError(f"each context must have one extracted control: {shard_dir}")

    state_index: dict[tuple[int, str], int] = {}
    for row in state_entries:
        key = (int(row["branch_index"]), str(row["name"]))
        if key in state_index:
            raise ValueError(f"state entry repeats: {key}")
        state_index[key] = int(row["state_entry_index"])

    rows = []
    with np.load(shard_dir / "product-state.npz", allow_pickle=False) as archive:
        for intervention in interventions:
            if not _as_bool(intervention["product_state_available"]):
                continue
            eligible = _as_bool(intervention["eligible_causal_outcome"])
            command_changed = _as_bool(intervention["command_changed"])
            if not eligible or not command_changed:
                continue
            branch = branch_by_index[int(intervention["branch_index"])]
            control = controls[str(intervention["context_id"])]
            branch_index = int(branch["branch_index"])
            control_index = int(control["branch_index"])
            state = {}
            for name in ("robot0_proprio-state", "object-state", "simulator_state"):
                branch_state_index = state_index[(branch_index, name)]
                control_state_index = state_index[(control_index, name)]
                state[name] = {
                    "before": _unpack_state(np, archive, "before", branch_state_index),
                    "after": _unpack_state(np, archive, "after", branch_state_index),
                    "control_after": _unpack_state(
                        np, archive, "after", control_state_index
                    ),
                }
            fault_command = _unpack(np, archive, "executed_command", branch_index)
            control_command = _unpack(np, archive, "executed_command", control_index)
            threshold = _as_float(branch["safe_threshold_at_fault"], "SAFE threshold")
            if threshold <= 0:
                raise ValueError("SAFE threshold must be positive")
            row = dict(intervention)
            row.update(
                {
                    "worker_shard": worker_shard,
                    "task_id": int(intervention["task_id"]),
                    "episode_index": int(intervention["episode_index"]),
                    "action_token_position": int(
                        intervention["action_token_position"]
                    ),
                    "layer_index": int(intervention["layer_index"]),
                    "command_group_size": int(intervention["command_group_size"]),
                    "eligible_causal_outcome": eligible,
                    "command_changed": command_changed,
                    "task_failure": _as_bool(intervention["task_failure"]),
                    "operational_silent_failure": _as_bool(
                        intervention["operational_silent_failure"]
                    ),
                    "fault_command": np.asarray(fault_command, dtype=np.float64),
                    "control_command": np.asarray(control_command, dtype=np.float64),
                    "state": state,
                    "safe_threshold_at_fault": threshold,
                }
            )
            rows.append(row)
    return rows, {
        "directory": str(shard_dir.resolve()),
        "manifest": artifact_record(shard_dir / "manifest.json"),
        "worker_shard": worker_shard,
        "rows": len(rows),
    }


def load_product_state(
    np: Any, shard_dirs: list[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    sources = []
    for shard_dir in shard_dirs:
        shard_rows, source = _load_shard(np, shard_dir)
        rows.extend(shard_rows)
        sources.append(source)
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("product-state shards repeat intervention records")
    return rows, {"shards": sources, "rows": len(rows)}


def eligible_rows(rows: list[dict[str, Any]], outcome: str) -> list[dict[str, Any]]:
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome}")
    result = [
        row
        for row in rows
        if row["eligible_causal_outcome"] and row["command_changed"]
    ]
    if outcome == "monitor_miss_given_failure":
        result = [row for row in result if row["task_failure"]]
        for row in result:
            row["monitor_miss_given_failure"] = row[
                "operational_silent_failure"
            ]
    return result


def _pad(np: Any, value: Any, width: int) -> Any:
    flattened = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(flattened) > width:
        raise ValueError(f"state width {len(flattened)} exceeds declared width {width}")
    result = np.zeros(width, dtype=np.float64)
    result[: len(flattened)] = flattened
    return result


def state_widths(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        name: max(len(row["state"][name]["before"].reshape(-1)) for row in rows)
        for name in ("robot0_proprio-state", "object-state", "simulator_state")
    }


def _one_hot(np: Any, index: int, width: int, label: str) -> Any:
    if index < 0 or index >= width:
        raise ValueError(f"{label} index {index} is outside [0, {width})")
    result = np.zeros(width, dtype=np.float64)
    result[index] = 1.0
    return result


def feature_names(
    description: str, stage: str, widths: dict[str, int]
) -> list[str]:
    if description not in STATE_DESCRIPTIONS or stage not in FEATURE_STAGES:
        raise ValueError(f"unknown feature description: {description}/{stage}")
    names = [f"task_{index}" for index in range(TASK_COUNT)]
    names += [f"phase_{phase}" for phase in PHASES]
    for prefix in ("control_command", "fault_command", "command_delta"):
        names += [f"{prefix}_{index}" for index in range(7)]
    names += [
        "safe_fault_ratio",
        "safe_control_ratio",
        "safe_score_shift_ratio",
        "safe_feature_normalized_l2",
    ]
    state_keys = (
        ("robot0_proprio-state", "proprio"),
        ("object-state", "object"),
    )
    if description == "simulator_product":
        state_keys = (("simulator_state", "simulator"),)
    for source, label in state_keys:
        for moment in ("before", "control_after", "after", "fault_minus_control"):
            names += [f"{label}_{moment}_{index}" for index in range(widths[source])]
    if stage in ("product_and_origin", "product_origin_and_path"):
        names += [f"action_token_{index}" for index in range(TOKEN_COUNT)]
        names += [f"source_layer_{index}" for index in range(LAYER_COUNT)]
    if stage == "product_origin_and_path":
        names += [
            "log_injection_l2",
            "injection_normalized_l2",
            "log_final_propagation_l2",
            "final_propagation_normalized_l2",
        ]
    return names


def feature_vector(
    np: Any,
    row: dict[str, Any],
    description: str,
    stage: str,
    widths: dict[str, int],
) -> Any:
    phase = str(row["phase"])
    if phase not in PHASES:
        raise ValueError(f"unknown rollout phase: {phase}")
    threshold = float(row["safe_threshold_at_fault"])
    control_command = np.asarray(row["control_command"], dtype=np.float64)
    fault_command = np.asarray(row["fault_command"], dtype=np.float64)
    values = [
        _one_hot(np, int(row["task_id"]), TASK_COUNT, "task"),
        _one_hot(np, PHASES.index(phase), len(PHASES), "phase"),
        control_command,
        fault_command,
        fault_command - control_command,
        np.asarray(
            [
                _as_float(row["score_at_fault"], "fault SAFE score") / threshold,
                _as_float(row["control_score_at_fault"], "control SAFE score")
                / threshold,
                _as_float(
                    row["score_change_from_control_at_fault"], "SAFE score shift"
                )
                / threshold,
                _as_float(
                    row["safe_feature_normalized_l2"], "SAFE feature displacement"
                ),
            ],
            dtype=np.float64,
        ),
    ]
    state_keys = ("robot0_proprio-state", "object-state")
    if description == "simulator_product":
        state_keys = ("simulator_state",)
    for name in state_keys:
        state = row["state"][name]
        before = _pad(np, state["before"], widths[name])
        control_after = _pad(np, state["control_after"], widths[name])
        after = _pad(np, state["after"], widths[name])
        values.extend((before, control_after, after, after - control_after))
    if stage in ("product_and_origin", "product_origin_and_path"):
        values.append(
            _one_hot(
                np,
                int(row["action_token_position"]),
                TOKEN_COUNT,
                "action token",
            )
        )
        values.append(
            _one_hot(np, int(row["layer_index"]), LAYER_COUNT, "source layer")
        )
    if stage == "product_origin_and_path":
        values.append(
            np.asarray(
                [
                    math.log1p(_as_float(row["injection_l2"], "injection L2")),
                    _as_float(
                        row["injection_normalized_l2"], "normalized injection L2"
                    ),
                    math.log1p(
                        _as_float(
                            row["final_propagation_l2"], "final propagation L2"
                        )
                    ),
                    _as_float(
                        row["final_propagation_normalized_l2"],
                        "normalized final propagation L2",
                    ),
                ],
                dtype=np.float64,
            )
        )
    result = np.concatenate(values)
    expected = feature_names(description, stage, widths)
    if len(result) != len(expected) or not np.isfinite(result).all():
        raise ValueError("product-state feature vector is malformed")
    return result


def feature_matrix(
    np: Any,
    rows: list[dict[str, Any]],
    description: str,
    stage: str,
    widths: dict[str, int],
) -> Any:
    return np.asarray(
        [feature_vector(np, row, description, stage, widths) for row in rows],
        dtype=np.float64,
    )
