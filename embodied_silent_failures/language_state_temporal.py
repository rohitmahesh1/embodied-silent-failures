from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from embodied_silent_failures.artifacts import artifact_record
from embodied_silent_failures.language_gates import COMMAND_COMPONENTS
from embodied_silent_failures.openvla_runtime import array_sha256
from embodied_silent_failures.provenance import file_sha256, load_json


NEIGHBOR_COUNTS = (1, 3, 5)
TEMPORAL_HORIZONS = (0, 5, 10, 25)
DISTANCE_MODES = (
    "delta_command",
    "executed_command",
    "state",
    "state_and_executed_command",
)


def load_context_states(
    campaign_dirs: list[Path], np: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and content-check one raw simulator state per completed context."""
    states = {}
    shape_counts: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    archive_bytes = 0
    runtime = SimpleNamespace(np=np)
    for campaign_dir in campaign_dirs:
        for local_path in sorted((campaign_dir / "contexts").glob("*/local.json")):
            local = load_json(local_path)
            context = local["context"]
            context_id = str(context["context_id"])
            if context_id in states:
                raise ValueError(f"duplicate captured context: {context_id}")
            manifest = local.get("captured_context_archive")
            if not isinstance(manifest, dict):
                raise ValueError(f"captured context has no archive: {context_id}")
            path = local_path.parent / str(manifest["artifact"]["name"])
            if artifact_record(path) != manifest["artifact"]:
                raise ValueError(f"captured context archive changed: {path}")
            with np.load(path, allow_pickle=False) as archive:
                state_record = manifest["simulator_state"]
                state = np.asarray(archive[str(state_record["archive_key"])]).copy()
            digest = array_sha256(runtime, state)
            if digest != state_record["sha256"]:
                raise ValueError(f"captured simulator state hash changed: {path}")
            if digest != local["captured_simulator_state_sha256"]:
                raise ValueError(f"state archive and context disagree: {context_id}")
            if list(state.shape) != state_record["shape"]:
                raise ValueError(f"state archive shape changed: {context_id}")
            task_id = int(context["task_id"])
            shape_counts[(task_id, tuple(state.shape))] += 1
            archive_bytes += int(manifest["artifact"]["bytes"])
            states[context_id] = {
                "context_id": context_id,
                "task_id": task_id,
                "episode_index": int(context["episode_index"]),
                "state": state.astype(float).tolist(),
                "state_sha256": digest,
            }
    task_shapes: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for task_id, shape in shape_counts:
        task_shapes[task_id].add(shape)
    inconsistent = {
        task_id: sorted(shapes)
        for task_id, shapes in task_shapes.items()
        if len(shapes) != 1
    }
    if inconsistent:
        raise ValueError(f"simulator state shape varies within a task: {inconsistent}")
    return states, {
        "contexts": len(states),
        "archive_bytes": archive_bytes,
        "task_state_schemas": [
            {
                "task_id": task_id,
                "shape": list(shape),
                "contexts": count,
            }
            for (task_id, shape), count in sorted(shape_counts.items())
        ],
    }


def load_physical_score_traces(
    campaign_dirs: list[Path], np: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    traces = {}
    monitor_hashes = set()
    primary_alphas = set()
    source_artifacts = []
    for campaign_dir in campaign_dirs:
        scoring = campaign_dir / "scoring"
        json_path = scoring / "physical-safe.json"
        archive_path = scoring / "physical-safe.npz"
        document = load_json(json_path)
        if file_sha256(archive_path) != document["score_archive"]["sha256"]:
            raise ValueError(f"physical SAFE score archive changed: {archive_path}")
        monitor_hashes.add(str(document["monitor"]["checkpoint_sha256"]))
        primary = float(document["monitor"]["primary_alpha"])
        primary_alphas.add(primary)
        with np.load(archive_path, allow_pickle=False) as archive:
            runs = [str(value) for value in archive["runs"]]
            lengths = archive["lengths"].astype(int)
            scores = archive["scores"]
            alphas = archive["alphas"].astype(float).tolist()
            matches = [
                index
                for index, alpha in enumerate(alphas)
                if math.isclose(alpha, primary, rel_tol=0, abs_tol=1e-8)
            ]
            if len(matches) != 1:
                raise ValueError("physical SAFE archive has no unique primary band")
            band = archive["bands"][matches[0]].astype(float)
            if len(runs) != len(document["records"]):
                raise ValueError("physical SAFE JSON and archive lengths disagree")
            for index, (run, record) in enumerate(
                zip(runs, document["records"], strict=True)
            ):
                if run != str(record["run"]):
                    raise ValueError("physical SAFE JSON and archive ordering changed")
                if run in traces:
                    raise ValueError(f"duplicate physical SAFE trace: {run}")
                length = int(lengths[index])
                traces[run] = {
                    "scores": scores[index, :length].astype(float).tolist(),
                    "band": band[:length].astype(float).tolist(),
                    "fault_step": int(record["fault"]["policy_step"]),
                    "success": bool(record["success"]),
                }
        source_artifacts.extend(
            [
                {"path": str(json_path.resolve()), "sha256": file_sha256(json_path)},
                {
                    "path": str(archive_path.resolve()),
                    "sha256": file_sha256(archive_path),
                },
            ]
        )
    if len(monitor_hashes) != 1 or len(primary_alphas) != 1:
        raise ValueError("physical SAFE traces do not use one frozen monitor")
    return traces, {
        "traces": len(traces),
        "monitor_checkpoint_sha256": next(iter(monitor_hashes)),
        "primary_alpha": next(iter(primary_alphas)),
        "artifacts": source_artifacts,
    }


def temporal_margin_features(trace: dict[str, Any]) -> dict[str, float]:
    scores = [float(value) for value in trace["scores"]]
    band = [float(value) for value in trace["band"]]
    step = int(trace["fault_step"])
    if len(scores) != len(band) or step < 0 or step >= len(scores):
        raise ValueError("physical SAFE trace has inconsistent dimensions")
    if not all(math.isfinite(value) and value > 0 for value in band):
        raise ValueError("physical SAFE threshold band must be positive and finite")
    ratios = [score / threshold for score, threshold in zip(scores, band, strict=True)]
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("physical SAFE trace contains non-finite normalized scores")
    result = {}
    for horizon in TEMPORAL_HORIZONS:
        stop = step + 1 if horizon == 0 else min(len(ratios), step + horizon)
        if stop <= step:
            raise ValueError("SAFE temporal window is empty")
        # score_safe.py::alarm_windows uses [fault_step, fault_step + horizon).
        # Distance below one therefore measures sub-threshold evidence under the
        # same frozen alarm rule, rather than defining a new monitor threshold.
        result[f"monitor_margin_{horizon}"] = 1.0 - max(ratios[step:stop])
    return result


def attach_interfaces(
    branches: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for branch in branches:
        context_id = str(branch["context_id"])
        physical_run = str(branch["physical_run"])
        if context_id not in states:
            raise ValueError(f"physical branch has no captured state: {context_id}")
        if physical_run not in traces:
            raise ValueError(f"physical branch has no SAFE trace: {physical_run}")
        state = states[context_id]
        trace = traces[physical_run]
        if int(branch["task_id"]) != int(state["task_id"]):
            raise ValueError(f"branch and state task disagree: {physical_run}")
        if bool(branch["task_failure"]) == bool(trace["success"]):
            raise ValueError(f"branch and physical SAFE outcome disagree: {physical_run}")
        row = dict(branch)
        row["state"] = list(state["state"])
        row["state_sha256"] = state["state_sha256"]
        row["delta_command"] = [
            float(row[f"delta_{name}"]) for name in COMMAND_COMPONENTS
        ]
        row["executed_command"] = [
            float(row[f"faulted_{name}"]) for name in COMMAND_COMPONENTS
        ]
        row.update(temporal_margin_features(trace))
        result.append(row)
    return result


def _scales(values: list[list[float]]) -> list[float]:
    if not values or not values[0]:
        raise ValueError("cannot scale an empty feature block")
    width = len(values[0])
    if any(len(value) != width for value in values):
        raise ValueError("feature block dimensions disagree")
    means = [sum(row[index] for row in values) / len(values) for index in range(width)]
    result = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in values) / len(values)
        scale = math.sqrt(variance)
        result.append(scale if scale > 1e-12 else 1.0)
    return result


def fit_distance_scales(rows: list[dict[str, Any]]) -> dict[int, dict[str, list[float]]]:
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    result = {}
    for task_id, task_rows in by_task.items():
        state_by_context = {}
        for row in task_rows:
            state_by_context.setdefault(str(row["context_id"]), list(row["state"]))
        result[task_id] = {
            "state": _scales(list(state_by_context.values())),
            "delta_command": _scales(
                [list(row["delta_command"]) for row in task_rows]
            ),
            "executed_command": _scales(
                [list(row["executed_command"]) for row in task_rows]
            ),
        }
    return result


def _block_distance(
    left: list[float], right: list[float], scales: list[float]
) -> float:
    if len(left) != len(right) or len(left) != len(scales) or not left:
        raise ValueError("distance feature blocks have inconsistent dimensions")
    return sum(
        ((left_value - right_value) / scale) ** 2
        for left_value, right_value, scale in zip(left, right, scales, strict=True)
    ) / len(left)


def interface_distance(
    left: dict[str, Any],
    right: dict[str, Any],
    scales: dict[int, dict[str, list[float]]],
    mode: str,
) -> float:
    task_id = int(left["task_id"])
    if task_id != int(right["task_id"]):
        raise ValueError("task-specific simulator states cannot be compared across tasks")
    task_scales = scales[task_id]
    if mode == "delta_command":
        return _block_distance(
            left["delta_command"], right["delta_command"], task_scales[mode]
        )
    if mode == "executed_command":
        return _block_distance(
            left["executed_command"], right["executed_command"], task_scales[mode]
        )
    state_distance = _block_distance(
        left["state"], right["state"], task_scales["state"]
    )
    if mode == "state":
        return state_distance
    if mode == "state_and_executed_command":
        command_distance = _block_distance(
            left["executed_command"],
            right["executed_command"],
            task_scales["executed_command"],
        )
        return 0.5 * (state_distance + command_distance)
    raise ValueError(f"unknown interface distance mode: {mode}")


def nearest_context_probabilities(
    training: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    *,
    mode: str,
    neighbors: int,
    leave_target_trajectory_out: bool = False,
) -> list[float]:
    if mode not in DISTANCE_MODES:
        raise ValueError(f"unknown interface distance mode: {mode}")
    if neighbors <= 0:
        raise ValueError("neighbor count must be positive")
    predictions = []
    for target in targets:
        target_key = (int(target["task_id"]), int(target["episode_index"]))
        eligible = [
            row
            for row in training
            if int(row["task_id"]) == int(target["task_id"])
            and (
                not leave_target_trajectory_out
                or (int(row["task_id"]), int(row["episode_index"])) != target_key
            )
        ]
        if not eligible:
            raise ValueError(f"no training branches for task {target['task_id']}")
        scales = fit_distance_scales(eligible)
        # A restored state may have several commands. Keep only its closest
        # command so contexts with more exact-command groups receive no extra vote.
        by_context: dict[str, tuple[float, dict[str, Any]]] = {}
        for row in eligible:
            distance = interface_distance(target, row, scales, mode)
            key = str(row["context_id"])
            candidate = (distance, row)
            previous = by_context.get(key)
            if previous is None or (distance, str(row["physical_run"])) < (
                previous[0],
                str(previous[1]["physical_run"]),
            ):
                by_context[key] = candidate
        ordered = sorted(
            by_context.values(), key=lambda value: (value[0], str(value[1]["physical_run"]))
        )
        selected = ordered[: min(neighbors, len(ordered))]
        predictions.append(
            sum(float(row["task_failure"]) for _distance, row in selected)
            / len(selected)
        )
    return predictions

