from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from typing import Any

from embodied_silent_failures.evidence_graph.census import (
    DEFAULT_ACTION_INTERFACE,
    DEFAULT_MONITOR_INTERFACE,
)
from embodied_silent_failures.evidence_graph.reduce import raw_trace_edges


# torch_trace.capture_torch_operations records these fields directly from the
# pinned model's named-module hooks. They identify execution points without a
# handwritten list of OpenVLA layers.
_MODULE_IDENTITY_FIELDS = ("module_path", "module_call_index")
_DEPTH_PATTERN = re.compile(
    r"^(?P<prefix>.*\.(?:layers|blocks))\.(?P<index>\d+)(?:\.|$)"
)
_NUMERIC_TYPES = {
    "numpy.ndarray",
    "torch.Tensor",
}


def _stable_id(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "ts-" + hashlib.sha256(encoded).hexdigest()[:16]


def architecture(identity: dict[str, Any]) -> dict[str, Any]:
    module_path = identity.get("module_path")
    if not isinstance(module_path, str):
        return {"module_path": None, "literal_module_role": None, "depth": None}
    match = _DEPTH_PATTERN.match(module_path)
    depth = None
    if match is not None:
        depth = {
            "family": match.group("prefix"),
            "index": int(match.group("index")),
        }
    return {
        "module_path": module_path,
        "literal_module_role": module_path.rsplit(".", 1)[-1],
        "depth": depth,
    }


def _regions(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = graph.get("regions")
    if not isinstance(values, list):
        raise ValueError("evidence graph has no region list")
    result = {}
    for region in values:
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or region_id in result:
            raise ValueError("evidence graph has a missing or duplicate region ID")
        result[region_id] = region
    return result


def _anchor_events(
    regions: dict[str, dict[str, Any]], interface: str
) -> set[str]:
    anchors = {
        event_id
        for region in regions.values()
        if region.get("fault_interface") == interface
        and not region.get("disposition")
        for event_id in region.get("event_ids", [])
    }
    if not anchors:
        raise ValueError(f"evidence graph has no undisposed {interface} anchor")
    return anchors


def _ancestors(
    edges: set[tuple[str, str, str, str]], targets: set[str]
) -> set[str]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, target, kind, _value in edges:
        if not kind.startswith("temporal_"):
            reverse[target].add(source)
    visited = set(targets)
    queue = deque(targets)
    while queue:
        current = queue.popleft()
        for source in reverse[current]:
            if source not in visited:
                visited.add(source)
                queue.append(source)
    return visited


def _reference_reaches(
    event_id: str,
    reference: dict[str, Any],
    outgoing: dict[str, list[tuple[str, str, str]]],
    ancestors: set[str],
    anchors: set[str],
    *,
    anchor_output: bool,
) -> bool:
    if anchor_output and event_id in anchors:
        return True
    identities = {reference.get("value_id"), reference.get("storage_id")}
    identities.discard(None)
    return any(
        not kind.startswith("temporal_")
        and value in identities
        and target in ancestors
        for target, kind, value in outgoing.get(event_id, [])
    )


def _schema(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        key: reference[key]
        for key in ("type", "shape", "dtype", "device")
        if key in reference
    }


def _is_numeric(reference: dict[str, Any]) -> bool:
    value_type = str(reference.get("type", ""))
    dtype = str(reference.get("dtype", "")).lower()
    if value_type not in _NUMERIC_TYPES or not dtype:
        return False
    return any(
        token in dtype
        for token in ("float", "bfloat", "int", "uint", "bool", "complex")
    )


def _value_family(reference: dict[str, Any]) -> str:
    dtype = str(reference.get("dtype", "")).lower()
    shape = reference.get("shape")
    if str(reference.get("type")) == "numpy.ndarray" and dtype == "uint8":
        if isinstance(shape, list) and len(shape) == 3:
            return "image_array"
    if "float" in dtype or "bfloat" in dtype or "complex" in dtype:
        return "continuous_tensor"
    return "discrete_tensor"


def _topology(action: bool, monitor: bool) -> str:
    if action and monitor:
        return "shared_action_and_monitor_evidence"
    if action:
        return "action_only"
    if monitor:
        return "monitor_evidence_only"
    return "neither_same_decision_path"


def _site_identity(
    event: dict[str, Any], reference: dict[str, Any], boundary_index: int
) -> dict[str, Any] | None:
    if event["kind"] == "module":
        details = event.get("details", {})
        if any(details.get(field) is None for field in _MODULE_IDENTITY_FIELDS):
            return None
        return {
            "kind": "module_output",
            "module_path": str(details["module_path"]),
            "module_call_index": int(details["module_call_index"]),
            "output_port": str(reference["port"]),
        }
    return {
        "kind": "declared_runtime_boundary",
        "event_name": str(event["name"]),
        "event_call_index": boundary_index,
        "output_port": str(reference["port"]),
    }


def _source_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events or events[0].get("kind") != "trace_start":
        raise ValueError("raw trace does not start with trace_start")
    if events[-1].get("kind") != "trace_end":
        raise ValueError("raw trace does not end with trace_end")
    if events[-1].get("details", {}).get("completed") is not True:
        raise ValueError("raw trace did not complete")
    details = events[0].get("details", {})
    return {
        key: details[key]
        for key in (
            "condition",
            "episode_index",
            "task_id",
            "task_suite",
            "traced_steps",
            "trial_seed",
            "upstream_revisions",
        )
        if key in details
    }


def observe_temporal_source(
    source: dict[str, Any],
    *,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    monitor_interface: str = DEFAULT_MONITOR_INTERFACE,
) -> dict[str, Any]:
    source_id = source.get("source_id")
    graph = source.get("graph")
    events = source.get("events")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("temporal-site source requires a nonempty source_id")
    if not isinstance(graph, dict) or not isinstance(events, list):
        raise ValueError(f"temporal-site source {source_id} is incomplete")

    regions = _regions(graph)
    event_to_region = graph.get("raw_to_region")
    if not isinstance(event_to_region, dict):
        raise ValueError("evidence graph has no raw-event mapping")
    action_anchors = _anchor_events(regions, action_interface)
    monitor_anchors = _anchor_events(regions, monitor_interface)
    edges = raw_trace_edges(events)
    action_ancestors = _ancestors(edges, action_anchors)
    monitor_ancestors = _ancestors(edges, monitor_anchors)
    outgoing: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for left, right, kind, value in edges:
        outgoing[left].append((right, kind, value))

    observations = []
    exclusions = Counter()
    boundary_calls: dict[tuple[int, str, str], int] = defaultdict(int)
    sites_by_value: dict[tuple[int, str], list[str]] = defaultdict(list)
    for event in events:
        kind = event.get("kind")
        if kind in {"operator", "state"}:
            exclusions[f"population_not_selected:{kind}"] += 1
            continue
        if kind not in {"boundary", "module", "opaque", "source"}:
            continue
        context = event.get("context", {})
        policy_step = context.get("policy_step")
        if type(policy_step) is not int or policy_step < 0:
            exclusions["event_has_no_policy_step"] += 1
            continue
        region_id = event_to_region.get(event["event_id"])
        region = regions.get(region_id)
        if region is None:
            raise ValueError(f"raw event {event['event_id']} has no reduced region")
        interface = region.get("fault_interface")
        if kind != "module" and not interface:
            exclusions["runtime_event_has_no_declared_boundary"] += 1
            continue
        if interface == "registered_model_state":
            exclusions["persistent_state_not_temporal_runtime_value"] += 1
            continue
        # OpenVLA 300dce2 and LIBERO 8f1084e pass the command into env.step;
        # the simulator-command event's outputs are the following observation,
        # reward, and done flag rather than values at the command interface.
        if kind != "module" and interface == action_interface:
            exclusions["action_anchor_is_an_input_interface"] += len(
                event.get("outputs", [])
            )
            continue

        boundary_key = (policy_step, str(kind), str(event["name"]))
        boundary_index = boundary_calls[boundary_key]
        boundary_calls[boundary_key] += 1
        for reference in event.get("outputs", []):
            if not _is_numeric(reference):
                exclusions["output_is_not_numeric_tensor_or_array"] += 1
                continue
            identity = _site_identity(event, reference, boundary_index)
            if identity is None:
                exclusions["module_identity_is_incomplete"] += 1
                continue
            site_id = _stable_id(identity)
            action = _reference_reaches(
                event["event_id"],
                reference,
                outgoing,
                action_ancestors,
                action_anchors,
                anchor_output=False,
            )
            monitor = _reference_reaches(
                event["event_id"],
                reference,
                outgoing,
                monitor_ancestors,
                monitor_anchors,
                anchor_output=True,
            )
            observation = {
                "site_id": site_id,
                "identity": identity,
                "source_id": source_id,
                "policy_step": policy_step,
                "schema": _schema(reference),
                "value_family": _value_family(reference),
                "fault_interface": interface,
                "reduced_region_id": region_id,
                "declared_owner": region.get("name"),
                "basis": sorted(region.get("basis", [])),
                "disposition": region.get("disposition"),
                "action_reachable": action,
                "monitor_reachable": monitor,
                "topology": _topology(action, monitor),
                "value_id": reference.get("value_id"),
                "storage_id": reference.get("storage_id"),
            }
            observations.append(observation)
            value_id = reference.get("value_id")
            if isinstance(value_id, str):
                sites_by_value[(policy_step, value_id)].append(site_id)

    aliases: dict[tuple[str, int], set[str]] = defaultdict(set)
    for (policy_step, _value_id), site_ids in sites_by_value.items():
        unique = sorted(set(site_ids))
        if len(unique) < 2:
            continue
        for site_id in unique:
            aliases[(site_id, policy_step)].update(
                other for other in unique if other != site_id
            )
    for observation in observations:
        observation["same_value_alias_site_ids"] = sorted(
            aliases.get((observation["site_id"], observation["policy_step"]), set())
        )

    traced_steps = sorted(
        {
            observation["policy_step"]
            for observation in observations
            if observation["identity"]["kind"] == "module_output"
        }
    )
    return {
        "source_id": source_id,
        "metadata": _source_metadata(events),
        "actual_module_trace_steps": traced_steps,
        "raw_event_count": len(events),
        "observations": observations,
        "excluded_event_counts": dict(sorted(exclusions.items())),
        **({"artifacts": source["artifacts"]} if "artifacts" in source else {}),
    }
