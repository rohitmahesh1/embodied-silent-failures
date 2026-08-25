from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256


# OpenVLA 300dce2 produces a decoded action in predict_action; the pinned
# LIBERO runner then normalizes its gripper and passes the resulting command to
# env.step at the adapter's simulator_command boundary.
DEFAULT_ACTION_INTERFACE = "simulator_command"

# SAFE b6036ab, failure_prob.data.openvla.load_rollouts and
# failure_prob.data.utils.process_tensor_idx_rel select the final action-token
# feature that enters SAFE at the adapter's safe_feature boundary.
DEFAULT_MONITOR_INTERFACE = "safe_feature"

_DEPTH_PATTERN = re.compile(
    r"^(?P<prefix>.*\.(?:layers|blocks))\.(?P<index>\d+)\.(?P<suffix>.+)$"
)


def _region_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    regions = graph.get("regions")
    edges = graph.get("edges")
    if not isinstance(regions, list) or not isinstance(edges, list):
        raise ValueError("graph must contain region and edge lists")
    by_id = {}
    for region in regions:
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or region_id in by_id:
            raise ValueError("graph contains a missing or duplicate region ID")
        by_id[region_id] = region
    for edge in edges:
        if edge.get("source") not in by_id or edge.get("target") not in by_id:
            raise ValueError("graph edge refers to an unknown region")
    return by_id


def _anchors(
    regions: dict[str, dict[str, Any]], interface: str
) -> list[str]:
    values = sorted(
        region_id
        for region_id, region in regions.items()
        if region.get("fault_interface") == interface
        and not region.get("disposition")
    )
    if not values:
        raise ValueError(f"graph has no undisposed {interface} anchor")
    return values


def _ancestors(
    graph: dict[str, Any], targets: list[str]
) -> set[str]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for edge in graph["edges"]:
        kind = str(edge.get("kind", ""))
        if kind.startswith("temporal_"):
            continue
        reverse[edge["target"]].add(edge["source"])
    visited = set(targets)
    queue = deque(targets)
    while queue:
        current = queue.popleft()
        for source in reverse[current]:
            if source not in visited:
                visited.add(source)
                queue.append(source)
    return visited


def _topology(action: bool, monitor: bool) -> str:
    if action and monitor:
        return "shared_action_and_monitor_evidence"
    if action:
        return "action_only"
    if monitor:
        return "monitor_evidence_only"
    return "neither_same_decision_path"


def _fault_population(region: dict[str, Any]) -> str:
    if region.get("disposition"):
        return "excluded_by_declared_disposition"
    interface = region.get("fault_interface")
    if interface == "registered_model_state":
        return "persistent_model_state"
    if interface:
        return "declared_runtime_boundary"
    return "observed_computation_without_declared_fault_interface"


def _module_parts(
    region: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    semantic_key = str(region.get("semantic_key", ""))
    marker = "/state/"
    if marker not in semantic_key:
        return None, None, None
    module_path = semantic_key.split(marker, 1)[1]
    match = _DEPTH_PATTERN.match(module_path)
    if match is None:
        return module_path, module_path.rsplit(".", 1)[-1], None
    return (
        module_path,
        match.group("suffix").rsplit(".", 1)[-1],
        {
            "family": match.group("prefix"),
            "index": int(match.group("index")),
        },
    )


def _depth_denominators(
    regions: dict[str, dict[str, Any]]
) -> dict[str, int]:
    maxima: dict[str, int] = {}
    for region in regions.values():
        _path, _role, depth = _module_parts(region)
        if depth is not None:
            family = depth["family"]
            maxima[family] = max(maxima.get(family, 0), depth["index"])
    return maxima


def _tensor_schema(reference: dict[str, Any]) -> tuple[Any, ...]:
    shape = reference.get("shape")
    if isinstance(shape, list):
        shape = tuple(shape)
    return (
        reference.get("port"),
        reference.get("type"),
        shape,
        reference.get("dtype"),
        reference.get("device"),
    )


def _schemas(
    region: dict[str, Any], events: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    inputs = set()
    outputs = set()
    for event_id in region.get("event_ids", []):
        event = events.get(event_id)
        if event is None:
            raise ValueError(f"region refers to an absent raw event: {event_id}")
        inputs.update(_tensor_schema(value) for value in event.get("inputs", []))
        outputs.update(_tensor_schema(value) for value in event.get("outputs", []))

    def records(values: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
        result = []
        for port, value_type, shape, dtype, device in sorted(
            values, key=lambda item: tuple(str(value) for value in item)
        ):
            result.append(
                {
                    "port": port,
                    "type": value_type,
                    **({"shape": list(shape)} if isinstance(shape, tuple) else {}),
                    **({"shape_description": shape} if isinstance(shape, str) else {}),
                    **({"dtype": dtype} if dtype is not None else {}),
                    **({"device": device} if device is not None else {}),
                }
            )
        return result

    return {"inputs": records(inputs), "outputs": records(outputs)}


def build_site_census(
    graph: dict[str, Any],
    raw_events: list[dict[str, Any]],
    *,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    monitor_interface: str = DEFAULT_MONITOR_INTERFACE,
) -> dict[str, Any]:
    regions = _region_index(graph)
    events = {
        event["event_id"]: event
        for event in raw_events
        if isinstance(event.get("event_id"), str)
    }
    action_anchors = _anchors(regions, action_interface)
    monitor_anchors = _anchors(regions, monitor_interface)
    action_ancestors = _ancestors(graph, action_anchors)
    monitor_ancestors = _ancestors(graph, monitor_anchors)
    depth_maxima = _depth_denominators(regions)
    sink_names = {
        sink["event_id"]: sink["name"] for sink in graph.get("sinks", [])
    }

    sites = []
    for region_id, region in sorted(regions.items()):
        module_path, module_role, depth = _module_parts(region)
        if depth is not None:
            maximum = depth_maxima[depth["family"]]
            depth = {
                **depth,
                "observed_maximum_index": maximum,
                "normalized": depth["index"] / maximum if maximum else 0.0,
            }
        reachable_sinks = [
            sink_names.get(event_id, event_id)
            for event_id in region.get("reachable_sinks", [])
        ]
        action = region_id in action_ancestors
        monitor = region_id in monitor_ancestors
        population = _fault_population(region)
        sites.append(
            {
                "site_id": region_id,
                "region": region.get("name"),
                "semantic_key": region.get("semantic_key"),
                "fault_interface": region.get("fault_interface"),
                "fault_population": population,
                "declared_injection_boundary": population
                in {"persistent_model_state", "declared_runtime_boundary"},
                "topology": _topology(action, monitor),
                "same_decision_reachability": {
                    "action_sink": action,
                    "monitor_evidence_sink": monitor,
                },
                "eventual_reachability": {
                    "monitor_timeline": "rollout.monitor_timeline" in reachable_sinks,
                    "task_outcome": "rollout.outcome" in reachable_sinks,
                },
                "architecture": {
                    "declared_owner": region.get("name"),
                    "module_path": module_path,
                    "literal_module_role": module_role,
                    "depth": depth,
                },
                "observed_value_schema": _schemas(region, events),
                "event_count": int(region.get("event_count", 0)),
                "basis": sorted(region.get("basis", [])),
                "disposition": region.get("disposition"),
            }
        )

    counts: dict[str, dict[str, int]] = {}
    for field in ("fault_population", "topology", "region"):
        values: dict[str, int] = defaultdict(int)
        for site in sites:
            values[str(site[field])] += 1
        counts[field] = dict(sorted(values.items()))
    return {
        "schema_version": 1,
        "sampling_frame": "one row per mechanically reduced evidence-graph region",
        "anchors": {
            "action": {
                "fault_interface": action_interface,
                "region_ids": action_anchors,
            },
            "monitor_evidence": {
                "fault_interface": monitor_interface,
                "region_ids": monitor_anchors,
            },
        },
        "edge_scope": "all non-temporal reduced edges",
        "trust_boundary": {
            "observed": (
                "Region membership, reduced dataflow, raw tensor schemas, and "
                "adapter provenance come from the audited trace artifacts."
            ),
            "derived": (
                "Topology is reverse reachability to the declared action-command "
                "and SAFE-feature anchors after removing temporal edges. Depth is "
                "the literal layer or block index divided by the largest observed "
                "index in the same module-path family."
            ),
            "not_established": (
                "The census does not establish site importance, tensor-value drift, "
                "physical hardware sharing, fault prevalence, or causal risk."
            ),
        },
        "counts": {"sites": len(sites), **counts},
        "sites": sites,
    }


def source_artifacts(
    graph_path: Path, raw_path: Path, audit_path: Path
) -> dict[str, Any]:
    builder_path = Path(__file__).resolve()
    return {
        "graph": {"path": str(graph_path.resolve()), "sha256": file_sha256(graph_path)},
        "raw_trace": {"path": str(raw_path.resolve()), "sha256": file_sha256(raw_path)},
        "audit": {
            "path": str(audit_path.resolve()),
            "sha256": file_sha256(audit_path),
            "passed": True,
        },
        "builder": {"path": str(builder_path), "sha256": file_sha256(builder_path)},
    }
