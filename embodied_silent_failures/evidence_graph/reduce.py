from collections import defaultdict, deque
from typing import Any

from embodied_silent_failures.evidence_graph.record import (
    GRAPH_EVENT_KINDS,
    TRACE_EVENT_KINDS,
)


def reduce_graph(
    events: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    trace_event_by_id = {
        event["event_id"]: event
        for event in events
        if event["kind"] in TRACE_EVENT_KINDS
    }
    event_by_id = {
        event_id: event
        for event_id, event in trace_event_by_id.items()
        if event["kind"] in GRAPH_EVENT_KINDS
    }
    trace_event_ids = set(trace_event_by_id)
    annotation_by_id = _annotation_index(
        annotations, event_by_id, trace_event_ids
    )
    trace_edges, storage_overlaps = _data_edges(trace_event_by_id)
    edges = _project_operator_paths(trace_event_by_id, set(event_by_id), trace_edges)

    roles = {
        event_id: annotation.get("role")
        for event_id, annotation in annotation_by_id.items()
    }
    sinks = sorted(event_id for event_id, role in roles.items() if role == "sink")
    reachability = _sink_reachability(event_by_id, edges, sinks)

    group_key = {}
    for event_id in sorted(event_by_id):
        annotation = annotation_by_id.get(event_id, {})
        disposition = annotation.get("disposition") or (
            "registered_state_not_observed_reaching_a_sink"
            if event_by_id[event_id]["kind"] == "state" and not reachability[event_id]
            else ""
        )
        group_key[event_id] = (
            annotation.get("region", "unresolved"),
            annotation.get("semantic_key", ""),
            annotation.get("lifetime", "step"),
            annotation.get("fault_interface") or "",
            disposition,
            tuple(reachability[event_id]),
        )

    grouped_events: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for event_id, key in group_key.items():
        grouped_events[key].append(event_id)
    groups = [
        (key, sorted(members)) for key, members in grouped_events.items()
    ]

    regions = []
    event_to_region = {}
    for index, (key, members) in enumerate(sorted(groups, key=lambda item: (item[0], item[1]))):
        (
            name,
            semantic_key,
            lifetime,
            fault_interface,
            disposition,
            reachable_sinks,
        ) = key
        region_id = f"r{index:04d}"
        for member in members:
            event_to_region[member] = region_id
        bases = sorted(
            {
                basis
                for member in members
                for basis in annotation_by_id.get(member, {}).get("basis", [])
            }
        )
        regions.append(
            {
                "region_id": region_id,
                "name": name,
                "semantic_key": semantic_key or name,
                "event_ids": members,
                "event_count": len(members),
                "lifetime": lifetime,
                "fault_interface": fault_interface or None,
                "disposition": disposition or None,
                "reachable_sinks": list(reachable_sinks),
                "basis": bases,
                "aggregation": (
                    "registered_state_with_same_mechanical_module_key"
                    if all(event_by_id[member]["kind"] == "state" for member in members)
                    else "same_declared_identity_and_sink_reachability"
                ),
            }
        )

    reduced_edges = set()
    for source, target, kind in edges:
        kind = _temporal_edge_kind(event_by_id, source, target, kind)
        source_region = event_to_region[source]
        target_region = event_to_region[target]
        if source_region != target_region:
            reduced_edges.add((source_region, target_region, kind))

    reduced_overlaps = set()
    for overlap in storage_overlaps:
        left = event_to_region.get(overlap["left"])
        right = event_to_region.get(overlap["right"])
        if left is None or right is None or left == right:
            continue
        reduced_overlaps.add(
            (
                min(left, right),
                max(left, right),
                overlap["storage_id"],
                overlap.get("byte_start"),
                overlap.get("byte_stop"),
            )
        )

    return {
        "schema_version": 1,
        "raw_event_count": len(trace_event_ids),
        "graph_event_count": len(event_by_id),
        "trace_event_count": len(trace_event_ids),
        "operator_event_count": sum(
            event["kind"] == "operator" for event in events
        ),
        "trace_edge_count": len(trace_edges),
        "storage_overlap_count": len(storage_overlaps),
        "sinks": [
            {
                "event_id": event_id,
                "name": event_by_id[event_id]["name"],
                "region_id": event_to_region[event_id],
            }
            for event_id in sinks
        ],
        "regions": regions,
        "edges": [
            {"source": source, "target": target, "kind": kind}
            for source, target, kind in sorted(reduced_edges)
        ],
        "raw_edges": [
            {
                "source": source,
                "target": target,
                "kind": _temporal_edge_kind(event_by_id, source, target, kind),
            }
            for source, target, kind in sorted(edges)
        ],
        "storage_overlaps": [
            {
                "left": left,
                "right": right,
                "storage_id": storage_id,
                **({"byte_start": start, "byte_stop": stop} if start is not None else {}),
            }
            for left, right, storage_id, start, stop in sorted(
                reduced_overlaps,
                key=lambda item: tuple("" if value is None else str(value) for value in item),
            )
        ],
        "raw_storage_overlaps": storage_overlaps,
        "raw_to_region": event_to_region,
        "raw_reachability": reachability,
    }


def raw_trace_edges(
    events: list[dict[str, Any]],
) -> set[tuple[str, str, str, str]]:
    """Return value-carrying trace edges with temporal relations classified."""
    trace_events = {
        event["event_id"]: event
        for event in events
        if event["kind"] in TRACE_EVENT_KINDS
    }
    edges, _overlaps = _data_edges(trace_events)
    return {
        (
            source,
            target,
            _temporal_edge_kind(trace_events, source, target, kind),
            value,
        )
        for source, target, kind, value in edges
    }


def _data_edges(
    event_by_id: dict[str, dict[str, Any]],
) -> tuple[set[tuple[str, str, str, str]], list[dict[str, Any]]]:
    edges = set()
    latest_producer: dict[str, str] = {}
    latest_mutations: dict[str, list[tuple[str, int | None, int | None]]] = defaultdict(list)
    overlaps = set()
    for event_id, event in event_by_id.items():
        for reference in event.get("inputs", []):
            value_id = reference["value_id"]
            source = latest_producer.get(value_id)
            if source is not None and source != event_id:
                edges.add((source, event_id, "dataflow", value_id))
            storage_id = reference.get("storage_id")
            if storage_id is None:
                continue
            start, stop = _byte_range(reference)
            for writer, writer_start, writer_stop in latest_mutations[storage_id]:
                if writer != event_id and _ranges_overlap(
                    start, stop, writer_start, writer_stop
                ):
                    edges.add((writer, event_id, "mutation", storage_id))
        semantics = event.get("details", {}).get("operator_semantics", {})
        for alias in semantics.get("declared_aliases", []):
            output_references = _references_at_port(
                event.get("outputs", []), alias["output"]
            )
            input_references = _references_at_port(
                event.get("inputs", []), alias["input"]
            )
            for input_reference in input_references:
                producer = latest_producer.get(input_reference["value_id"])
                if producer is None or producer == event_id:
                    continue
                for output_reference in output_references:
                    storage_id = input_reference.get("storage_id")
                    if storage_id is None or storage_id != output_reference.get("storage_id"):
                        continue
                    intersection = _range_intersection(
                        *_byte_range(input_reference),
                        *_byte_range(output_reference),
                    )
                    if _ranges_overlap(
                        *_byte_range(input_reference),
                        *_byte_range(output_reference),
                    ):
                        overlaps.add((producer, event_id, storage_id, *intersection))
        mutated_ports = semantics.get("mutated_input_ports", [])
        for reference in event.get("inputs", []):
            if not any(_port_is_within(reference["port"], port) for port in mutated_ports):
                continue
            storage_id = reference.get("storage_id")
            if storage_id is None:
                continue
            start, stop = _byte_range(reference)
            latest_mutations[storage_id] = [
                item
                for item in latest_mutations[storage_id]
                if not _ranges_overlap(start, stop, item[1], item[2])
            ]
            latest_mutations[storage_id].append((event_id, start, stop))
        for reference in event.get("outputs", []):
            latest_producer[reference["value_id"]] = event_id

    storage_overlaps = [
        {
            "left": left,
            "right": right,
            "storage_id": storage_id,
            **({"byte_start": start, "byte_stop": stop} if start is not None else {}),
        }
        for left, right, storage_id, start, stop in sorted(
            overlaps,
            key=lambda item: tuple("" if value is None else str(value) for value in item),
        )
    ]
    return edges, storage_overlaps


def _project_operator_paths(
    trace_event_by_id: dict[str, dict[str, Any]],
    graph_event_ids: set[str],
    trace_edges: set[tuple[str, str, str, str]],
) -> set[tuple[str, str, str]]:
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, target, kind, _value in trace_edges:
        incoming[target].append((source, kind))

    frontier: dict[str, set[tuple[str, str]]] = {}
    projected = set()
    for event_id in trace_event_by_id:
        inherited = set()
        for source, kind in incoming[event_id]:
            if source in graph_event_ids:
                inherited.add((source, kind))
                continue
            for ancestor, prior_kind in frontier.get(source, set()):
                inherited.add((ancestor, _path_kind(prior_kind, kind)))
        if event_id in graph_event_ids:
            projected.update(
                (source, event_id, kind)
                for source, kind in inherited
                if source != event_id
            )
            frontier[event_id] = {(event_id, "dataflow")}
        else:
            frontier[event_id] = inherited
    return projected


def _path_kind(left: str, right: str) -> str:
    return "mutation" if "mutation" in (left, right) else "dataflow"


def _temporal_edge_kind(
    events: dict[str, dict[str, Any]], source: str, target: str, kind: str
) -> str:
    relation = events[source].get("details", {}).get("temporal_relation")
    if relation == "world_feedback" and events[target]["name"] not in {
        "libero.next_observation",
        "libero.current_observation",
    }:
        return kind
    return f"temporal_{relation}" if relation else kind


def _byte_range(reference: dict[str, Any]) -> tuple[int | None, int | None]:
    return reference.get("storage_byte_start"), reference.get("storage_byte_stop")


def _ranges_overlap(
    left_start: int | None,
    left_stop: int | None,
    right_start: int | None,
    right_stop: int | None,
) -> bool:
    if None in (left_start, left_stop, right_start, right_stop):
        return True
    return bool(left_start < right_stop and right_start < left_stop)


def _range_intersection(
    left_start: int | None,
    left_stop: int | None,
    right_start: int | None,
    right_stop: int | None,
) -> tuple[int | None, int | None]:
    if None in (left_start, left_stop, right_start, right_stop):
        return None, None
    return max(left_start, right_start), min(left_stop, right_stop)


def _port_is_within(port: str, root: str) -> bool:
    return port == root or port.startswith(f"{root}.") or port.startswith(f"{root}[")


def _references_at_port(
    references: list[dict[str, Any]], root: str
) -> list[dict[str, Any]]:
    return [reference for reference in references if _port_is_within(reference["port"], root)]


def _annotation_index(
    annotations: list[dict[str, Any]],
    event_by_id: dict[str, dict[str, Any]],
    trace_event_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    seen = set()
    for annotation in annotations:
        event_id = annotation.get("event_id")
        if event_id not in trace_event_ids:
            raise ValueError(f"annotation refers to unknown graph event: {event_id}")
        if event_id in seen:
            raise ValueError(f"graph event has multiple annotations: {event_id}")
        seen.add(event_id)
        if event_id not in event_by_id:
            continue
        result[event_id] = annotation
    return result


def _sink_reachability(
    event_by_id: dict[str, dict[str, Any]],
    edges: set[tuple[str, str, str]],
    sinks: list[str],
) -> dict[str, list[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, target, _kind in edges:
        reverse[target].add(source)
    reachable = {event_id: set() for event_id in event_by_id}
    for sink in sinks:
        queue = deque([sink])
        visited = {sink}
        while queue:
            current = queue.popleft()
            reachable[current].add(sink)
            for previous in reverse[current]:
                if previous not in visited:
                    visited.add(previous)
                    queue.append(previous)
    return {event_id: sorted(values) for event_id, values in reachable.items()}
