from collections import Counter
from typing import Any

from embodied_silent_failures.evidence_graph.record import (
    GRAPH_EVENT_KINDS,
    TRACE_EVENT_KINDS,
)


VALID_BASIS_PREFIXES = ("paper:", "code:", "protocol:", "observed:")
TENSOR_ORIGIN_OPERATIONS = {
    "aten.lift_fresh.default": {
        "reason": "PyTorch materialized a constructor tensor before dispatch capture",
        "basis": "observed:torch-dispatch:aten.lift_fresh.default-tensor-origin",
    }
}


def audit_graph(
    events: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    graph: dict[str, Any],
    required_endpoints: tuple[str, ...],
    repeated_endpoints: tuple[str, ...] = (),
    contract_issues: list[str] | None = None,
) -> dict[str, Any]:
    graph_events = [
        event
        for event in events
        if event["kind"] in GRAPH_EVENT_KINDS
    ]
    trace_events = [event for event in events if event["kind"] in TRACE_EVENT_KINDS]
    event_by_id = {event["event_id"]: event for event in graph_events}
    event_ids = set(event_by_id)
    trace_event_by_id = {event["event_id"]: event for event in trace_events}
    trace_event_ids = set(trace_event_by_id)
    names = Counter(event["name"] for event in graph_events)
    annotation_counts = Counter(item.get("event_id") for item in annotations)
    annotations_by_id = {item.get("event_id"): item for item in annotations}

    scope_issues = []
    if not events or events[0].get("kind") != "trace_start":
        scope_issues.append("trace does not start with trace_start")
    if not events or events[-1].get("kind") != "trace_end":
        scope_issues.append("trace does not end with trace_end")
    elif events[-1].get("details", {}).get("completed") is not True:
        scope_issues.append("trace did not complete")
    for endpoint in required_endpoints:
        if names[endpoint] != 1:
            scope_issues.append(
                f"required endpoint {endpoint} appears {names[endpoint]} times"
            )
    for endpoint in repeated_endpoints:
        if names[endpoint] < 1:
            scope_issues.append(f"repeated endpoint {endpoint} does not appear")
    unresolved = []
    for event in trace_events:
        annotation = annotations_by_id.get(event["event_id"])
        if annotation is None or not annotation.get("region"):
            unresolved.append(
                {"event_id": event["event_id"], "name": event["name"]}
            )
    opaque_without_reason = [
        event["event_id"]
        for event in trace_events
        if event["kind"] == "opaque"
        and not event.get("details", {}).get("opaque_reason")
    ]
    operators_without_schema = [
        event["event_id"]
        for event in trace_events
        if event["kind"] == "operator"
        and event.get("details", {}).get("operator_semantics", {}).get(
            "schema_status"
        )
        != "available"
    ]
    unproduced_tensor_inputs, tensor_origins = _classify_unproduced_tensor_inputs(
        trace_events
    )

    reduction_issues = []
    raw_to_region = graph.get("raw_to_region", {})
    missing_mappings = sorted(event_ids - set(raw_to_region))
    extra_mappings = sorted(set(raw_to_region) - event_ids)
    if missing_mappings:
        reduction_issues.append(f"unmapped raw events: {missing_mappings}")
    if extra_mappings:
        reduction_issues.append(f"unknown mapped events: {extra_mappings}")
    regions = {region["region_id"]: region for region in graph.get("regions", [])}
    sinkless_regions = [
        {
            "region_id": region["region_id"],
            "name": region["name"],
            "event_count": region["event_count"],
            "basis": region.get("basis", []),
            "disposition": region.get("disposition"),
        }
        for region in regions.values()
        if not region.get("reachable_sinks")
    ]
    unjustified_sinkless = [
        region["region_id"]
        for region in sinkless_regions
        if not region.get("disposition")
    ]
    if unjustified_sinkless:
        reduction_issues.append(
            f"sinkless regions have no disposition: {unjustified_sinkless}"
        )
    for event_id, region_id in raw_to_region.items():
        region = regions.get(region_id)
        if region is None:
            reduction_issues.append(
                f"raw event {event_id} maps to missing region {region_id}"
            )
            continue
        raw_signature = graph.get("raw_reachability", {}).get(event_id)
        if raw_signature != region.get("reachable_sinks"):
            reduction_issues.append(
                f"region {region_id} changes sink reachability for {event_id}"
            )

    provenance_issues = []
    provenance_counts = Counter()
    duplicate_annotations = sorted(
        event_id
        for event_id, count in annotation_counts.items()
        if count > 1
    )
    if duplicate_annotations:
        provenance_issues.append(
            f"events have multiple annotations: {duplicate_annotations}"
        )
    for event_id, annotation in sorted(annotations_by_id.items()):
        if event_id not in trace_event_ids:
            provenance_issues.append(f"annotation refers to unknown event {event_id}")
            continue
        bases = annotation.get("basis")
        if not isinstance(bases, list) or not bases:
            provenance_issues.append(f"annotation {event_id} has no basis")
            continue
        invalid = [
            basis
            for basis in bases
            if not isinstance(basis, str) or not basis.startswith(VALID_BASIS_PREFIXES)
        ]
        if invalid:
            provenance_issues.append(
                f"annotation {event_id} has invalid bases: {invalid}"
            )
        provenance_counts[
            (
                annotation.get("region"),
                tuple(bases),
            )
        ] += 1

    sections = {
        "trace_integrity": {
            "passed": not scope_issues,
            "issues": scope_issues,
            "required_endpoints": list(required_endpoints),
            "repeated_endpoints": list(repeated_endpoints),
        },
        "annotation_coverage": {
            "passed": (
                not unresolved
                and not opaque_without_reason
                and not operators_without_schema
                and not unproduced_tensor_inputs
            ),
            "events": unresolved,
            "opaque_without_reason": opaque_without_reason,
            "operators_without_schema": operators_without_schema,
            "unproduced_tensor_inputs": unproduced_tensor_inputs,
            "tensor_origins": tensor_origins,
        },
        "reduction_integrity": {
            "passed": not reduction_issues,
            "issues": reduction_issues,
            "raw_event_count": len(trace_events),
            "graph_event_count": len(graph_events),
            "trace_event_count": len(trace_events),
            "mapped_event_count": len(raw_to_region),
            "sinkless_regions": sinkless_regions,
        },
        "model_contracts": {
            "passed": not (contract_issues or []),
            "issues": list(contract_issues or []),
        },
        "provenance_format": {
            "passed": not provenance_issues,
            "issues": provenance_issues,
            "annotation_count": len(annotations),
            "groups": [
                {
                    "region": region,
                    "basis": list(bases),
                    "event_count": count,
                }
                for (region, bases), count in sorted(
                    provenance_counts.items(),
                    key=lambda item: (
                        str(item[0][0]),
                        item[0][1],
                    ),
                )
            ],
        },
    }
    return {
        "schema_version": 1,
        "passed": all(section["passed"] for section in sections.values()),
        "trust_boundary": {
            "observed": "Runtime events, tensor identities, operator schemas, and dataflow are mechanically recorded.",
            "declared": "Adapters assign research-region names and boundary meanings with cited source or protocol provenance.",
            "not_established": "Passing checks do not validate adapter interpretation, physical hardware placement, or fault prevalence.",
        },
        "sections": sections,
    }


def _classify_unproduced_tensor_inputs(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    produced_values = set()
    produced_storage = set()
    unresolved = []
    origins = []
    for event in events:
        for reference in event.get("inputs", []):
            value_id = reference["value_id"]
            storage_id = reference.get("storage_id")
            value_type = reference.get("type", "")
            is_tensor = value_type == "torch.Tensor" or value_type.startswith(
                "torch.nn.parameter."
            )
            if (
                is_tensor
                and value_id not in produced_values
                and (storage_id is None or storage_id not in produced_storage)
            ):
                item = {
                    "event_id": event["event_id"],
                    "name": event["name"],
                    "port": reference["port"],
                    "value_id": value_id,
                }
                origin = TENSOR_ORIGIN_OPERATIONS.get(event["name"])
                if origin is None:
                    unresolved.append(item)
                else:
                    origins.append({**item, **origin})
        for reference in event.get("outputs", []):
            produced_values.add(reference["value_id"])
            storage_id = reference.get("storage_id")
            if storage_id is not None:
                produced_storage.add(storage_id)
    return unresolved, origins
