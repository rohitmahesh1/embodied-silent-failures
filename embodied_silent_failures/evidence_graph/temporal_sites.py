from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from embodied_silent_failures.evidence_graph.census import (
    DEFAULT_ACTION_INTERFACE,
    DEFAULT_MONITOR_INTERFACE,
)
from embodied_silent_failures.evidence_graph.temporal_trace import (
    architecture as observed_architecture,
    observe_temporal_source,
)


def _site_row(site_id: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    identity = observations[0]["identity"]
    if any(observation["identity"] != identity for observation in observations):
        raise ValueError(f"stable site ID collision at {site_id}")

    by_source: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for observation in observations:
        steps = by_source[observation["source_id"]]
        policy_step = observation["policy_step"]
        if policy_step in steps:
            raise ValueError(
                f"site {site_id} occurs twice at policy step {policy_step} "
                f"in {observation['source_id']}"
            )
        steps[policy_step] = observation

    opportunities = []
    mismatches = []
    consecutive_pairs = 0
    for source_id, steps in sorted(by_source.items()):
        for policy_step, current in sorted(steps.items()):
            prior = steps.get(policy_step - 1)
            if prior is None:
                continue
            consecutive_pairs += 1
            if prior["schema"] != current["schema"]:
                mismatches.append(
                    {
                        "source_id": source_id,
                        "prior_policy_step": policy_step - 1,
                        "policy_step": policy_step,
                        "prior_schema": prior["schema"],
                        "current_schema": current["schema"],
                    }
                )
                continue
            if current["disposition"]:
                continue
            if not current["action_reachable"]:
                continue
            opportunities.append(
                {
                    "source_id": source_id,
                    "prior_policy_step": policy_step - 1,
                    "policy_step": policy_step,
                    "topology": current["topology"],
                }
            )

    reasons = []
    if all(observation["disposition"] for observation in observations):
        reasons.append("all_observations_have_a_declared_disposition")
    if not any(observation["action_reachable"] for observation in observations):
        reasons.append("output_port_does_not_reach_the_executed_command")
    if not opportunities:
        if consecutive_pairs == 0:
            reasons.append("no_consecutive_policy_steps_were_traced_for_this_site")
        elif len(mismatches) == consecutive_pairs:
            reasons.append("all_consecutive_observations_have_different_schemas")
        else:
            reasons.append("no_consecutive_observation_passes_static_eligibility")

    if opportunities:
        status = "structurally_eligible_pending_canary"
    elif consecutive_pairs == 0 and any(
        observation["action_reachable"] and not observation["disposition"]
        for observation in observations
    ):
        status = "unresolved_without_consecutive_trace"
    else:
        status = "structurally_ineligible"

    schemas = {
        json.dumps(observation["schema"], sort_keys=True, separators=(",", ":"))
        for observation in observations
    }
    aliases = {
        alias
        for observation in observations
        for alias in observation["same_value_alias_site_ids"]
    }
    architecture = observed_architecture(identity)
    return {
        "site_id": site_id,
        "identity": identity,
        "status": status,
        "canary_status": "not_run",
        "eligibility_reasons": reasons,
        "hook_mechanism": (
            "torch_module_forward_hook"
            if identity["kind"] == "module_output"
            else "declared_adapter_boundary"
        ),
        "value_families": sorted(
            {observation["value_family"] for observation in observations}
        ),
        "schemas": [json.loads(value) for value in sorted(schemas)],
        "topologies": sorted(
            {observation["topology"] for observation in observations}
        ),
        "fault_interfaces": sorted(
            {
                str(observation["fault_interface"])
                for observation in observations
                if observation["fault_interface"] is not None
            }
        ),
        "architecture": {
            **architecture,
            "observed_owners": sorted(
                {
                    str(observation["declared_owner"])
                    for observation in observations
                    if observation["declared_owner"] is not None
                }
            ),
        },
        "basis": sorted(
            {
                basis
                for observation in observations
                for basis in observation["basis"]
            }
        ),
        "observed_source_count": len(by_source),
        "observation_count": len(observations),
        "consecutive_pair_count": consecutive_pairs,
        "matching_schema_pair_count": consecutive_pairs - len(mismatches),
        "eligible_opportunity_count": len(opportunities),
        "eligible_opportunities": opportunities,
        "schema_mismatches": mismatches,
        "same_value_alias_site_ids": sorted(aliases),
    }


def _normalize_depth(sites: list[dict[str, Any]]) -> None:
    maxima: dict[str, int] = {}
    for site in sites:
        depth = site["architecture"]["depth"]
        if depth is None:
            continue
        family = depth["family"]
        maxima[family] = max(maxima.get(family, 0), depth["index"])
    for site in sites:
        depth = site["architecture"]["depth"]
        if depth is None:
            continue
        maximum = maxima[depth["family"]]
        depth["observed_maximum_index"] = maximum
        depth["normalized"] = depth["index"] / maximum if maximum else 0.0


def build_temporal_site_table(
    sources: Iterable[dict[str, Any]],
    *,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    monitor_interface: str = DEFAULT_MONITOR_INTERFACE,
) -> dict[str, Any]:
    observed_sources = []
    observations_by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_ids = set()
    excluded = Counter()
    for source in sources:
        observed = observe_temporal_source(
            source,
            action_interface=action_interface,
            monitor_interface=monitor_interface,
        )
        source_id = observed["source_id"]
        if source_id in source_ids:
            raise ValueError(f"duplicate temporal-site source ID: {source_id}")
        source_ids.add(source_id)
        for observation in observed.pop("observations"):
            observations_by_site[observation["site_id"]].append(observation)
        excluded.update(observed["excluded_event_counts"])
        observed_sources.append(observed)
    if not observed_sources:
        raise ValueError("temporal-site table requires at least one source")

    sites = [
        _site_row(site_id, observations_by_site[site_id])
        for site_id in sorted(observations_by_site)
    ]
    _normalize_depth(sites)
    status_counts = Counter(site["status"] for site in sites)
    topology_counts = Counter(
        topology for site in sites for topology in site["topologies"]
    )
    return {
        "schema_version": 1,
        "sampling_frame": (
            "one row per observed numeric module-output port or declared runtime-"
            "boundary port; one opportunity is the same site observed at consecutive "
            "policy decisions with an identical schema"
        ),
        "fault_operator": "replace x_t with x_(t-1) at one output port",
        "anchors": {
            "action": action_interface,
            "monitor_evidence": monitor_interface,
        },
        "trust_boundary": {
            "observed": (
                "Module paths, call indices, tensor ports and schemas, aliases, and "
                "value-carrying dataflow come from passing runtime trace artifacts."
            ),
            "declared": (
                "The primary granularity includes all numeric module-output ports and "
                "provenance-backed runtime boundaries, while primitive operators and "
                "persistent model state remain separate fault populations."
            ),
            "derived": (
                "A temporal opportunity requires observations at t-1 and t with an "
                "identical schema and a non-temporal path from the current output port "
                "to the executed simulator command."
            ),
            "not_established": (
                "Structural eligibility does not establish hook equivalence, numerical "
                "difference, fault consequence, physical prevalence, or experimental "
                "injectability; current-value canaries remain required."
            ),
        },
        "counts": {
            "sources": len(observed_sources),
            "sites": len(sites),
            "eligible_opportunities": sum(
                site["eligible_opportunity_count"] for site in sites
            ),
            "status": dict(sorted(status_counts.items())),
            "topology": dict(sorted(topology_counts.items())),
            "excluded_events": dict(sorted(excluded.items())),
        },
        "sources": observed_sources,
        "sites": sites,
    }


def csv_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for site in table["sites"]:
        identity = site["identity"]
        architecture = site["architecture"]
        depth = architecture["depth"] or {}
        rows.append(
            {
                "site_id": site["site_id"],
                "status": site["status"],
                "canary_status": site["canary_status"],
                "kind": identity["kind"],
                "module_path": identity.get("module_path", ""),
                "module_call_index": identity.get("module_call_index", ""),
                "event_name": identity.get("event_name", ""),
                "event_call_index": identity.get("event_call_index", ""),
                "output_port": identity["output_port"],
                "value_families": "|".join(site["value_families"]),
                "schemas": json.dumps(site["schemas"], sort_keys=True),
                "topologies": "|".join(site["topologies"]),
                "fault_interfaces": "|".join(site["fault_interfaces"]),
                "observed_owners": "|".join(architecture["observed_owners"]),
                "depth_family": depth.get("family", ""),
                "depth_index": depth.get("index", ""),
                "normalized_depth": depth.get("normalized", ""),
                "observed_source_count": site["observed_source_count"],
                "observation_count": site["observation_count"],
                "consecutive_pair_count": site["consecutive_pair_count"],
                "matching_schema_pair_count": site["matching_schema_pair_count"],
                "eligible_opportunity_count": site["eligible_opportunity_count"],
                "same_value_alias_count": len(site["same_value_alias_site_ids"]),
                "eligibility_reasons": "|".join(site["eligibility_reasons"]),
            }
        )
    return rows
