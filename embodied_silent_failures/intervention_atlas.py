from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.language_campaign import (
    PHASES,
    clean_rollout_frame,
    select_clean_trajectories,
    trajectory_keys_from_manifests,
)
from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.temporal_campaign import depth_band, output_family


KNOWN_TOPOLOGIES = {
    "shared_action_and_monitor_evidence",
    "action_only",
    "monitor_evidence_only",
}


def topology_label(site: dict[str, Any]) -> str:
    observed = sorted(set(site.get("topologies", [])) & KNOWN_TOPOLOGIES)
    if len(observed) == 1:
        return observed[0]
    if observed:
        return "mixed:" + "|".join(observed)
    return "neither_declared_sink"


def atlas_stratum(site: dict[str, Any]) -> str:
    architecture = site.get("architecture", {})
    owners = architecture.get("observed_owners", [])
    owner = "unassigned" if not owners else "|".join(sorted(str(v) for v in owners))
    return ":".join(
        (
            topology_label(site),
            str(site["identity"]["kind"]),
            owner,
            depth_band(site),
            output_family(site),
        )
    )


def atlas_eligible_sites(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        site
        for site in table["sites"]
        if site.get("status") == "structurally_eligible_pending_canary"
        and topology_label(site) != "neither_declared_sink"
    ]


def intervention_rule(site: dict[str, Any]) -> dict[str, Any]:
    identity = site["identity"]
    module_path = str(identity.get("module_path", ""))
    event_name = str(identity.get("event_name", ""))
    output_port = str(identity["output_port"])
    shapes = [schema.get("shape") for schema in site.get("schemas", [])]
    language_sequence = bool(
        shapes
        and all(isinstance(shape, list) and len(shape) == 3 for shape in shapes)
        and (
            module_path.startswith("policy.language_model")
            or output_port.startswith("value.hidden_states")
            or (
                event_name == "openvla.policy_call"
                and output_port.startswith("value.final_layer_states")
            )
        )
    )
    if language_sequence:
        return {
            "value_slice": "final_sequence_position",
            "basis": (
                "OpenVLA 300dce26 generates seven action tokens autoregressively; "
                "modeling_prismatic.py::predict_action returns a full prompt sequence "
                "on the first call and one position thereafter. Selecting the final "
                "sequence position keeps the language activation interface constant."
            ),
        }
    return {
        "value_slice": "full",
        "basis": "the temporal-site table defines this complete numeric output port",
    }


def sample_atlas_sites(
    table: dict[str, Any],
    *,
    seed: int,
    sites_per_stratum: int,
    census_below: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if sites_per_stratum <= 0:
        raise ValueError("sites per stratum must be positive")
    if census_below < 0:
        raise ValueError("census threshold cannot be negative")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for site in atlas_eligible_sites(table):
        groups[atlas_stratum(site)].append(site)

    rng = random.Random(seed)
    selected = []
    populations = {name: len(values) for name, values in sorted(groups.items())}
    for stratum, values in sorted(groups.items()):
        population = sorted(values, key=lambda value: value["site_id"])
        count = (
            len(population)
            if len(population) <= census_below
            else min(sites_per_stratum, len(population))
        )
        sample = population if count == len(population) else rng.sample(population, count)
        for site in sample:
            selected.append(
                {
                    **site,
                    "intervention": intervention_rule(site),
                    "sampling": {
                        "method": (
                            "complete_stratum_census"
                            if count == len(population)
                            else "seeded_uniform_without_replacement_within_stratum"
                        ),
                        "stratum": stratum,
                        "stratum_population": len(population),
                        "stratum_sample": count,
                        "site_inclusion_probability": count / len(population),
                        "site_inverse_probability_weight": len(population) / count,
                    },
                }
            )
    return sorted(selected, key=lambda value: value["site_id"]), populations


def build_atlas_contexts(
    trajectories: list[dict[str, Any]], *, worker_count: int
) -> list[dict[str, Any]]:
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    trajectory_keys = sorted(
        (int(value["task_id"]), int(value["episode_index"]))
        for value in trajectories
    )
    shard_by_trajectory = {
        key: index % worker_count for index, key in enumerate(trajectory_keys)
    }
    contexts = []
    for trajectory in trajectories:
        policy_steps = int(trajectory["policy_steps"])
        if policy_steps < 2:
            raise ValueError("selected clean trajectory has no temporal pair")
        key = (int(trajectory["task_id"]), int(trajectory["episode_index"]))
        for phase, fraction in PHASES.items():
            policy_step = max(
                1,
                min(policy_steps - 1, round((policy_steps - 1) * fraction)),
            )
            contexts.append(
                {
                    "task_id": key[0],
                    "episode_index": key[1],
                    "trial_seed": int(trajectory["trial_seed"]),
                    "initial_state_sha256": trajectory["initial_state_sha256"],
                    "clean_policy_steps": policy_steps,
                    "analysis_split": trajectory["analysis_split"],
                    "phase": phase,
                    "phase_fraction": fraction,
                    "policy_step": policy_step,
                    "source_policy_step": policy_step - 1,
                    "worker_shard": shard_by_trajectory[key],
                }
            )
    contexts.sort(
        key=lambda value: (
            value["task_id"],
            value["episode_index"],
            PHASES[value["phase"]],
        )
    )
    for index, context in enumerate(contexts):
        context["context_id"] = f"c{index:04d}"
    return contexts


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_intervention_atlas_manifest(
    table_path: Path,
    clean_root: Path,
    *,
    seed: int,
    sites_per_stratum: int,
    census_below: int,
    trajectories_per_task: int,
    development_trajectories_per_task: int,
    worker_count: int,
    exclude_manifest_paths: list[Path] | None = None,
    clean_population: str = "successes",
) -> dict[str, Any]:
    if clean_population not in {"successes", "all"}:
        raise ValueError("clean population must be 'successes' or 'all'")
    exclude_manifest_paths = exclude_manifest_paths or []
    table = load_json(table_path)
    sites, stratum_populations = sample_atlas_sites(
        table,
        seed=seed,
        sites_per_stratum=sites_per_stratum,
        census_below=census_below,
    )
    excluded = trajectory_keys_from_manifests(exclude_manifest_paths)
    trajectories = select_clean_trajectories(
        clean_rollout_frame(
            clean_root, successful_only=clean_population == "successes"
        ),
        seed=seed ^ 0x51A7E,
        trajectories_per_task=trajectories_per_task,
        development_trajectories_per_task=development_trajectories_per_task,
        excluded_trajectories=excluded,
    )
    contexts = build_atlas_contexts(trajectories, worker_count=worker_count)
    clean_trajectories = []
    for trajectory in trajectories:
        source = trajectory["source"]
        clean_trajectories.append(
            {
                **{key: value for key, value in trajectory.items() if key != "source"},
                "artifacts": {
                    name: _artifact_record(path) for name, path in source.items()
                },
            }
        )

    topology_counts = Counter(topology_label(site) for site in sites)
    split_counts = Counter(context["analysis_split"] for context in contexts)
    return {
        "schema_version": 1,
        "campaign": "openvla_graph_derived_temporal_intervention_atlas",
        "seed": seed,
        "fault_model": {
            "operator": "replace x_t with the same site's x_(t-1)",
            "duration": "one policy decision",
            "scope": "one graph-derived output port per intervention",
            "language_sequence_rule": (
                "for rank-three outputs inside policy.language_model, replace only "
                "the final sequence position on every autoregressive call"
            ),
            "physical_prevalence_claim": "none",
        },
        "site_table": {
            "path": str(table_path.resolve()),
            "sha256": file_sha256(table_path),
            "eligible_population": len(atlas_eligible_sites(table)),
        },
        "sampling_design": {
            "site_strata": (
                "observed sink topology x hook kind x graph-declared owner x "
                "normalized depth band x syntactic output family"
            ),
            "site_selection": (
                "seeded uniform sampling without replacement inside every stratum; "
                "small strata are censused"
            ),
            "site_aliases": (
                "recorded for audit but not collapsed, because changing an upstream "
                "producer and a downstream alias need not affect the same consumers"
            ),
            "sites_per_large_stratum": sites_per_stratum,
            "census_strata_at_or_below": census_below,
            "stratum_populations": stratum_populations,
            "trajectory_selection": (
                f"seeded uniform sample of {trajectories_per_task} eligible clean "
                "rollouts per LIBERO-10 task without replacement"
            ),
            "clean_population": clean_population,
            "phase_fractions": PHASES,
            "analysis_split": (
                f"within each task, {development_trajectories_per_task} trajectories "
                "are development and the remainder holdout; all phases from one "
                "trajectory remain together"
            ),
            "matched_contexts": (
                "every selected site is evaluated at every selected context; exact "
                "executed-command aliases share one physical continuation"
            ),
            "worker_assignment": (
                "sorted trajectories are assigned round-robin and never split across workers"
            ),
        },
        "analysis_contract": {
            "primary_unit": "site by trajectory context intervention",
            "primary_outcomes": [
                "policy failure relative to the matched clean continuation",
                "SAFE miss conditional on policy failure",
                "joint silent-failure probability",
            ],
            "primary_test": (
                "fit graph topology and pre-outcome local measurements on development "
                "trajectories, then test residual-risk ranking on untouched holdout trajectories"
            ),
            "ranking_comparison": (
                "compare conventional policy-failure ranking with residual-risk-after-SAFE ranking"
            ),
            "secondary_tests": [
                "restore clean versus faulted SAFE evidence on the same physical suffix",
                "measure when physical and monitor evidence paths first separate "
                "after intervention",
                "test whether origin and propagation history add information after "
                "observed product state",
            ],
            "causal_limits": (
                "topology strata are unlike architectural populations, so differences "
                "are descriptive unless a matched evidence-restoration comparison "
                "isolates the mechanism"
            ),
            "weights": (
                "retain site and trajectory inclusion probabilities; report both "
                "sampled-population estimates and inverse-probability-weighted "
                "graph-population estimates"
            ),
        },
        "capture_contract": {
            "before_intervention": [
                "exact simulator state and full policy observation",
                "exact selected-site values at t-1 and t",
                "clean command, action tokens, action logits, and SAFE input",
            ],
            "per_intervention": [
                "site reach or error status",
                "source-current tensor change",
                "faulted command, action tokens, action logits, and SAFE input",
            ],
            "terminal_branch": [
                "lossless images and numeric observations at each decision",
                "simulator states, commands, action tokens, action logits, SAFE inputs, "
                "and outcome",
            ],
            "failure_policy": (
                "keep local errors, missing sites, unresolved branches, masked interventions, "
                "and failed controls as explicit records; only systematic infrastructure "
                "faults stop a worker"
            ),
        },
        "counts": {
            "selected_sites": len(sites),
            "selected_sites_by_topology": dict(sorted(topology_counts.items())),
            "trajectories": len(trajectories),
            "contexts": len(contexts),
            "contexts_by_split": dict(sorted(split_counts.items())),
            "planned_local_interventions": len(sites) * len(contexts),
            "worker_count": worker_count,
        },
        "sites": sites,
        "clean_trajectories": clean_trajectories,
        "excluded_prior_manifests": [
            _artifact_record(path) for path in exclude_manifest_paths
        ],
        "excluded_prior_trajectory_count": len(excluded),
        "excluded_prior_trajectories": [
            {"task_id": task_id, "episode_index": episode_index}
            for task_id, episode_index in sorted(excluded)
        ],
        "contexts": contexts,
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_intervention_atlas_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("intervention atlas manifest must use schema version 1")
    sites = manifest.get("sites")
    contexts = manifest.get("contexts")
    trajectories = manifest.get("clean_trajectories")
    if not isinstance(sites, list) or not sites:
        raise ValueError("intervention atlas has no sites")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("intervention atlas has no contexts")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("intervention atlas has no clean trajectories")
    site_ids = [site.get("site_id") for site in sites]
    if None in site_ids or len(set(site_ids)) != len(site_ids):
        raise ValueError("intervention atlas site IDs are missing or duplicated")
    context_ids = [context.get("context_id") for context in contexts]
    if None in context_ids or len(set(context_ids)) != len(context_ids):
        raise ValueError("intervention atlas context IDs are missing or duplicated")
    for site in sites:
        sampling = site.get("sampling", {})
        probability = float(sampling.get("site_inclusion_probability", 0))
        if not 0 < probability <= 1:
            raise ValueError("intervention atlas has an invalid site probability")
    for context in contexts:
        if int(context["source_policy_step"]) != int(context["policy_step"]) - 1:
            raise ValueError("intervention atlas context does not use source step t-1")
    split_by_trajectory: dict[tuple[int, int], set[str]] = defaultdict(set)
    shard_by_trajectory: dict[tuple[int, int], set[int]] = defaultdict(set)
    for context in contexts:
        key = (int(context["task_id"]), int(context["episode_index"]))
        split_by_trajectory[key].add(str(context["analysis_split"]))
        shard_by_trajectory[key].add(int(context["worker_shard"]))
    if any(len(values) != 1 for values in split_by_trajectory.values()):
        raise ValueError("a trajectory appears in both atlas analysis splits")
    if any(len(values) != 1 for values in shard_by_trajectory.values()):
        raise ValueError("a trajectory is split across atlas workers")
    context_counts = Counter(
        (int(value["task_id"]), int(value["episode_index"])) for value in contexts
    )
    if any(count != len(PHASES) for count in context_counts.values()):
        raise ValueError("each atlas trajectory must contribute every declared phase")
    expected = manifest.get("counts", {})
    if int(expected.get("selected_sites", -1)) != len(sites):
        raise ValueError("atlas site count disagrees with its records")
    if int(expected.get("contexts", -1)) != len(contexts):
        raise ValueError("atlas context count disagrees with its records")
