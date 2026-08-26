from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, load_json


PHASES = {
    "early": 0.25,
    "middle": 0.50,
    "late": 0.75,
}


def depth_band(site: dict[str, Any]) -> str:
    depth = site["architecture"].get("depth")
    if not isinstance(depth, dict):
        return "not_applicable"
    normalized = float(depth["normalized"])
    if normalized < 1 / 3:
        return "early"
    if normalized < 2 / 3:
        return "middle"
    return "late"


def output_family(site: dict[str, Any]) -> str:
    port = str(site["identity"]["output_port"])
    if port == "value":
        return "direct"
    if port.startswith("value.past_key_values"):
        return "past_key_values"
    if port.startswith("value.hidden_states"):
        return "returned_hidden_states"
    if port == "value.last_hidden_state":
        return "last_hidden_state"
    if port == "value.logits":
        return "logits"
    if port.startswith("value["):
        return "tuple_port"
    return "declared_or_other_port"


def sampling_stratum(site: dict[str, Any]) -> str:
    owners = site["architecture"].get("observed_owners", [])
    owner = owners[0] if len(owners) == 1 else "|".join(sorted(owners))
    return ":".join((owner, depth_band(site), output_family(site)))


def eligible_sites(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        site
        for site in table["sites"]
        if site["status"] == "structurally_eligible_pending_canary"
    ]


def sample_sites(
    table: dict[str, Any], *, seed: int, shared_per_stratum: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if shared_per_stratum <= 0:
        raise ValueError("shared sites per stratum must be positive")
    candidates = eligible_sites(table)
    action_only = [site for site in candidates if site["topologies"] == ["action_only"]]
    shared_modules = [
        site
        for site in candidates
        if site["topologies"] == ["shared_action_and_monitor_evidence"]
        and site["identity"]["kind"] == "module_output"
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for site in shared_modules:
        groups[sampling_stratum(site)].append(site)

    rng = random.Random(seed)
    selected = []
    stratum_sizes = {name: len(values) for name, values in sorted(groups.items())}
    for name, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda site: site["site_id"])
        count = min(shared_per_stratum, len(ordered))
        for site in rng.sample(ordered, count):
            selected.append(
                {
                    "site": site,
                    "selection": "stratified_seeded_uniform_without_replacement",
                    "stratum": name,
                    "stratum_population": len(ordered),
                    "stratum_sample": count,
                    "site_inclusion_probability": count / len(ordered),
                }
            )
    for site in sorted(action_only, key=lambda value: value["site_id"]):
        selected.append(
            {
                "site": site,
                "selection": "action_only_census",
                "stratum": "action_only_census",
                "stratum_population": len(action_only),
                "stratum_sample": len(action_only),
                "site_inclusion_probability": 1.0,
            }
        )
    return sorted(selected, key=lambda value: value["site"]["site_id"]), stratum_sizes


def clean_success_frame(clean_root: Path) -> list[dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(clean_root.rglob("*.complete.json")):
        result = load_json(path)
        if result.get("condition") != "clean" or result.get("status") != "complete":
            continue
        key = (int(result["task_id"]), int(result["episode_index"]))
        if key in indexed:
            raise ValueError(f"duplicate clean baseline result for {key}")
        if result.get("success") is not True:
            continue
        files = result.get("files", {})
        csv_path = path.parent / str(files.get("csv"))
        pickle_path = path.parent / str(files.get("pickle"))
        if not csv_path.is_file() or not pickle_path.is_file():
            raise FileNotFoundError(f"clean baseline artifacts are incomplete: {path}")
        indexed[key] = {
            "task_id": key[0],
            "episode_index": key[1],
            "policy_steps": int(result["policy_steps"]),
            "trial_seed": int(result["trial_seed"]),
            "initial_state_sha256": str(result["initial_state_sha256"]),
            "source": {
                "completion": path,
                "csv": csv_path,
                "pickle": pickle_path,
            },
        }
    if not indexed:
        raise ValueError("clean baseline contains no successful trajectories")
    return [indexed[key] for key in sorted(indexed)]


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "staged_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _site_record(selection: dict[str, Any]) -> dict[str, Any]:
    site = selection["site"]
    return {
        "site_id": site["site_id"],
        "identity": site["identity"],
        "topologies": site["topologies"],
        "architecture": site["architecture"],
        "value_families": site["value_families"],
        "schemas": site["schemas"],
        "same_value_alias_site_ids": site["same_value_alias_site_ids"],
        **{key: value for key, value in selection.items() if key != "site"},
    }


def build_campaign_manifest(
    table_path: Path,
    clean_root: Path,
    *,
    seed: int,
    shared_per_stratum: int = 2,
) -> dict[str, Any]:
    table = load_json(table_path)
    sites, stratum_sizes = sample_sites(
        table, seed=seed, shared_per_stratum=shared_per_stratum
    )
    clean_frame = clean_success_frame(clean_root)
    context_rng = random.Random(seed ^ 0xA57C0F11)
    artifact_cache: dict[Path, dict[str, Any]] = {}
    attempts = []
    selected_clean: dict[tuple[int, int], dict[str, Any]] = {}
    for selection in sites:
        site = _site_record(selection)
        for phase_name, phase_fraction in PHASES.items():
            clean = clean_frame[context_rng.randrange(len(clean_frame))]
            key = (clean["task_id"], clean["episode_index"])
            intervention_step = max(
                1,
                min(
                    clean["policy_steps"] - 1,
                    round((clean["policy_steps"] - 1) * phase_fraction),
                ),
            )
            if clean["policy_steps"] < 2:
                raise ValueError(f"clean trajectory {key} has no temporal pair")
            artifacts = {}
            for name, path in clean["source"].items():
                if path not in artifact_cache:
                    artifact_cache[path] = _artifact_record(path)
                artifacts[name] = artifact_cache[path]
            selected_clean[key] = {
                key_name: value
                for key_name, value in clean.items()
                if key_name != "source"
            } | {"artifacts": artifacts}
            attempt_index = len(attempts)
            attempts.append(
                {
                    "attempt_id": f"a{attempt_index:04d}",
                    "site_id": site["site_id"],
                    "task_id": clean["task_id"],
                    "episode_index": clean["episode_index"],
                    "phase": phase_name,
                    "phase_fraction": phase_fraction,
                    "policy_step": intervention_step,
                    "source_policy_step": intervention_step - 1,
                    "clean_frame_population": len(clean_frame),
                    "clean_draw_probability": 1 / len(clean_frame),
                    "site_inclusion_probability": site[
                        "site_inclusion_probability"
                    ],
                    "joint_draw_probability": site["site_inclusion_probability"]
                    / len(clean_frame),
                }
            )

    topology_counts = Counter(
        topology for item in sites for topology in item["site"]["topologies"]
    )
    return {
        "schema_version": 1,
        "campaign": "openvla_single_temporal_value_replacement_pilot",
        "fault_operator": "replace x_t with x_(t-1) at one table-defined site",
        "seed": seed,
        "site_table": {
            "path": str(table_path.resolve()),
            "sha256": file_sha256(table_path),
            "eligible_population": len(eligible_sites(table)),
        },
        "sampling_design": {
            "shared_sites": (
                "Within every nonempty owner x normalized-depth-band x syntactic-"
                "output-family stratum, draw an equal number of sites uniformly "
                "without replacement. Literal module roles remain observed rather "
                "than selected by hand."
            ),
            "action_only_sites": (
                "Census all five action-only sites and analyze them descriptively; "
                "they are not matched controls for internal shared sites."
            ),
            "trajectory_context": (
                "For each selected site and each predeclared normalized phase, draw "
                "one trajectory uniformly from the frozen clean-success frame."
            ),
            "shared_per_stratum": shared_per_stratum,
            "shared_stratum_populations": stratum_sizes,
            "phase_fractions": PHASES,
            "clean_success_population": len(clean_frame),
        },
        "counts": {
            "selected_sites": len(sites),
            "attempts": len(attempts),
            "selected_clean_trajectories": len(selected_clean),
            "selected_topologies": dict(sorted(topology_counts.items())),
        },
        "sites": [_site_record(item) for item in sites],
        "clean_trajectories": [selected_clean[key] for key in sorted(selected_clean)],
        "attempts": attempts,
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_campaign_manifest(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise ValueError("temporal campaign manifest must use schema version 1")
    sites = value.get("sites")
    attempts = value.get("attempts")
    if not isinstance(sites, list) or not sites:
        raise ValueError("temporal campaign manifest has no sites")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("temporal campaign manifest has no attempts")
    site_ids = {site.get("site_id") for site in sites}
    if None in site_ids or len(site_ids) != len(sites):
        raise ValueError("temporal campaign site IDs are missing or duplicated")
    attempt_ids = [attempt.get("attempt_id") for attempt in attempts]
    if None in attempt_ids or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("temporal campaign attempt IDs are missing or duplicated")
    for attempt in attempts:
        if attempt.get("site_id") not in site_ids:
            raise ValueError("temporal campaign attempt refers to an unknown site")
        if int(attempt["source_policy_step"]) != int(attempt["policy_step"]) - 1:
            raise ValueError("temporal campaign attempt does not use source step t-1")
