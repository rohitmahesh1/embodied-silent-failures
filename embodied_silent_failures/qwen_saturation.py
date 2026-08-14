import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, json_sha256, load_json


SATURATION_BASIS = (
    "protocol:qwen-internal-saturation-v1:six-condition-alarm-discovery-strata-"
    "plus-thirty-seeded-distinct-trajectory-holdouts"
)
STRATA = (
    ("ordinary", "clean", False),
    ("ordinary", "clean", True),
    ("control", "current_image_control", False),
    ("control", "current_image_control", True),
    ("stale", "stale_image", False),
    ("stale", "stale_image", True),
)
COVERAGE_KINDS = ("regions", "edges", "operators", "processor_shapes")


def query_candidates(
    trials_dir: Path,
    *,
    configuration_sha256: str,
    history_frames: int,
    condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(trials_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"Qwen saturation selection found no trials: {trials_dir}")
    candidates = []
    census = []
    for path in paths:
        trial = load_json(path)
        if trial.get("status") != "complete":
            raise ValueError(f"Qwen saturation selection found incomplete trial: {path}")
        if trial.get("configuration_sha256") != configuration_sha256:
            raise ValueError(f"Qwen saturation trial configuration disagrees: {path}")
        if trial.get("condition") != condition:
            continue
        trial_sha256 = file_sha256(path)
        eligible = []
        for query in trial.get("timeline", []):
            if len(query.get("frame_steps", [])) != history_frames:
                continue
            if query.get("parse_error") is not None or query.get("parsed_response") is None:
                continue
            tokens = query.get("generated_token_ids")
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(f"eligible Qwen query has no generated tokens: {path}")
            record = {
                "trial_path": path.resolve(),
                "trial": trial,
                "trial_sha256": trial_sha256,
                "query": query,
                "policy_step": int(query["policy_step"]),
                "generated_tokens": len(tokens),
                "alarm": bool(query["alarm"]),
            }
            candidates.append(record)
            eligible.append(
                {
                    "policy_step": record["policy_step"],
                    "generated_tokens": record["generated_tokens"],
                    "alarm": record["alarm"],
                }
            )
        census.append(
            {
                "trial": path.name,
                "trial_sha256": trial_sha256,
                "eligible_queries": eligible,
            }
        )
    if not candidates:
        raise ValueError(f"Qwen saturation stratum has no queries: {condition}")
    return candidates, census


def select_saturation_queries(
    sources: dict[str, dict[str, Any]], *, seed: int, holdouts_per_stratum: int = 5
) -> dict[str, Any]:
    if holdouts_per_stratum <= 0:
        raise ValueError("holdouts per saturation stratum must be positive")
    discovery = []
    holdouts_by_stratum = []
    census_records = []
    stratum_records = []
    discovery_trials: dict[str, set[str]] = {}
    for label, condition, alarm in STRATA:
        source = sources[label]
        candidates, census = query_candidates(
            Path(source["trials_dir"]),
            configuration_sha256=source["run"]["configuration_sha256"],
            history_frames=int(
                source["run"]["configuration"]["protocol"]["history_frames"]
            ),
            condition=condition,
        )
        stratum = f"{label}--alarm-{int(alarm)}"
        eligible = [item for item in candidates if item["alarm"] is alarm]
        if not eligible:
            raise ValueError(f"Qwen saturation stratum is empty: {stratum}")
        eligible.sort(
            key=lambda item: (
                -item["generated_tokens"],
                item["trial_path"].name,
                item["policy_step"],
            )
        )
        eligible_discovery = [
            item
            for item in eligible
            if item["trial_path"].name not in discovery_trials.setdefault(label, set())
        ]
        if not eligible_discovery:
            raise ValueError(
                f"Qwen saturation stratum lacks a distinct discovery trial: {stratum}"
            )
        chosen_discovery = _selected_record(
            eligible_discovery[0], source, phase="discovery", stratum=stratum
        )
        chosen_discovery["selection_rule"] = (
            "longest-valid-full-history-response-excluding-prior-condition-"
            "discovery-trajectories-then-lexical-trial-and-policy-step"
        )
        discovery.append(chosen_discovery)
        discovery_trials[label].add(chosen_discovery["trial_path"].name)
        stratum_records.append(
            {
                "label": label,
                "source": source,
                "stratum": stratum,
                "eligible": eligible,
            }
        )
        census_records.append(
            {
                "stratum": stratum,
                "condition": condition,
                "alarm": alarm,
                "census_sha256": json_sha256(census),
                "eligible_queries": len(eligible),
                "eligible_trajectories": len(
                    {item["trial_path"].name for item in eligible}
                ),
            }
        )

    # Within each condition, discovery and holdout cells use different
    # trajectories. The fixed STRATA order makes the exclusion deterministic.
    reserved_by_label: dict[str, set[str]] = {}
    for item in discovery:
        reserved_by_label.setdefault(item["condition_label"], set()).add(
            item["trial_path"].name
        )
    for record in stratum_records:
        label = record["label"]
        source = record["source"]
        stratum = record["stratum"]
        by_trial: dict[str, list[dict[str, Any]]] = {}
        for item in record["eligible"]:
            if item["trial_path"].name in reserved_by_label[label]:
                continue
            by_trial.setdefault(item["trial_path"].name, []).append(item)
        ranked_trials = sorted(
            by_trial,
            key=lambda trial: (_seed_rank(seed, stratum, "trial", trial), trial),
        )
        if len(ranked_trials) < holdouts_per_stratum:
            raise ValueError(f"Qwen saturation stratum lacks distinct holdouts: {stratum}")
        chosen_holdouts = []
        for trial_name in ranked_trials[:holdouts_per_stratum]:
            trial_queries = sorted(
                by_trial[trial_name],
                key=lambda item: (
                    _seed_rank(
                        seed,
                        stratum,
                        "query",
                        f"{trial_name}:{item['policy_step']}",
                    ),
                    item["policy_step"],
                ),
            )
            selected = _selected_record(
                trial_queries[0], source, phase="holdout", stratum=stratum
            )
            selected["selection_rule"] = (
                "seeded-sha256-rank-condition-distinct-trajectory-then-seeded-query"
            )
            chosen_holdouts.append(selected)
            reserved_by_label[label].add(trial_name)
        holdouts_by_stratum.append(chosen_holdouts)

    # Round-robin ordering prevents one stratum from being confounded with a
    # contiguous late segment of a long-running campaign.
    holdouts = [
        stratum_items[round_index]
        for round_index in range(holdouts_per_stratum)
        for stratum_items in holdouts_by_stratum
    ]
    selections = [*discovery, *holdouts]
    for index, selected in enumerate(selections):
        selected["index"] = index
    manifest_records = [_public_selection(item) for item in selections]
    return {
        "schema_version": 1,
        "basis": SATURATION_BASIS,
        "seed": seed,
        "alarm_role": "predeclared_balance_stratum_only; never used to measure novelty",
        "holdouts_per_stratum": holdouts_per_stratum,
        "discovery_queries": len(discovery),
        "holdout_queries": len(holdouts),
        "total_queries": len(selections),
        "census": census_records,
        "census_sha256": json_sha256(census_records),
        "selections": selections,
        "public_selections": manifest_records,
        "selection_sha256": json_sha256(manifest_records),
    }


def coverage_record(
    events: Iterable[dict[str, Any]],
    graph: dict[str, Any],
    processed_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    region_by_id = {}
    regions = {}
    for region in graph["regions"]:
        record = {
            "name": region["name"],
            "semantic_key": region["semantic_key"],
            "lifetime": region.get("lifetime"),
            "fault_interface": region.get("fault_interface"),
            "disposition": region.get("disposition"),
            "basis": sorted(region.get("basis", [])),
        }
        signature = json_sha256(record)
        region_by_id[region["region_id"]] = signature
        regions[signature] = record
    edges = {}
    for edge in graph["edges"]:
        record = {
            "source": region_by_id[edge["source"]],
            "target": region_by_id[edge["target"]],
            "kind": edge["kind"],
        }
        edges[json_sha256(record)] = record
    operators = {}
    for event in events:
        if event.get("kind") != "operator":
            continue
        details = event.get("details", {})
        semantics = details.get("operator_semantics", {})
        calls = details.get("module_calls", [])
        scopes = details.get("module_scope", [])
        module_path = calls[-1].get("path") if calls else (scopes[-1] if scopes else "")
        record = {
            "module_path": module_path,
            "schema": semantics.get("schema"),
        }
        operators[json_sha256(record)] = record
    processor_shapes = {}
    for item in processed_inputs:
        record = {
            "name": item["name"],
            "shape": item["shape"],
            "dtype": item["dtype"],
        }
        processor_shapes[json_sha256(record)] = record
    return {
        "schema_version": 1,
        "definition": {
            "regions": "canonical declared region identity",
            "edges": "canonical region endpoints and reduced edge kind",
            "operators": "PyTorch operator schema within the innermost observed module",
            "processor_shapes": "named processor tensor shape and dtype",
        },
        "regions": [regions[key] | {"signature": key} for key in sorted(regions)],
        "edges": [edges[key] | {"signature": key} for key in sorted(edges)],
        "operators": [
            operators[key] | {"signature": key} for key in sorted(operators)
        ],
        "processor_shapes": [
            processor_shapes[key] | {"signature": key}
            for key in sorted(processor_shapes)
        ],
    }


def coverage_signatures(coverage: dict[str, Any]) -> dict[str, set[str]]:
    return {
        kind: {item["signature"] for item in coverage[kind]}
        for kind in COVERAGE_KINDS
    }


def coverage_novelty(
    coverage: dict[str, Any], baseline: dict[str, set[str]]
) -> dict[str, Any]:
    current = coverage_signatures(coverage)
    counts = {kind: len(current[kind] - baseline[kind]) for kind in COVERAGE_KINDS}
    return {"novel": any(counts.values()), "new": counts}


def update_coverage_union(
    baseline: dict[str, set[str]], coverage: dict[str, Any]
) -> None:
    current = coverage_signatures(coverage)
    for kind in COVERAGE_KINDS:
        baseline[kind].update(current[kind])


def empty_coverage_union() -> dict[str, set[str]]:
    return {kind: set() for kind in COVERAGE_KINDS}


def zero_discovery_upper_bound(zero_queries: int, confidence: float = 0.95) -> float:
    if zero_queries <= 0:
        raise ValueError("zero-discovery query count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    return 1.0 - (1.0 - confidence) ** (1.0 / zero_queries)


def _seed_rank(seed: int, stratum: str, level: str, key: str) -> str:
    value = f"{seed}\0{stratum}\0{level}\0{key}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _selected_record(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    phase: str,
    stratum: str,
) -> dict[str, Any]:
    return {
        **candidate,
        "run": source["run"],
        "run_path": Path(source["run_path"]).resolve(),
        "run_sha256": source["run_sha256"],
        "phase": phase,
        "stratum": stratum,
        "condition_label": source["label"],
    }


def _public_selection(selected: dict[str, Any]) -> dict[str, Any]:
    trial = selected["trial"]
    query = selected["query"]
    return {
        "index": selected.get("index"),
        "phase": selected["phase"],
        "stratum": selected["stratum"],
        "condition": trial["condition"],
        "alarm": selected["alarm"],
        "trial": selected["trial_path"].name,
        "trial_sha256": selected["trial_sha256"],
        "run_sha256": selected["run_sha256"],
        "task_id": int(trial["task_id"]),
        "episode_index": int(trial["episode_index"]),
        "policy_step": selected["policy_step"],
        "frame_steps": [int(value) for value in query["frame_steps"]],
        "generated_tokens": selected["generated_tokens"],
        "selection_rule": selected["selection_rule"],
    }
