from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.temporal_campaign import (
    PHASES,
    clean_rollout_frame,
)


ACTION_TOKEN_COUNT = 7
LANGUAGE_BLOCK_COUNT = 32
TRAJECTORIES_PER_TASK = 5
DEVELOPMENT_TRAJECTORIES_PER_TASK = 3
_BLOCK_PATH = re.compile(r"^policy\.language_model\.model\.layers\.(\d+)$")
CACHE_AWARE_BOUNDARY_STATE = (
    "post-block residual plus exact differential key/value cache entries "
    "from the fault output through the replay boundary"
)
CACHE_REPLAY_STORAGE = (
    "lossless sparse overrides relative to the corresponding archived fault trace; "
    "an absent replay row is bitwise identical"
)
EXACT_CACHE_PORTS = {
    "exact post-rotary current-token key cache entry",
    "exact current-token value cache entry",
}
CONTEXT_INTERFACE_PORTS = {
    "exact processed model inputs",
    "full prompt key/value cache",
    "fused language-block-zero input",
    "post-block final-token residual",
    "pre-block final-token residual",
    "post-attention final-token residual",
    "post-rotary final-token query",
    "complete 256-entry action-token logits",
    *EXACT_CACHE_PORTS,
}


def language_block_sites(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the traced residual output for every block and generation call."""
    # build_temporal_site_table.py records executed module paths and ports from
    # SAFE OpenVLA 300dce26. This filter selects the observed Llama block return
    # `value[0]`; it does not infer or name architectural regions by hand.
    selected = []
    for site in table["sites"]:
        identity = site.get("identity", {})
        match = _BLOCK_PATH.fullmatch(str(identity.get("module_path", "")))
        if (
            match is not None
            and "layer_index" in site
            and "action_token_position" in site
            and site.get("status") is None
        ):
            # A pinned prior language-campaign manifest already contains the
            # mechanically filtered 32 x 7 census and its original site IDs.
            # Reusing those records preserves that provenance without needing
            # the much larger source trace on every worker.
            selected.append(
                {
                    "layer_index": int(site["layer_index"]),
                    "action_token_position": int(site["action_token_position"]),
                    "site_id": site["site_id"],
                    "identity": identity,
                    "observed_schemas": site["observed_schemas"],
                    "topologies": site["topologies"],
                }
            )
            continue
        if (
            site.get("status") != "structurally_eligible_pending_canary"
            or identity.get("kind") != "module_output"
            or identity.get("output_port") != "value[0]"
            or match is None
        ):
            continue
        selected.append(
            {
                "layer_index": int(match.group(1)),
                "action_token_position": int(identity["module_call_index"]),
                "site_id": site["site_id"],
                "identity": identity,
                "observed_schemas": site["schemas"],
                "topologies": site["topologies"],
            }
        )

    expected = {
        (layer, token)
        for layer in range(LANGUAGE_BLOCK_COUNT)
        for token in range(ACTION_TOKEN_COUNT)
    }
    observed = {
        (site["layer_index"], site["action_token_position"])
        for site in selected
    }
    if observed != expected or len(selected) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "language-block site table is not a 32 block x 7 token census: "
            f"missing={missing[:8]}, extra={extra[:8]}, records={len(selected)}"
        )
    return sorted(
        selected,
        key=lambda site: (site["layer_index"], site["action_token_position"]),
    )


def select_clean_trajectories(
    clean_frame: list[dict[str, Any]],
    *,
    seed: int,
    trajectories_per_task: int = TRAJECTORIES_PER_TASK,
    development_trajectories_per_task: int = DEVELOPMENT_TRAJECTORIES_PER_TASK,
    excluded_trajectories: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if trajectories_per_task <= 0:
        raise ValueError("trajectories per task must be positive")
    if not 0 <= development_trajectories_per_task <= trajectories_per_task:
        raise ValueError("development trajectories must be between zero and the total")
    excluded_trajectories = excluded_trajectories or set()
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in clean_frame:
        by_task[int(trajectory["task_id"])].append(trajectory)
    if sorted(by_task) != list(range(10)):
        raise ValueError("clean-success frame does not contain all ten LIBERO-10 tasks")

    rng = random.Random(seed)
    selected = []
    for task_id in range(10):
        full_population = sorted(
            by_task[task_id], key=lambda value: int(value["episode_index"])
        )
        population = [
            value
            for value in full_population
            if (task_id, int(value["episode_index"])) not in excluded_trajectories
        ]
        if len(population) < trajectories_per_task:
            raise ValueError(
                f"task {task_id} has only {len(population)} eligible clean successes, "
                f"fewer than the requested {trajectories_per_task}"
            )
        sample = rng.sample(population, trajectories_per_task)
        for index, trajectory in enumerate(sample):
            selected.append(
                {
                    **trajectory,
                    "analysis_split": (
                        "development"
                        if index < development_trajectories_per_task
                        else "holdout"
                    ),
                    "task_clean_population": len(full_population),
                    "excluded_prior_trajectories": len(full_population) - len(population),
                    "eligible_trajectory_population": len(population),
                    "trajectory_inclusion_probability": (
                        trajectories_per_task / len(population)
                    ),
                }
            )
    return selected


def trajectory_keys_from_manifests(paths: list[Path]) -> set[tuple[int, int]]:
    keys = set()
    for path in paths:
        manifest = load_json(path)
        trajectories = manifest.get("clean_trajectories")
        if not isinstance(trajectories, list):
            raise ValueError(f"excluded manifest has no clean trajectories: {path}")
        for value in trajectories:
            keys.add((int(value["task_id"]), int(value["episode_index"])))
        # A later campaign manifest records the trajectories excluded by its
        # own pinned predecessors. Carry those concrete task/episode identities
        # forward so a new sample is disjoint from the whole declared history,
        # not merely from the immediately preceding sample.
        for value in manifest.get("excluded_prior_trajectories", []):
            keys.add((int(value["task_id"]), int(value["episode_index"])))
    return keys


def build_contexts(
    trajectories: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    contexts = []
    for trajectory in trajectories:
        policy_steps = int(trajectory["policy_steps"])
        if policy_steps < 2:
            raise ValueError("selected clean trajectory has no temporal pair")
        for phase, fraction in PHASES.items():
            policy_step = max(
                1,
                min(policy_steps - 1, round((policy_steps - 1) * fraction)),
            )
            contexts.append(
                {
                    "task_id": int(trajectory["task_id"]),
                    "episode_index": int(trajectory["episode_index"]),
                    "trial_seed": int(trajectory["trial_seed"]),
                    "initial_state_sha256": trajectory["initial_state_sha256"],
                    "clean_policy_steps": policy_steps,
                    "analysis_split": trajectory["analysis_split"],
                    "phase": phase,
                    "phase_fraction": fraction,
                    "policy_step": policy_step,
                    "source_policy_step": policy_step - 1,
                }
            )

    contexts.sort(
        key=lambda value: (
            value["task_id"],
            value["episode_index"],
            PHASES[value["phase"]],
        )
    )
    assignment_rng = random.Random(seed ^ 0x7A11C0DE)
    complete_cycles, remainder = divmod(len(contexts), ACTION_TOKEN_COUNT)
    token_positions = list(range(ACTION_TOKEN_COUNT)) * complete_cycles
    token_positions.extend(assignment_rng.sample(range(ACTION_TOKEN_COUNT), remainder))
    assignment_rng.shuffle(token_positions)
    trajectory_keys = sorted(
        {(value["task_id"], value["episode_index"]) for value in contexts}
    )
    worker_by_trajectory = {
        key: index % 2 for index, key in enumerate(trajectory_keys)
    }
    for index, (context, token_position) in enumerate(
        zip(contexts, token_positions, strict=True)
    ):
        context["context_id"] = f"c{index:03d}"
        context["action_token_position"] = token_position
        context["worker_shard"] = worker_by_trajectory[
            (context["task_id"], context["episode_index"])
        ]
    return contexts


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build_language_campaign_manifest(
    table_path: Path,
    clean_root: Path,
    *,
    seed: int,
    trajectories_per_task: int = TRAJECTORIES_PER_TASK,
    development_trajectories_per_task: int = DEVELOPMENT_TRAJECTORIES_PER_TASK,
    exclude_manifest_paths: list[Path] | None = None,
    instrumentation: dict[str, Any] | None = None,
    clean_population: str = "successes",
) -> dict[str, Any]:
    exclude_manifest_paths = exclude_manifest_paths or []
    table = load_json(table_path)
    sites = language_block_sites(table)
    excluded_trajectories = trajectory_keys_from_manifests(exclude_manifest_paths)
    if clean_population not in {"successes", "all"}:
        raise ValueError("clean population must be 'successes' or 'all'")
    trajectories = select_clean_trajectories(
        clean_rollout_frame(
            clean_root, successful_only=clean_population == "successes"
        ),
        seed=seed,
        trajectories_per_task=trajectories_per_task,
        development_trajectories_per_task=development_trajectories_per_task,
        excluded_trajectories=excluded_trajectories,
    )
    contexts = build_contexts(trajectories, seed=seed)

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

    token_counts = Counter(
        context["action_token_position"] for context in contexts
    )
    split_counts = Counter(context["analysis_split"] for context in contexts)
    worker_counts = Counter(context["worker_shard"] for context in contexts)
    return {
        "schema_version": 2,
        "campaign": "openvla_language_block_temporal_replacement",
        "seed": seed,
        "fault_model": {
            "operator": (
                "replace one block's final action-token vector at decision t "
                "with the corresponding vector from decision t-1"
            ),
            "duration": "one policy inference",
            "tensor_slice": "value[0][:, -1:, :]",
        },
        "site_table": {
            "path": str(table_path.resolve()),
            "sha256": file_sha256(table_path),
        },
        "sampling_design": {
            "site_population": "census all 32 language-block residual outputs",
            "trajectory_sampling": (
                f"seeded uniform sample of {trajectories_per_task} eligible "
                + (
                    "completed clean rollouts per task without replacement"
                    if clean_population == "all"
                    else "successful clean rollouts per task without replacement"
                )
            ),
            "clean_population": (
                "completed unperturbed rollouts, regardless of task outcome"
                if clean_population == "all"
                else "completed successful unperturbed rollouts"
            ),
            "prior_trajectory_exclusion": (
                "exclude every task and episode pair named by the pinned prior manifests"
                if exclude_manifest_paths
                else "none"
            ),
            "phase_fractions": PHASES,
            "analysis_split": (
                f"within each task, keep {development_trajectories_per_task} "
                "trajectories in development and "
                f"{trajectories_per_task - development_trajectories_per_task} in "
                "holdout; all phases from one trajectory stay together"
            ),
            "token_assignment": (
                "seeded shuffle of a balanced seven-position multiset; counts differ "
                "by at most one"
            ),
            "worker_assignment": (
                "alternate sorted trajectories between two workers; all three phases "
                "from one trajectory remain on one worker"
            ),
        },
        "counts": {
            "language_blocks": LANGUAGE_BLOCK_COUNT,
            "action_token_positions": ACTION_TOKEN_COUNT,
            "trajectories": len(trajectories),
            "trajectories_per_task": trajectories_per_task,
            "development_trajectories_per_task": (
                development_trajectories_per_task
            ),
            "contexts": len(contexts),
            "local_interventions": len(contexts) * LANGUAGE_BLOCK_COUNT,
            "contexts_by_split": dict(sorted(split_counts.items())),
            "contexts_by_token_position": {
                str(key): value for key, value in sorted(token_counts.items())
            },
            "contexts_by_worker_shard": {
                str(key): value for key, value in sorted(worker_counts.items())
            },
        },
        "sites": sites,
        "excluded_prior_manifests": [
            _artifact_record(path) for path in exclude_manifest_paths
        ],
        "excluded_prior_trajectory_count": len(excluded_trajectories),
        "excluded_prior_trajectories": [
            {"task_id": task_id, "episode_index": episode_index}
            for task_id, episode_index in sorted(excluded_trajectories)
        ],
        "instrumentation": instrumentation or {},
        "clean_trajectories": clean_trajectories,
        "contexts": contexts,
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_language_campaign_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("language campaign manifest must use schema version 1 or 2")
    sites = manifest.get("sites")
    contexts = manifest.get("contexts")
    if not isinstance(sites, list) or len(sites) != 224:
        raise ValueError("language campaign must contain the 32 x 7 site census")
    expected_contexts = int(manifest.get("counts", {}).get("contexts", -1))
    if not isinstance(contexts, list) or len(contexts) != expected_contexts:
        raise ValueError("language campaign contexts disagree with its frozen count")
    if manifest.get("schema_version") == 1 and len(contexts) != 150:
        raise ValueError("version 1 language campaign must contain 150 contexts")
    counts = manifest.get("counts", {})
    trajectories_per_task = int(
        counts.get("trajectories_per_task", TRAJECTORIES_PER_TASK)
    )
    development_per_task = int(
        counts.get(
            "development_trajectories_per_task",
            DEVELOPMENT_TRAJECTORIES_PER_TASK,
        )
    )
    trajectories = manifest.get("clean_trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 10 * trajectories_per_task:
        raise ValueError("language campaign has the wrong balanced trajectory count")
    trajectories_by_task = Counter(int(value["task_id"]) for value in trajectories)
    if trajectories_by_task != Counter({task: trajectories_per_task for task in range(10)}):
        raise ValueError("language campaign trajectories are not balanced by task")
    keys = {
        (int(site["layer_index"]), int(site["action_token_position"]))
        for site in sites
    }
    if len(keys) != 224:
        raise ValueError("language campaign sites are duplicated")
    context_ids = [context.get("context_id") for context in contexts]
    if None in context_ids or len(set(context_ids)) != len(context_ids):
        raise ValueError("language campaign context IDs are missing or duplicated")
    for context in contexts:
        if int(context["source_policy_step"]) != int(context["policy_step"]) - 1:
            raise ValueError("language campaign context does not use source step t-1")
    token_counts = Counter(int(value["action_token_position"]) for value in contexts)
    if set(token_counts) != set(range(ACTION_TOKEN_COUNT)):
        raise ValueError("language campaign does not cover all action-token positions")
    if max(token_counts.values()) - min(token_counts.values()) > 1:
        raise ValueError("language campaign action-token positions are not balanced")
    trajectories_by_split: dict[tuple[int, int], set[str]] = defaultdict(set)
    workers_by_trajectory: dict[tuple[int, int], set[int]] = defaultdict(set)
    for context in contexts:
        key = (int(context["task_id"]), int(context["episode_index"]))
        trajectories_by_split[key].add(str(context["analysis_split"]))
        workers_by_trajectory[key].add(int(context["worker_shard"]))
    if any(len(splits) != 1 for splits in trajectories_by_split.values()):
        raise ValueError("a trajectory appears in both analysis splits")
    if any(len(workers) != 1 for workers in workers_by_trajectory.values()):
        raise ValueError("a trajectory is split across workers")
    context_counts = Counter(
        (int(value["task_id"]), int(value["episode_index"])) for value in contexts
    )
    if set(context_counts) != set(trajectories_by_split) or any(
        count != len(PHASES) for count in context_counts.values()
    ):
        raise ValueError("each selected trajectory must contribute every declared phase")
    for task_id in range(10):
        task_splits = Counter(
            str(value["analysis_split"])
            for value in trajectories
            if int(value["task_id"]) == task_id
        )
        expected_splits = Counter(
            {
                "development": development_per_task,
                "holdout": trajectories_per_task - development_per_task,
            }
        )
        if task_splits != expected_splits:
            raise ValueError("language campaign analysis split is not balanced by task")
    excluded = {
        (int(value["task_id"]), int(value["episode_index"]))
        for value in manifest.get("excluded_prior_trajectories", [])
    }
    selected = set(trajectories_by_split)
    if selected & excluded:
        raise ValueError("language campaign reuses an excluded prior trajectory")
    if int(manifest.get("excluded_prior_trajectory_count", len(excluded))) != len(
        excluded
    ):
        raise ValueError("excluded prior trajectory count disagrees with its records")
    prior_paths = manifest.get("excluded_prior_manifests", [])
    if manifest.get("schema_version") == 2 and not isinstance(prior_paths, list):
        raise ValueError("prior manifest provenance must be a list")
    instrumentation = manifest.get("instrumentation", {})
    if instrumentation.get("full_language_interfaces"):
        ports = set(instrumentation.get("language_ports", []))
        if not EXACT_CACHE_PORTS <= ports:
            raise ValueError(
                "full-interface campaign does not capture both exact cache entries"
            )
        if instrumentation.get("boundary_state") != CACHE_AWARE_BOUNDARY_STATE:
            raise ValueError(
                "full-interface campaign does not declare cache-aware boundary replay"
            )
        if instrumentation.get("boundary_replay_storage") != CACHE_REPLAY_STORAGE:
            raise ValueError(
                "full-interface campaign does not declare lossless replay storage"
            )
    if instrumentation.get("context_conditioned_interfaces"):
        ports = set(instrumentation.get("language_ports", []))
        missing = CONTEXT_INTERFACE_PORTS - ports
        if missing:
            raise ValueError(
                f"context-interface campaign omits declared ports: {sorted(missing)}"
            )
        if instrumentation.get("terminal_branches") is not False:
            raise ValueError(
                "context-interface collection must explicitly defer terminal branches"
            )
