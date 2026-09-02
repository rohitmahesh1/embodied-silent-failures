from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_campaign import (
    CACHE_AWARE_BOUNDARY_STATE,
    CACHE_REPLAY_STORAGE,
    CONTEXT_INTERFACE_PORTS,
    build_language_campaign_manifest,
    validate_language_campaign_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the OpenVLA language-block residual-risk campaign."
    )
    parser.add_argument("--site-table", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--trajectories-per-task", type=int, default=5)
    parser.add_argument("--development-trajectories-per-task", type=int, default=3)
    parser.add_argument(
        "--clean-population",
        choices=("successes", "all"),
        default="successes",
        help="Sample successful clean rollouts or every completed clean rollout.",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        type=Path,
        help="Exclude every task/episode trajectory used by this prior manifest.",
    )
    parser.add_argument(
        "--instrument-full-interfaces",
        action="store_true",
        help=(
            "Archive declared full-vector language ports, action logits, "
            "post-intervention trajectories, and measured boundary replays."
        ),
    )
    parser.add_argument(
        "--instrument-context-interfaces",
        action="store_true",
        help=(
            "Collect exact model inputs, prompt cache, attention conditioning, "
            "and internal language cuts without terminal rollouts."
        ),
    )
    args = parser.parse_args()
    if args.instrument_full_interfaces and args.instrument_context_interfaces:
        parser.error("choose one interface instrumentation contract")

    instrumentation = {}
    if args.instrument_context_interfaces:
        instrumentation = {
            "full_language_interfaces": True,
            "context_conditioned_interfaces": True,
            "language_ports": sorted(CONTEXT_INTERFACE_PORTS),
            "boundary_replays": [],
            "boundary_state": CACHE_AWARE_BOUNDARY_STATE,
            "boundary_replay_storage": CACHE_REPLAY_STORAGE,
            "terminal_branches": False,
            "collection_scope": (
                "source, clean, and all 32 mechanically enumerated local "
                "language-block interventions at each selected context"
            ),
            "deferred_outputs": (
                "terminal control and fault branches are deferred until a local "
                "interface candidate has measurable held-out signal"
            ),
        }
    elif args.instrument_full_interfaces:
        instrumentation = {
            "full_language_interfaces": True,
            "language_ports": [
                "post-block final-token residual",
                "exact post-rotary current-token key cache entry",
                "exact current-token value cache entry",
                "complete 256-entry action-token logits",
            ],
            "boundary_replays": ["immediate", "final"],
            "boundary_state": CACHE_AWARE_BOUNDARY_STATE,
            "boundary_replay_storage": CACHE_REPLAY_STORAGE,
            "terminal_trajectory": (
                "exact simulator state, every stable numeric observation, "
                "and action-side policy evidence from intervention through "
                "terminal outcome"
            ),
        }

    manifest = build_language_campaign_manifest(
        args.site_table,
        args.clean_root,
        seed=args.seed,
        trajectories_per_task=args.trajectories_per_task,
        development_trajectories_per_task=(
            args.development_trajectories_per_task
        ),
        exclude_manifest_paths=args.exclude_manifest,
        instrumentation=instrumentation,
        clean_population=args.clean_population,
    )
    validate_language_campaign_manifest(manifest)
    write_json_atomic(args.output, manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
