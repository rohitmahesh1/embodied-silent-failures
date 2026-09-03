from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.intervention_atlas import (
    build_intervention_atlas_manifest,
    validate_intervention_atlas_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a graph-derived OpenVLA temporal intervention atlas."
    )
    parser.add_argument("--site-table", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--sites-per-stratum", type=int, default=2)
    parser.add_argument("--census-below", type=int, default=5)
    parser.add_argument("--trajectories-per-task", type=int, default=10)
    parser.add_argument("--development-trajectories-per-task", type=int, default=6)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument(
        "--clean-population", choices=("successes", "all"), default="successes"
    )
    parser.add_argument("--exclude-manifest", action="append", default=[], type=Path)
    args = parser.parse_args()
    manifest = build_intervention_atlas_manifest(
        args.site_table,
        args.clean_root,
        seed=args.seed,
        sites_per_stratum=args.sites_per_stratum,
        census_below=args.census_below,
        trajectories_per_task=args.trajectories_per_task,
        development_trajectories_per_task=args.development_trajectories_per_task,
        worker_count=args.worker_count,
        exclude_manifest_paths=args.exclude_manifest,
        clean_population=args.clean_population,
    )
    validate_intervention_atlas_manifest(manifest)
    write_json_atomic(args.output, manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
