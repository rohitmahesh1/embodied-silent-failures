from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.command_interpolation import (
    INTERIOR_LAMBDAS,
    build_interpolation_plan,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a deterministic command-boundary interpolation canary."
    )
    parser.add_argument("--physical-branches", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--branches-per-stratum", type=int, default=1)
    parser.add_argument("--lambda", action="append", dest="lambdas", type=float)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    plan = build_interpolation_plan(
        args.physical_branches,
        args.campaign_manifest,
        seed=args.seed,
        branches_per_stratum=args.branches_per_stratum,
        lambdas=tuple(args.lambdas or INTERIOR_LAMBDAS),
    )
    write_json_atomic(args.output, plan)
    print(
        json.dumps(
            {
                "experiment": plan["experiment"],
                "branches": len(plan["branches"]),
                "terminal_rollouts": sum(
                    len(branch["lambdas"]) for branch in plan["branches"]
                ),
                "selection": plan["selection"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
