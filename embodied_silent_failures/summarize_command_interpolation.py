from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.command_interpolation_analysis import (
    branch_boundary_summary,
)
from embodied_silent_failures.command_interpolation_worker import CONDITION
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine command-boundary interpolation canary results."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    return parser.parse_args()


def _plan_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {str(branch["physical_run"]): branch for branch in plan["branches"]}
    if len(result) != len(plan["branches"]):
        raise ValueError("interpolation plan repeats a physical branch")
    return result


def main() -> None:
    args = _arguments()
    plan = load_json(args.plan)
    planned = _plan_index(plan)
    records = []
    run_artifacts = []
    for run_dir in args.run_dir:
        run_path = run_dir / "run.json"
        status_path = run_dir / "status.json"
        run = load_json(run_path)
        if run.get("condition") != CONDITION:
            raise ValueError(f"run has another condition: {run_dir}")
        run_artifacts.append(
            {
                "directory": str(run_dir.resolve()),
                "run_sha256": file_sha256(run_path),
                "status_sha256": (
                    file_sha256(status_path) if status_path.is_file() else None
                ),
            }
        )
        for path in sorted(run_dir.glob("attempts/*/lambda-*/*.complete.json")):
            result = load_json(path)
            if (
                result.get("status") != "complete"
                or result.get("condition") != CONDITION
            ):
                raise ValueError(f"invalid interpolation completion: {path}")
            fault = result["fault"]
            physical_run = str(fault["source_physical_run"])
            if physical_run not in planned:
                raise ValueError(f"result is absent from plan: {physical_run}")
            branch = planned[physical_run]
            replay = result["context_replay"]
            records.append(
                {
                    "physical_run": physical_run,
                    "context_id": branch["context_id"],
                    "worker_shard": int(branch["worker_shard"]),
                    "analysis_split": branch["analysis_split"],
                    "task_id": int(branch["context"]["task_id"]),
                    "episode_index": int(branch["context"]["episode_index"]),
                    "phase": branch["context"]["phase"],
                    "policy_step": int(branch["context"]["policy_step"]),
                    "interpolation": float(fault["interpolation"]),
                    "success": bool(result["success"]),
                    "policy_steps": int(result["policy_steps"]),
                    "rollout_seconds": float(result["rollout_seconds"]),
                    "context_replay_state_exact": bool(
                        replay["simulator_state_exact_equal"]
                    ),
                    "context_replay_state_linf": float(replay["simulator_state_linf"]),
                    "recapture_used_archived_prefix": "source_prefix" in fault,
                    "maximum_archived_clean_command_error": float(
                        fault["maximum_archived_clean_command_error"]
                    ),
                    "experiment_revision": result["execution"]["experiment_code"][
                        "revision"
                    ],
                    "completion_sha256": file_sha256(path),
                }
            )

    keys = [(record["physical_run"], record["interpolation"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("run directories contain duplicate interpolation results")
    expected = {
        (str(branch["physical_run"]), float(value))
        for branch in plan["branches"]
        for value in branch["lambdas"]
    }
    observed = set(keys)
    by_lambda = {}
    for interpolation in sorted({value for _run, value in expected}):
        values = [
            record for record in records if record["interpolation"] == interpolation
        ]
        by_lambda[format(interpolation, "g")] = {
            "completed": len(values),
            "successes": sum(record["success"] for record in values),
        }
    revisions = Counter(record["experiment_revision"] for record in records)
    output = {
        "schema_version": 1,
        "analysis": "state-blocked command-boundary interpolation canary",
        "status": "implementation and boundary-shape canary; not a prevalence estimate",
        "source_plan": {
            "path": str(args.plan.resolve()),
            "sha256": file_sha256(args.plan),
        },
        "run_artifacts": run_artifacts,
        "coverage": {
            "planned": len(expected),
            "completed": len(observed),
            "missing": [
                {"physical_run": run, "interpolation": interpolation}
                for run, interpolation in sorted(expected - observed)
            ],
        },
        "experiment_revisions": dict(sorted(revisions.items())),
        "outcomes_by_interpolation": by_lambda,
        "boundary_shape": branch_boundary_summary(records),
    }
    write_json_atomic(args.output, output)
    columns = sorted({key for record in records for key in record})
    write_csv_atomic(
        args.records_csv,
        [{key: record.get(key, "") for key in columns} for record in records],
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
