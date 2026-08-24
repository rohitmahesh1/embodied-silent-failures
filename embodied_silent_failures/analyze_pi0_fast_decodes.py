from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi0_fast_contract import ACTION_DIMENSION
from embodied_silent_failures.provenance import file_sha256, load_json


RESHAPE_ERROR = re.compile(
    r"cannot reshape array of size (?P<coefficients>\d+) into shape "
    r"\((?P<action_dimension>\d+)\)"
)


def parse_attempt_log(text: str) -> dict[str, Any]:
    matches = {
        (int(match.group("coefficients")), int(match.group("action_dimension")))
        for match in RESHAPE_ERROR.finditer(text)
    }
    if not matches:
        return {"family": "other", "reshape_signatures": []}
    signatures = [
        {"coefficient_count": count, "action_dimension": dimension}
        for count, dimension in sorted(matches)
    ]
    return {
        "family": "fast_dct_shape" if len(matches) == 1 else "ambiguous",
        "reshape_signatures": signatures,
    }


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("attempt ledger contains no log path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"attempt log escapes campaign directory: {value}")
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"attempt log is missing: {resolved}")
    return resolved


def analyze_campaign(run_dir: Path) -> dict[str, Any]:
    run = load_json(run_dir / "run.json")
    plan_records = run.get("trial_plan")
    if not isinstance(plan_records, list) or not plan_records:
        raise ValueError("campaign run.json has no trial plan")
    planned = {
        (record.get("task_id"), record.get("episode_index"))
        for record in plan_records
        if isinstance(record, dict)
    }
    if len(planned) != len(plan_records) or any(
        type(task_id) is not int or type(episode_index) is not int
        for task_id, episode_index in planned
    ):
        raise ValueError("campaign trial plan contains invalid or duplicate identities")

    complete = {
        (record["task_id"], record["episode_index"])
        for path in sorted(run_dir.glob("*.complete.json"))
        for record in [load_json(path)]
    }
    unresolved_paths = sorted(run_dir.glob("*.unresolved.json"))
    unresolved = {
        (record["task_id"], record["episode_index"]): path
        for path in unresolved_paths
        for record in [load_json(path)]
    }
    if complete & set(unresolved):
        raise ValueError("a campaign identity is both complete and unresolved")
    observed = complete | set(unresolved)
    if observed != planned:
        raise ValueError(
            "terminal campaign identities disagree with the frozen trial plan: "
            f"missing={sorted(planned - observed)}, unexpected={sorted(observed - planned)}"
        )

    records = []
    coefficient_counts: collections.Counter[int] = collections.Counter()
    task_counts: collections.Counter[int] = collections.Counter()
    for task_id, episode_index in sorted(unresolved):
        ledger_path = run_dir / "attempts" / f"task{task_id}--ep{episode_index}.json"
        ledger = load_json(ledger_path)
        attempts_value = ledger.get("attempts")
        if not isinstance(attempts_value, list) or not attempts_value:
            raise ValueError(f"attempt ledger is empty: {ledger_path}")
        attempts = []
        trial_signatures = set()
        for attempt in attempts_value:
            if not isinstance(attempt, dict):
                raise ValueError(f"attempt ledger contains a non-object: {ledger_path}")
            log_path = _relative_file(run_dir, attempt.get("log"))
            parsed = parse_attempt_log(
                log_path.read_text(encoding="utf-8", errors="replace")
            )
            if parsed["family"] == "fast_dct_shape":
                signature = parsed["reshape_signatures"][0]
                trial_signatures.add(
                    (signature["coefficient_count"], signature["action_dimension"])
                )
            attempts.append(
                {
                    "attempt": attempt.get("attempt"),
                    "return_code": attempt.get("return_code"),
                    "log": str(log_path.relative_to(run_dir)),
                    "log_sha256": file_sha256(log_path),
                    **parsed,
                }
            )

        reproducible = (
            len(trial_signatures) == 1
            and len(attempts) >= 2
            and all(item["family"] == "fast_dct_shape" for item in attempts)
        )
        coefficient_count = None
        action_dimension = None
        if len(trial_signatures) == 1:
            coefficient_count, action_dimension = next(iter(trial_signatures))
            coefficient_counts[coefficient_count] += 1
        task_counts[task_id] += 1

        heartbeat_path = (
            run_dir / "heartbeats" / f"task{task_id}--ep{episode_index}.json"
        )
        heartbeat = load_json(heartbeat_path) if heartbeat_path.is_file() else None
        records.append(
            {
                "task_id": task_id,
                "episode_index": episode_index,
                "attempt_count": len(attempts),
                "all_attempts_same_shape_failure": reproducible,
                "coefficient_count": coefficient_count,
                "action_dimension": action_dimension,
                "coefficient_count_divisible_by_action_dimension": (
                    coefficient_count % action_dimension == 0
                    if coefficient_count is not None and action_dimension
                    else None
                ),
                "last_successful_decision": (
                    {
                        "environment_step": heartbeat.get("environment_step"),
                        "model_decisions": heartbeat.get("model_decisions"),
                    }
                    if heartbeat is not None
                    else None
                ),
                "attempts": attempts,
            }
        )

    affected_tasks = sorted(task_counts)
    reproducible_tasks = sorted(
        {
            record["task_id"]
            for record in records
            if record["all_attempts_same_shape_failure"]
        }
    )
    audit_trials = [
        min(
            (
                record
                for record in records
                if record["task_id"] == task_id
                and record["all_attempts_same_shape_failure"]
            ),
            key=lambda record: record["episode_index"],
        )
        for task_id in reproducible_tasks
    ]
    all_shape_failures = all(
        record["coefficient_count"] is not None for record in records
    )
    all_expected_dimension = all(
        record["action_dimension"] == ACTION_DIMENSION for record in records
    )
    return {
        "schema_version": 1,
        "source_campaign": {
            "directory": str(run_dir.resolve()),
            "run_json_sha256": file_sha256(run_dir / "run.json"),
            "experiment_revision": run.get("repository_states", {})
            .get("experiment_code", {})
            .get("revision"),
        },
        "terminal_accounting": {
            "planned": len(planned),
            "complete": len(complete),
            "unresolved": len(unresolved),
            "all_planned_trials_have_one_terminal_state": True,
        },
        "summary": {
            "all_unresolved_are_fast_dct_shape_failures": (
                all_shape_failures and all_expected_dimension
            ),
            "all_shape_failures_have_nondivisible_coefficient_counts": all(
                record["coefficient_count_divisible_by_action_dimension"] is False
                for record in records
            ),
            "same_failure_repeated_on_at_least_two_attempts": sum(
                record["all_attempts_same_shape_failure"] for record in records
            ),
            "unresolved_fraction_of_plan": len(unresolved) / len(planned),
            "affected_tasks": affected_tasks,
            "tasks_with_repeated_shape_failures": reproducible_tasks,
            "unresolved_by_task": {
                str(key): value for key, value in sorted(task_counts.items())
            },
            "coefficient_count_distribution": {
                str(key): value for key, value in sorted(coefficient_counts.items())
            },
        },
        "audit_selection": {
            "rule": (
                "For each task with a repeated FAST DCT shape failure, select the "
                "lowest episode index."
            ),
            "purpose": (
                "Attribute the malformed output across task strata; this cohort is "
                "not used to estimate prevalence."
            ),
            "trials": [
                {
                    "task_id": record["task_id"],
                    "episode_index": record["episode_index"],
                    "baseline_coefficient_count": record["coefficient_count"],
                }
                for record in audit_trials
            ],
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit malformed pi0-FAST decode records from one frozen campaign."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    result = analyze_campaign(args.run_dir)
    write_json_atomic(args.output, result)
    if args.manifest_output is not None:
        write_json_atomic(
            args.manifest_output,
            {
                "schema_version": 1,
                "selection_rule": result["audit_selection"]["rule"],
                "source_campaign_run_json_sha256": result["source_campaign"][
                    "run_json_sha256"
                ],
                "trials": [
                    {
                        "task_id": record["task_id"],
                        "episode_index": record["episode_index"],
                    }
                    for record in result["audit_selection"]["trials"]
                ],
            },
        )


if __name__ == "__main__":
    main()
