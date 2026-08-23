from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi05_contract import DEFAULT_REPLAN_STEPS
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import file_sha256, load_json


SELECTION_PROTOCOL = "pi05-stale-sites-task-stratified-rollout-uniform-v1"


@dataclass(frozen=True)
class Pi05StaleSpec:
    trial: Trial
    intervention_decision: int
    clean_decisions: int
    replan_steps: int
    order_bit: int

    @property
    def source_decision(self) -> int:
        return self.intervention_decision - 1

    @property
    def intervention_environment_step(self) -> int:
        return self.intervention_decision * self.replan_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.trial.to_dict(),
            "intervention_decision": self.intervention_decision,
            "source_decision": self.source_decision,
            "clean_decisions": self.clean_decisions,
            "replan_steps": self.replan_steps,
            "intervention_environment_step": self.intervention_environment_step,
            "stale_age": {
                "policy_decisions": 1,
                "environment_steps": self.replan_steps,
            },
            "order_bit": self.order_bit,
        }


@dataclass(frozen=True)
class Pi05StaleManifest:
    seed: int
    source_run_json_sha256: str
    specs: dict[Trial, Pi05StaleSpec]


def _completed_successes(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("*.complete.json")):
        value = load_json(path)
        if value.get("status") != "complete" or value.get("condition") != "clean":
            raise ValueError(f"unexpected completion marker: {path}")
        if value.get("model") != "pi0.5":
            raise ValueError(f"completion marker is not for pi0.5: {path}")
        if value.get("success") is True and int(value.get("model_decisions", 0)) > 1:
            value = {**value, "_completion": path.name}
            by_task.setdefault(int(value["task_id"]), []).append(value)
    for values in by_task.values():
        values.sort(key=lambda item: int(item["episode_index"]))
    return by_task


def build_manifest(
    run_dir: Path,
    output: Path,
    *,
    per_task: int,
    seed: int,
) -> dict[str, Any]:
    if per_task <= 0 or seed < 0:
        raise ValueError("per-task count must be positive and seed must be non-negative")
    run_dir = run_dir.resolve()
    run_path = run_dir / "run.json"
    run = load_json(run_path)
    configuration = run.get("configuration", {})
    replan_steps = int(configuration.get("replan_steps", -1))
    if (
        run.get("condition") != "clean"
        or configuration.get("model") != "pi0.5"
        or replan_steps != DEFAULT_REPLAN_STEPS
    ):
        raise ValueError("selection requires the native replan_steps=5 pi0.5 baseline")

    candidates = _completed_successes(run_dir)
    task_ids = sorted(candidates)
    if task_ids != list(range(10)):
        raise ValueError(f"selection population does not contain tasks 0-9: {task_ids}")
    short = {task: len(values) for task, values in candidates.items() if len(values) < per_task}
    if short:
        raise ValueError(f"tasks have too few successful rollouts: {short}")

    rng = random.Random(seed)
    selected = []
    for task_id in task_ids:
        for completion in rng.sample(candidates[task_id], per_task):
            decisions = int(completion["model_decisions"])
            selected.append(
                {
                    "trial": Trial(task_id, int(completion["episode_index"])),
                    "intervention_decision": rng.randrange(1, decisions),
                    "clean_decisions": decisions,
                    "completion": completion["_completion"],
                }
            )
    selected.sort(key=lambda item: item["trial"])
    order_bits = [index % 2 for index in range(len(selected))]
    rng.shuffle(order_bits)

    trials = []
    for item, order_bit in zip(selected, order_bits):
        spec = Pi05StaleSpec(
            trial=item["trial"],
            intervention_decision=item["intervention_decision"],
            clean_decisions=item["clean_decisions"],
            replan_steps=replan_steps,
            order_bit=order_bit,
        )
        trials.append(
            {
                **spec.to_dict(),
                "source_completion": item["completion"],
                "source_completion_sha256": file_sha256(
                    run_dir / item["completion"]
                ),
            }
        )

    result = {
        "schema_version": 1,
        "selection_protocol": SELECTION_PROTOCOL,
        "seed": seed,
        "population": {
            "source": "completed successful clean pi0.5 rollouts",
            "tasks": task_ids,
            "eligible_by_task": {
                str(task): len(candidates[task]) for task in task_ids
            },
            "behavior_used": "terminal success and rollout length only",
        },
        "sampling": {
            "rollouts": "uniform without replacement within each task",
            "decisions": "uniform from decision 1 through the penultimate boundary",
            "per_task": per_task,
            "total": len(trials),
            "order_bits": {
                "zero": order_bits.count(0),
                "one": order_bits.count(1),
            },
        },
        "source": {
            "run_dir": str(run_dir),
            "run_json_sha256": file_sha256(run_path),
            "experiment_revision": run.get("repository_states", {})
            .get("experiment_code", {})
            .get("revision"),
            "replan_steps": replan_steps,
        },
        "trials": trials,
    }
    write_json_atomic(output, result)
    return result


def load_manifest(path: Path) -> Pi05StaleManifest:
    value = load_json(path)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported pi0.5 stale manifest schema")
    if value.get("selection_protocol") != SELECTION_PROTOCOL:
        raise ValueError("unexpected pi0.5 stale selection protocol")
    seed = value.get("seed")
    source = value.get("source")
    trials = value.get("trials")
    if type(seed) is not int or seed < 0 or not isinstance(source, dict):
        raise ValueError("pi0.5 stale manifest has invalid metadata")
    if not isinstance(trials, list) or not trials:
        raise ValueError("pi0.5 stale manifest has no trials")

    specs = {}
    for item in trials:
        if not isinstance(item, dict):
            raise ValueError("pi0.5 stale manifest trial is not an object")
        trial = Trial(int(item["task_id"]), int(item["episode_index"]))
        spec = Pi05StaleSpec(
            trial=trial,
            intervention_decision=int(item["intervention_decision"]),
            clean_decisions=int(item["clean_decisions"]),
            replan_steps=int(item["replan_steps"]),
            order_bit=int(item["order_bit"]),
        )
        if trial in specs:
            raise ValueError(f"duplicate pi0.5 stale manifest trial: {trial}")
        if (
            spec.intervention_decision <= 0
            or spec.intervention_decision >= spec.clean_decisions
            or spec.replan_steps != DEFAULT_REPLAN_STEPS
            or spec.order_bit not in (0, 1)
            or int(item.get("source_decision", -1)) != spec.source_decision
            or int(item.get("intervention_environment_step", -1))
            != spec.intervention_environment_step
        ):
            raise ValueError(f"invalid pi0.5 stale manifest trial: {trial}")
        specs[trial] = spec
    digest = source.get("run_json_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("pi0.5 stale manifest has no source run digest")
    return Pi05StaleManifest(seed=seed, source_run_json_sha256=digest, specs=specs)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predeclare task-stratified stale-camera sites for pi0.5."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = build_manifest(
        args.run_dir, args.output, per_task=args.per_task, seed=args.seed
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
