from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.language_campaign import (
    manifest_sha256,
    validate_language_campaign_manifest,
)
from embodied_silent_failures.language_gates import COMMAND_COMPONENTS
from embodied_silent_failures.provenance import file_sha256, load_json


BOUNDARY_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SPLITS = ("development", "holdout")


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected CSV boolean, found {value!r}")


def _command(row: dict[str, str], prefix: str) -> list[float]:
    result = [float(row[f"{prefix}_{component}"]) for component in COMMAND_COMPONENTS]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{prefix} command contains a non-finite value")
    return result


def load_physical_branches(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as file:
        source = list(csv.DictReader(file))
    result = []
    for row in source:
        result.append(
            {
                "physical_run": row["physical_run"],
                "worker_shard": int(row["worker_shard"]),
                "analysis_split": row["analysis_split"],
                "context_id": row["context_id"],
                "task_failure": _boolean(row["task_failure"]),
                "operational_silent_failure": _boolean(
                    row["operational_silent_failure"]
                ),
                "clean_command": _command(row, "clean"),
                "faulted_command": _command(row, "faulted"),
            }
        )
    if not result:
        raise ValueError("physical-branch table is empty")
    return result


def _selection_key(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}:{row['physical_run']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_interpolation_plan(
    branch_path: Path,
    campaign_manifest_path: Path,
    *,
    seed: int,
    branches_per_stratum: int,
    lambdas: tuple[float, ...] = BOUNDARY_LAMBDAS,
) -> dict[str, Any]:
    if branches_per_stratum <= 0:
        raise ValueError("branches per stratum must be positive")
    if not lambdas or len(lambdas) != len(set(lambdas)):
        raise ValueError("interpolation lambdas must be nonempty and unique")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in lambdas):
        raise ValueError("interpolation lambdas must be finite and between zero and one")

    manifest = load_json(campaign_manifest_path)
    validate_language_campaign_manifest(manifest)
    contexts = {
        str(context["context_id"]): context for context in manifest["contexts"]
    }
    branches = load_physical_branches(branch_path)
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch in branches:
        by_context[branch["context_id"]].append(branch)
    mixed_contexts = {
        context_id
        for context_id, values in by_context.items()
        if any(value["task_failure"] for value in values)
        and any(not value["task_failure"] for value in values)
    }

    eligible = []
    exclusions: dict[str, int] = defaultdict(int)
    for branch in branches:
        if not branch["task_failure"]:
            exclusions["prior_branch_succeeded"] += 1
            continue
        if branch["context_id"] not in mixed_contexts:
            exclusions["context_had_no_mixed_faulted_command_outcomes"] += 1
            continue
        if branch["clean_command"][-1] != branch["faulted_command"][-1]:
            exclusions["categorical_gripper_change"] += 1
            continue
        if branch["clean_command"] == branch["faulted_command"]:
            exclusions["command_unchanged"] += 1
            continue
        if branch["context_id"] not in contexts:
            raise ValueError(
                "physical branch context is absent from manifest: "
                f"{branch['context_id']}"
            )
        eligible.append(branch)

    by_stratum: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for branch in eligible:
        by_stratum[(branch["worker_shard"], branch["analysis_split"])].append(
            branch
        )
    expected_strata = [(worker, split) for worker in (0, 1) for split in SPLITS]
    selected = []
    frame = []
    for worker, split in expected_strata:
        values = sorted(
            by_stratum[(worker, split)],
            key=lambda row: (_selection_key(seed, row), row["physical_run"]),
        )
        if len(values) < branches_per_stratum:
            raise ValueError(
                f"stratum worker={worker}, split={split} has only "
                f"{len(values)} branches"
            )
        chosen = values[:branches_per_stratum]
        frame.append(
            {
                "worker_shard": worker,
                "analysis_split": split,
                "eligible_branches": len(values),
                "selected_branches": len(chosen),
            }
        )
        selected.extend(chosen)

    planned = []
    for branch in selected:
        context = contexts[branch["context_id"]]
        if int(context["worker_shard"]) != branch["worker_shard"]:
            raise ValueError(f"worker shard disagrees for {branch['physical_run']}")
        if context["analysis_split"] != branch["analysis_split"]:
            raise ValueError(f"analysis split disagrees for {branch['physical_run']}")
        planned.append(
            {
                **branch,
                "context": context,
                "lambdas": list(lambdas),
                "selection_sha256": _selection_key(seed, branch),
            }
        )

    return {
        "schema_version": 1,
        "experiment": "state-blocked command-boundary interpolation canary",
        "purpose": (
            "Restore one current MuJoCo state per branch, rerun both observed command "
            "endpoints, and test interior points along the same direction for a local "
            "task boundary. This canary does not estimate prevalence."
        ),
        "source": {
            "physical_branches": {
                "path": str(branch_path.resolve()),
                "sha256": file_sha256(branch_path),
            },
            "campaign_manifest": {
                "path": str(campaign_manifest_path.resolve()),
                "file_sha256": file_sha256(campaign_manifest_path),
                "content_sha256": manifest_sha256(manifest),
            },
        },
        "selection": {
            "seed": seed,
            "rule": (
                "Within each worker and prior split, hash the prior failed branches "
                "from mixed-outcome contexts and take the smallest hashes."
            ),
            "requirements": [
                "The original control command succeeded.",
                "The selected faulted command failed.",
                "The context had both successful and failed faulted commands.",
                "Clean and failed commands used the same categorical gripper value.",
            ],
            "branches_per_stratum": branches_per_stratum,
            "frame": frame,
            "exclusions": dict(sorted(exclusions.items())),
        },
        "branches": planned,
    }


def interpolate_command(
    clean: list[float], failed: list[float], interpolation: float
) -> list[float]:
    if len(clean) != len(COMMAND_COMPONENTS) or len(failed) != len(
        COMMAND_COMPONENTS
    ):
        raise ValueError("command interpolation requires seven-dimensional endpoints")
    if clean[-1] != failed[-1]:
        raise ValueError("gripper interpolation is categorical and is not supported")
    if not 0 <= interpolation <= 1:
        raise ValueError("command interpolation must be between zero and one")
    result = [
        reference + interpolation * (target - reference)
        for reference, target in zip(clean, failed, strict=True)
    ]
    result[-1] = clean[-1]
    return result
