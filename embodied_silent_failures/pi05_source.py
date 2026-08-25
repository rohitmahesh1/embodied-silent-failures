from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Tuple

from embodied_silent_failures.provenance import load_json


SourceRun = Tuple[Path, Path, Dict[str, Any]]
CompletionRecord = Tuple[Path, Path, Dict[str, Any]]


def validated_clean_runs(
    run_dirs: Path | Sequence[Path], expected_replan_steps: int
) -> list[SourceRun]:
    values = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    if not values:
        raise ValueError("at least one source run is required")
    resolved = sorted({Path(value).resolve() for value in values}, key=str)
    if len(resolved) != len(values):
        raise ValueError("source run list contains a duplicate directory")

    sources = []
    for run_dir in resolved:
        run_path = run_dir / "run.json"
        run = load_json(run_path)
        configuration = run.get("configuration", {})
        if configuration.get("model") != "pi0.5" or run.get("condition") != "clean":
            raise ValueError(f"source run is not a clean pi0.5 campaign: {run_dir}")
        if int(configuration.get("replan_steps", -1)) != expected_replan_steps:
            raise ValueError(
                "source replan_steps does not match the requested data contract"
            )
        sources.append((run_dir, run_path, run))

    reference = sources[0][2]
    for run_dir, _run_path, run in sources[1:]:
        if run.get("configuration") != reference.get("configuration"):
            raise ValueError(f"source run configuration differs: {run_dir}")
        if run.get("repository_states") != reference.get("repository_states"):
            raise ValueError(f"source repository revisions differ: {run_dir}")
    return sources


def clean_completion_records(sources: list[SourceRun]) -> list[CompletionRecord]:
    records = []
    for run_dir, _run_path, _run in sources:
        for path in sorted(run_dir.glob("*.complete.json")):
            value = load_json(path)
            if value.get("status") != "complete" or value.get("condition") != "clean":
                raise ValueError(f"unexpected completion marker: {path}")
            if value.get("model") != "pi0.5":
                raise ValueError(f"completion marker is not for pi0.5: {path}")
            records.append((run_dir, path, value))
    if not records:
        raise ValueError("no completed clean pi0.5 rollouts found in source runs")
    records.sort(key=lambda item: (item[2]["task_id"], item[2]["episode_index"]))
    identities = [
        (int(value["task_id"]), int(value["episode_index"]))
        for _run_dir, _path, value in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("source runs contain duplicate task/episode identities")
    return records
