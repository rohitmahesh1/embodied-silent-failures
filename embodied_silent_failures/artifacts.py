import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from embodied_silent_failures.plan import Trial


def completion_path(output_dir: Path, trial: Trial) -> Path:
    return output_dir / f"task{trial.task_id}--ep{trial.episode_index}.complete.json"


def exclusion_path(output_dir: Path, trial: Trial) -> Path:
    return output_dir / f"task{trial.task_id}--ep{trial.episode_index}.excluded.json"


def safe_stem(trial: Trial, success: bool) -> str:
    return f"task{trial.task_id}--ep{trial.episode_index}--succ{int(success)}"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_trial(
    output_dir: Path, trial: Trial, resume: bool
) -> Literal["complete", "excluded"] | None:
    """Return the terminal state when a trial should be skipped."""
    marker = completion_path(output_dir, trial)
    if marker.exists():
        if not resume:
            raise FileExistsError(
                f"trial {trial.task_id}/{trial.episode_index} is already complete; "
                "pass --resume to skip completed trials"
            )
        with marker.open("r", encoding="utf-8") as file:
            result = json.load(file)
        if result.get("status") != "complete":
            raise ValueError(f"invalid completion marker: {marker}")
        if result.get("task_id") != trial.task_id:
            raise ValueError(f"completion marker has the wrong task ID: {marker}")
        if result.get("episode_index") != trial.episode_index:
            raise ValueError(f"completion marker has the wrong episode index: {marker}")
        files = result.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"completion marker has no artifact list: {marker}")
        required = [files.get("csv"), files.get("pickle")]
        if files.get("video") is not None:
            required.append(files["video"])
        missing = [
            name
            for name in required
            if not name or not (output_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"completion marker references missing artifacts {missing}: {marker}"
            )
        if exclusion_path(output_dir, trial).exists():
            raise ValueError(f"trial has both completion and exclusion markers: {trial}")
        return "complete"

    marker = exclusion_path(output_dir, trial)
    if marker.exists():
        if not resume:
            raise FileExistsError(
                f"trial {trial.task_id}/{trial.episode_index} is already excluded; "
                "pass --resume to skip excluded trials"
            )
        with marker.open("r", encoding="utf-8") as file:
            result = json.load(file)
        if result.get("status") != "excluded":
            raise ValueError(f"invalid exclusion marker: {marker}")
        if result.get("task_id") != trial.task_id:
            raise ValueError(f"exclusion marker has the wrong task ID: {marker}")
        if result.get("episode_index") != trial.episode_index:
            raise ValueError(f"exclusion marker has the wrong episode index: {marker}")
        return "excluded"

    prefix = f"task{trial.task_id}--ep{trial.episode_index}--"
    for pattern in (f"{prefix}*", f".{prefix}*"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    return None
