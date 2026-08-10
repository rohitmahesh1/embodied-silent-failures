import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from embodied_silent_failures.plan import Trial


def completion_path(output_dir: Path, trial: Trial) -> Path:
    return output_dir / f"task{trial.task_id}--ep{trial.episode_index}.complete.json"


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


def prepare_trial(output_dir: Path, trial: Trial, resume: bool) -> bool:
    """Return True when a completed trial should be skipped."""
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
        return True

    prefix = f"task{trial.task_id}--ep{trial.episode_index}--"
    for path in output_dir.glob(f"{prefix}*"):
        if path.is_file():
            path.unlink()
    return False
