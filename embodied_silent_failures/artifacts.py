from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
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


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty rollout log")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = temporary_path(path)
    try:
        with pending.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def write_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = temporary_path(path)
    try:
        with pending.open("wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def write_npz_atomic(path: Path, np: Any, arrays: dict[str, Any]) -> None:
    if not arrays:
        raise ValueError("cannot write an empty NumPy archive")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = temporary_path(path)
    try:
        with pending.open("wb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def artifact_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _validate_artifact_manifest(
    output_dir: Path, marker: Path, result: dict[str, Any]
) -> None:
    manifest = result.get("artifact_manifest")
    if manifest is None:
        return
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(
            f"completion marker has an invalid artifact manifest: {marker}"
        )
    for record in manifest:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError(
                f"completion marker has an invalid artifact record: {marker}"
            )
        path = output_dir / record["name"]
        if path.name != record["name"] or not path.is_file():
            raise FileNotFoundError(
                f"completion marker references a missing artifact: {path}"
            )
        actual = artifact_record(path)
        if actual != record:
            raise ValueError(
                f"artifact size or digest disagrees with completion marker: {path}"
            )


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
        _validate_artifact_manifest(output_dir, marker, result)
        evidence = result.get("evidence_graph")
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValueError(f"completion marker has invalid evidence metadata: {marker}")
            evidence_dir = Path(str(evidence.get("directory", "")))
            relative = evidence.get("directory_relative_to_run")
            if not evidence_dir.is_dir() and isinstance(relative, str):
                evidence_dir = (output_dir / relative).resolve()
            evidence_files = (
                "raw.jsonl",
                "annotations.json",
                "graph.json",
                "audit.json",
                "composition.json",
            )
            missing_evidence = [
                name for name in evidence_files if not (evidence_dir / name).is_file()
            ]
            if missing_evidence:
                raise FileNotFoundError(
                    "completion marker references missing evidence artifacts "
                    f"{missing_evidence}: {marker}"
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
