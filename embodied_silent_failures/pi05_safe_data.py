from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    temporary_path,
    write_json_atomic,
)
from embodied_silent_failures.pi05_contract import DEFAULT_REPLAN_STEPS, SAFE_REVISION
from embodied_silent_failures.provenance import file_sha256, load_json


FEATURE_PROTOCOL = "safe-pi0-pre-velocity-first-horizon-final-diffusion-v1"


def reduce_pre_velocity(pre_velocity: Any, np: Any) -> Any:
    """Select the pi0 feature used by the published SAFE-MLP configuration."""
    values = np.asarray(pre_velocity)
    if values.ndim != 4:
        raise ValueError(
            "pre_velocity must have decision, diffusion, horizon, and feature axes"
        )
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("pre_velocity axes must be nonempty")

    # SAFE b6036ab, failure_prob/data/pizero.py::load_rollouts_from_root and
    # Table 10 of arXiv:2506.09937v2: apply horizon=First, then diffusion=Last.
    # SAFE's process_tensor_idx_rel maps those relative indices to axis -2.
    selected = values[:, -1, 0, :]
    if selected.ndim != 2 or selected.shape[1] == 0:
        raise ValueError("SAFE feature reduction produced an invalid shape")
    if not np.isfinite(selected).all():
        raise ValueError("SAFE feature reduction produced a non-finite value")
    return np.ascontiguousarray(selected, dtype=np.float32)


def _completion_records(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in sorted(run_dir.glob("*.complete.json")):
        value = load_json(path)
        if value.get("status") != "complete" or value.get("condition") != "clean":
            raise ValueError(f"unexpected completion marker: {path}")
        if value.get("model") != "pi0.5":
            raise ValueError(f"completion marker is not for pi0.5: {path}")
        records.append((path, value))
    if not records:
        raise ValueError(f"no completed clean pi0.5 rollouts found in {run_dir}")
    records.sort(key=lambda item: (item[1]["task_id"], item[1]["episode_index"]))
    identities = [
        (int(value["task_id"]), int(value["episode_index"])) for _, value in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("clean pi0.5 run contains duplicate task/episode identities")
    return records


def _pickle_artifact(
    run_dir: Path, marker: Path, completion: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    files = completion.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("pickle"), str):
        raise ValueError(f"completion marker does not identify a pickle: {marker}")
    name = files["pickle"]
    path = run_dir / name
    if path.name != name or not path.is_file():
        raise FileNotFoundError(f"completion marker references no pickle: {path}")
    matches = [
        item
        for item in completion.get("artifact_manifest", [])
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"completion marker has no unique pickle digest: {marker}")
    return path, matches[0]


def export_features(
    run_dir: Path,
    output_dir: Path,
    *,
    expected_replan_steps: int = DEFAULT_REPLAN_STEPS,
    verify_source_digests: bool = True,
) -> dict[str, Any]:
    import numpy as np

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    run_path = run_dir / "run.json"
    run = load_json(run_path)
    configuration = run.get("configuration", {})
    if configuration.get("model") != "pi0.5" or run.get("condition") != "clean":
        raise ValueError("source run is not a clean pi0.5 campaign")
    if int(configuration.get("replan_steps", -1)) != expected_replan_steps:
        raise ValueError(
            "source replan_steps does not match the requested SAFE data contract"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"feature output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    offsets = [0]
    source_records = []
    task_ids = []
    episode_indices = []
    successes = []
    for marker, completion in _completion_records(run_dir):
        if int(completion.get("replan_steps", -1)) != expected_replan_steps:
            raise ValueError(f"rollout has a different replan_steps value: {marker}")
        pickle_path, expected_artifact = _pickle_artifact(run_dir, marker, completion)
        if verify_source_digests:
            actual_artifact = artifact_record(pickle_path)
            if actual_artifact != expected_artifact:
                raise ValueError(f"rollout pickle disagrees with its digest: {pickle_path}")
        with pickle_path.open("rb") as file:
            payload = pickle.load(file)
        identity = (int(completion["task_id"]), int(completion["episode_index"]))
        expected = {
            "model": "pi0.5",
            "condition": "clean",
            "task_id": identity[0],
            "episode_idx": identity[1],
            "episode_success": bool(completion["success"]),
            "replan_steps": expected_replan_steps,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"rollout payload disagrees on {key}: {pickle_path}")
        decisions = payload.get("decisions")
        if not isinstance(decisions, dict) or "pre_velocity" not in decisions:
            raise ValueError(f"rollout has no pre_velocity evidence: {pickle_path}")
        selected = reduce_pre_velocity(decisions["pre_velocity"], np)
        if len(selected) != int(completion["model_decisions"]):
            raise ValueError(f"rollout decision count disagrees: {pickle_path}")

        features.append(selected)
        offsets.append(offsets[-1] + len(selected))
        task_ids.append(identity[0])
        episode_indices.append(identity[1])
        successes.append(bool(completion["success"]))
        source_records.append(
            {
                "task_id": identity[0],
                "episode_index": identity[1],
                "success": bool(completion["success"]),
                "decisions": len(selected),
                "completion": marker.name,
                "completion_sha256": file_sha256(marker),
                "pickle": expected_artifact,
            }
        )

    widths = {int(value.shape[1]) for value in features}
    if len(widths) != 1:
        raise ValueError(f"rollouts do not share one SAFE feature width: {widths}")
    feature_width = widths.pop()
    archive_path = output_dir / "features.npz"
    pending = temporary_path(archive_path)
    try:
        with pending.open("wb") as file:
            np.savez(
                file,
                features=np.concatenate(features, axis=0),
                offsets=np.asarray(offsets, dtype=np.int64),
                task_ids=np.asarray(task_ids, dtype=np.int16),
                episode_indices=np.asarray(episode_indices, dtype=np.int16),
                successes=np.asarray(successes, dtype=bool),
            )
        pending.replace(archive_path)
    finally:
        pending.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "feature_protocol": FEATURE_PROTOCOL,
        "feature": {
            "source": "pre_velocity",
            "horizon": "first",
            "diffusion_step": "last",
            "width": feature_width,
            "safe_revision": SAFE_REVISION,
            "paper": "SAFE arXiv:2506.09937v2 Table 10",
            "implementation": (
                "SAFE b6036ab failure_prob/data/pizero.py::"
                "load_rollouts_from_root"
            ),
        },
        "source": {
            "run_dir": str(run_dir),
            "run_json_sha256": file_sha256(run_path),
            "experiment_revision": run.get("repository_states", {})
            .get("experiment_code", {})
            .get("revision"),
            "replan_steps": expected_replan_steps,
            "source_digests_verified": verify_source_digests,
        },
        "rollouts": len(features),
        "decisions": offsets[-1],
        "successes": sum(successes),
        "failures": len(successes) - sum(successes),
        "archive": artifact_record(archive_path),
        "source_rollouts": source_records,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce clean pi0.5 rollouts to the feature used by SAFE-MLP."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-replan-steps", type=int, default=DEFAULT_REPLAN_STEPS
    )
    parser.add_argument(
        "--verify-source-digests", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = export_features(
        args.run_dir,
        args.output_dir,
        expected_replan_steps=args.expected_replan_steps,
        verify_source_digests=args.verify_source_digests,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
