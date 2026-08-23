from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record, write_json_atomic
from embodied_silent_failures.pi05_contract import SAFE_REVISION
from embodied_silent_failures.pi05_pair import prepare_pair
from embodied_silent_failures.pi05_safe_data import FEATURE_PROTOCOL
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.provenance import git_dirty, git_revision, load_json


DECISION_WINDOWS = {
    "intervention_decision": 1,
    "within_5_decisions": 5,
    "within_10_decisions": 10,
    "through_terminal_outcome": None,
}


def _monitor_paths(monitor_dir: Path) -> dict[str, Path]:
    return {
        "monitor": monitor_dir / "monitor.json",
        "scores": monitor_dir / "clean_scores.npz",
        "checkpoint": monitor_dir / "artifacts" / "model_final.ckpt",
        "configuration": monitor_dir / "artifacts" / "config.yaml",
        "split_manifest": monitor_dir / "split.json",
    }


def _validate_monitor(monitor_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = _monitor_paths(monitor_dir)
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"frozen pi0.5 monitor is incomplete: {monitor_dir}")
    monitor = load_json(paths["monitor"])
    if monitor.get("model") != "SAFE-MLP":
        raise ValueError("pi0.5 monitor is not SAFE-MLP")
    if monitor.get("feature_protocol") != FEATURE_PROTOCOL:
        raise ValueError("pi0.5 monitor uses a different feature protocol")
    for key in ("checkpoint", "configuration", "split_manifest", "score_archive"):
        path_key = "scores" if key == "score_archive" else key
        if artifact_record(paths[path_key]) != monitor.get(key):
            raise ValueError(f"pi0.5 monitor {key} disagrees with its digest")
    return monitor, paths


def _alarm_summary(
    values: Any,
    band: Any,
    intervention: int,
) -> dict[str, Any]:
    import numpy as np

    if intervention < 0 or intervention >= len(values) or len(band) < len(values):
        raise ValueError("SAFE scores, band, and intervention do not align")
    crossings = np.flatnonzero(values >= band[: len(values)])
    first = int(crossings[0]) if len(crossings) else None
    windows = {}
    for name, horizon in DECISION_WINDOWS.items():
        stop = len(values) if horizon is None else min(len(values), intervention + horizon)
        selected = crossings[(crossings >= intervention) & (crossings < stop)]
        windows[name] = {
            "triggered": bool(len(selected)),
            "first_decision": int(selected[0]) if len(selected) else None,
            "first_environment_step_after_intervention": (
                int((selected[0] - intervention) * 5) if len(selected) else None
            ),
        }
    return {
        "alarm_before_intervention": bool(
            first is not None and first < intervention
        ),
        "first_alarm_decision": first,
        "windows": windows,
    }


def score(
    safe_root: Path,
    monitor_dir: Path,
    pair_dir: Path,
    output_prefix: Path,
) -> dict[str, Any]:
    safe_root = safe_root.resolve()
    monitor_dir = monitor_dir.resolve()
    pair_dir = pair_dir.resolve()
    if git_revision(safe_root) != SAFE_REVISION or git_dirty(safe_root):
        raise RuntimeError(f"SAFE must be clean and pinned at {SAFE_REVISION}")
    monitor, paths = _validate_monitor(monitor_dir)

    sys.path.insert(0, str(safe_root))
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from failure_prob.model import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to score the frozen SAFE-MLP")
    archive = np.load(paths["scores"])
    alphas = archive["alphas"].astype(float)
    bands = archive["bands"].astype(float)
    primary_matches = np.flatnonzero(
        np.isclose(alphas, float(monitor["primary_alpha"]), rtol=0, atol=1e-8)
    )
    if len(primary_matches) != 1:
        raise ValueError("frozen monitor has no unique primary alpha")
    primary_band = bands[int(primary_matches[0])]

    cfg = OmegaConf.load(paths["configuration"])
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    input_dim = int(checkpoint["projector.0.weight"].shape[1])
    model = get_model(cfg, input_dim)
    model.load_state_dict(checkpoint)
    model.to("cuda").eval()

    records = []
    score_values = []
    completions = sorted(pair_dir.glob("pairs/*/pair.complete.json"))
    if not completions:
        raise ValueError(f"no completed pi0.5 pairs found in {pair_dir}")
    for completion_path in completions:
        completion = load_json(completion_path)
        trial = Trial(
            int(completion["task_id"]), int(completion["episode_index"])
        )
        if prepare_pair(pair_dir, trial, True) != "complete":
            raise ValueError(f"pi0.5 pair did not validate: {completion_path}")
        intervention = int(completion["intervention_decision"])
        for label, branch in sorted(completion["branches"].items()):
            pickle_path = completion_path.parent / branch["files"]["pickle"]
            with pickle_path.open("rb") as file:
                payload = pickle.load(file)
            if payload.get("feature_protocol") != FEATURE_PROTOCOL:
                raise ValueError(f"branch uses a different SAFE feature: {pickle_path}")
            features = np.asarray(payload["safe_features"], dtype=np.float32)
            if features.ndim != 2 or features.shape[1] != input_dim:
                raise ValueError(f"branch SAFE features have the wrong shape: {pickle_path}")
            with torch.no_grad():
                values = (
                    model({"features": torch.from_numpy(features)[None].to("cuda")})
                    .squeeze(0)
                    .squeeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            if not np.isfinite(values).all():
                raise ValueError(f"branch SAFE scores are non-finite: {pickle_path}")
            summary = _alarm_summary(values, primary_band, intervention)
            records.append(
                {
                    "task_id": trial.task_id,
                    "episode_index": trial.episode_index,
                    "pair_condition": completion["pair_condition"],
                    "label": label,
                    "success": bool(branch["success"]),
                    "decisions": len(values),
                    "intervention_decision": intervention,
                    "primary_alpha": float(monitor["primary_alpha"]),
                    "alarm": summary,
                }
            )
            score_values.append(values)

    maximum = max(len(values) for values in score_values)
    padded = np.full((len(score_values), maximum), np.nan, dtype=np.float32)
    for index, values in enumerate(score_values):
        padded[index, : len(values)] = values
    scores_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("wb") as file:
        np.savez_compressed(
            file,
            task_ids=np.asarray([item["task_id"] for item in records], dtype=np.int16),
            episode_indices=np.asarray(
                [item["episode_index"] for item in records], dtype=np.int16
            ),
            labels=np.asarray([item["label"] for item in records]),
            lengths=np.asarray([len(value) for value in score_values], dtype=np.int16),
            scores=padded,
        )
    result = {
        "schema_version": 1,
        "analysis": "frozen SAFE-MLP scores for paired pi0.5 camera trials",
        "pair_run": str(pair_dir),
        "monitor": {
            "directory": str(monitor_dir),
            "checkpoint": monitor["checkpoint"],
            "configuration": monitor["configuration"],
            "split_manifest": monitor["split_manifest"],
            "primary_alpha": monitor["primary_alpha"],
        },
        "alarm_rule": "score >= frozen time-varying upper band",
        "decision_windows": {
            name: {
                "policy_decisions": horizon,
                "environment_steps": horizon * 5 if horizon is not None else None,
            }
            for name, horizon in DECISION_WINDOWS.items()
        },
        "score_archive": artifact_record(scores_path),
        "records": records,
    }
    write_json_atomic(json_path, result)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score paired pi0.5 camera trials with frozen SAFE-MLP."
    )
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--pair-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = score(
        args.safe_root, args.monitor_dir, args.pair_dir, args.output_prefix
    )
    print(
        json.dumps(
            {
                "branches": len(result["records"]),
                "failures": sum(not item["success"] for item in result["records"]),
                "output": str(args.output_prefix.with_suffix(".json")),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
