import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from embodied_silent_failures.analysis import Alarm, TREATMENT_CONDITIONS
from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.rollout import attach_monitor_timeline


SAFE_REVISION = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
ALARM_WINDOWS = {
    "post_fault_any": None,
    "within_5_steps": 5,
    "within_10_steps": 10,
    "within_25_steps": 25,
}
SAFE_MONITOR_KINDS = {
    "indep": "safe_mlp",
    "lstm": "safe_lstm",
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score OpenVLA fault rollouts with a frozen SAFE monitor."
    )
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument(
        "--run-dir", required=True, action="append", dest="run_dirs", type=Path
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def _monitor_kind(model_name: str) -> str:
    try:
        return SAFE_MONITOR_KINDS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported SAFE monitor model: {model_name}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _completion_results(run_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    results = {}
    for path in sorted(run_dir.glob("*.complete.json")):
        result = _load_json(path)
        if result.get("status") != "complete":
            raise ValueError(f"fault result is not complete: {path}")
        if result.get("condition") not in TREATMENT_CONDITIONS:
            raise ValueError(f"result is not a supported intervention rollout: {path}")
        key = (int(result["task_id"]), int(result["episode_index"]))
        if key in results:
            raise ValueError(f"duplicate completion result for {key} in {run_dir}")
        results[key] = result
    if not results:
        raise ValueError(f"no completed fault rollouts found in {run_dir}")
    return results


def alarm_windows(
    scores: Sequence[float],
    alphas: Sequence[float],
    bands: Sequence[Sequence[float]],
    fault_step: int,
    nonfinite_is_alarm: bool = False,
) -> dict[str, dict[str, dict[str, int | bool | None]]]:
    if len(alphas) != len(bands):
        raise ValueError("each monitor alpha must have one threshold band")
    if fault_step < 0 or fault_step >= len(scores):
        raise ValueError("fault step must fall within the monitor scores")

    result = {}
    for alpha, band in zip(alphas, bands):
        windows = {}
        for name, horizon in ALARM_WINDOWS.items():
            stop_step = (
                len(scores)
                if horizon is None
                else min(len(scores), fault_step + horizon)
            )
            alarm = _alarm_from_band(
                scores,
                band,
                start_step=fault_step,
                stop_step=stop_step,
                nonfinite_is_alarm=nonfinite_is_alarm,
            )
            windows[name] = {
                "triggered": alarm.triggered,
                "first_step": alarm.first_step,
            }
        result[format(alpha, "g")] = windows
    return result


def _alarm_from_band(
    scores: Sequence[float],
    thresholds: Sequence[float],
    start_step: int,
    stop_step: int,
    nonfinite_is_alarm: bool,
) -> Alarm:
    if start_step < 0 or stop_step <= start_step or stop_step > len(scores):
        raise ValueError("alarm window must be a nonempty range within the scores")
    if len(thresholds) < stop_step:
        raise ValueError("monitor threshold band is shorter than the alarm window")

    for step in range(start_step, stop_step):
        score = scores[step]
        threshold = thresholds[step]
        if not math.isfinite(threshold):
            raise ValueError(f"monitor threshold at step {step} is not finite")
        if nonfinite_is_alarm and not math.isfinite(score):
            return Alarm(triggered=True, first_step=step)
        if score >= threshold:
            return Alarm(triggered=True, first_step=step)
    return Alarm(triggered=False, first_step=None)


def _validate_monitor(monitor_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "monitor": monitor_dir / "monitor.json",
        "scores": monitor_dir / "clean_scores.npz",
        "checkpoint": monitor_dir / "artifacts" / "model_final.ckpt",
        "configuration": monitor_dir / "artifacts" / "config.yaml",
        "split_manifest": monitor_dir / "split.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"monitor {name} is not a file: {path}")

    monitor = _load_json(paths["monitor"])
    expected_hashes = {
        "checkpoint": monitor["checkpoint"]["sha256"],
        "configuration": monitor["configuration"]["sha256"],
        "split_manifest": monitor["split_manifest"]["sha256"],
    }
    for name, expected in expected_hashes.items():
        actual = _sha256(paths[name])
        if actual != expected:
            raise ValueError(
                f"monitor {name} hash is {actual}, expected frozen hash {expected}"
            )
    return monitor, paths


def main() -> None:
    args = _parse_arguments()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    project_root = Path(__file__).resolve().parents[1]
    if _git_revision(args.safe_root) != SAFE_REVISION:
        raise RuntimeError(f"SAFE must be checked out at {SAFE_REVISION}")
    for name, path in (("experiment code", project_root), ("SAFE", args.safe_root)):
        if _git_dirty(path):
            raise RuntimeError(f"{name} has uncommitted changes: {path}")
    if not args.run_dirs:
        raise ValueError("at least one fault run directory is required")
    labels = [path.name for path in args.run_dirs]
    if len(labels) != len(set(labels)):
        raise ValueError("fault run directories must have distinct names")
    run_dirs_by_label = dict(zip(labels, args.run_dirs))

    monitor, monitor_paths = _validate_monitor(args.monitor_dir)

    sys.path.insert(0, str(args.safe_root.resolve()))
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from failure_prob.data import openvla
    from failure_prob.model import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to score SAFE fault traces")

    score_archive = np.load(monitor_paths["scores"])
    alphas = score_archive["alphas"].astype(float).tolist()
    bands = score_archive["bands"].astype(float)
    if bands.ndim != 2 or len(alphas) != bands.shape[0]:
        raise ValueError("frozen monitor bands have invalid dimensions")
    if not np.isfinite(bands).all():
        raise ValueError("frozen monitor bands contain non-finite values")

    cfg = OmegaConf.load(monitor_paths["configuration"])
    safe_model_name = str(cfg.model.name)
    monitor_kind = _monitor_kind(safe_model_name)
    indexed_rollouts = []
    for label, run_dir in zip(labels, args.run_dirs):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"fault run directory does not exist: {run_dir}")
        run = _load_json(run_dir / "run.json")
        if run.get("condition") not in TREATMENT_CONDITIONS:
            raise ValueError(
                f"run metadata is not for a supported intervention: {run_dir}"
            )
        completions = _completion_results(run_dir)

        cfg.dataset.data_path = f"{run_dir.resolve()}/"
        rollouts = sorted(
            openvla.load_rollouts(cfg), key=lambda item: (item.task_id, item.episode_idx)
        )
        if len(rollouts) != len(completions):
            raise ValueError(
                f"{run_dir} has {len(rollouts)} SAFE traces but "
                f"{len(completions)} completion records"
            )
        for rollout in rollouts:
            key = (int(rollout.task_id), int(rollout.episode_idx))
            completion = completions[key]
            if bool(rollout.episode_success) != bool(completion["success"]):
                raise ValueError(f"SAFE trace label disagrees with result for {label}/{key}")
            if len(rollout.hidden_states) != int(completion["policy_steps"]):
                raise ValueError(f"SAFE trace length disagrees with result for {label}/{key}")
            indexed_rollouts.append((label, rollout, completion))

    indexed_rollouts.sort(key=lambda item: (item[0], item[1].task_id, item[1].episode_idx))
    input_dim = int(indexed_rollouts[0][1].hidden_states.shape[-1])
    if any(int(item[1].hidden_states.shape[-1]) != input_dim for item in indexed_rollouts):
        raise ValueError("fault traces do not share one SAFE feature dimension")

    model = get_model(cfg, input_dim)
    state_dict = torch.load(monitor_paths["checkpoint"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()

    scores = []
    for start in range(0, len(indexed_rollouts), args.batch_size):
        chunk = indexed_rollouts[start : start + args.batch_size]
        max_length = max(len(item[1].hidden_states) for item in chunk)
        features = torch.zeros((len(chunk), max_length, input_dim), dtype=torch.float32)
        for index, (_, rollout, _) in enumerate(chunk):
            length = len(rollout.hidden_states)
            features[index, :length] = rollout.hidden_states
        with torch.no_grad():
            padded = model({"features": features.to("cuda")}).squeeze(-1)
        for index, (_, rollout, _) in enumerate(chunk):
            length = len(rollout.hidden_states)
            values = padded[index, :length].detach().cpu().numpy().astype(np.float32)
            scores.append(values)

    maximum_length = bands.shape[1]
    padded_scores = np.full(
        (len(indexed_rollouts), maximum_length), np.nan, dtype=np.float32
    )
    records = []
    for row, ((label, rollout, completion), values) in enumerate(
        zip(indexed_rollouts, scores)
    ):
        if len(values) > maximum_length:
            raise ValueError("fault trace is longer than the frozen monitor band")
        padded_scores[row, : len(values)] = values
        fault = completion["fault"]
        nonfinite_steps = np.flatnonzero(~np.isfinite(values))
        records.append(
            {
                "run": label,
                "task_id": int(rollout.task_id),
                "episode_index": int(rollout.episode_idx),
                "condition": str(completion["condition"]),
                "success": bool(completion["success"]),
                "length": len(values),
                "fault": fault,
                "score_validity": {
                    "all_finite": not bool(len(nonfinite_steps)),
                    "nonfinite_count": int(len(nonfinite_steps)),
                    "first_nonfinite_step": (
                        int(nonfinite_steps[0]) if len(nonfinite_steps) else None
                    ),
                },
                "alarms": alarm_windows(
                    values.tolist(), alphas, bands, int(fault["policy_step"])
                ),
                "finite_guard_alarms": alarm_windows(
                    values.tolist(),
                    alphas,
                    bands,
                    int(fault["policy_step"]),
                    nonfinite_is_alarm=True,
                ),
            }
        )

    json_path = args.output_prefix.with_suffix(".json")
    scores_path = args.output_prefix.with_suffix(".npz")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_scores = scores_path.with_suffix(scores_path.suffix + ".tmp")
    with temporary_scores.open("wb") as file:
        np.savez_compressed(
            file,
            runs=np.asarray([item[0] for item in indexed_rollouts]),
            task_ids=np.asarray([item[1].task_id for item in indexed_rollouts], dtype=np.int16),
            episode_indices=np.asarray(
                [item[1].episode_idx for item in indexed_rollouts], dtype=np.int16
            ),
            successes=np.asarray(
                [item[2]["success"] for item in indexed_rollouts], dtype=bool
            ),
            lengths=np.asarray([len(values) for values in scores], dtype=np.int16),
            scores=padded_scores,
            alphas=np.asarray(alphas, dtype=np.float32),
            bands=bands.astype(np.float32),
        )
    temporary_scores.replace(scores_path)

    output = {
        "schema_version": 2,
        "experiment_code_revision": _git_revision(
            project_root
        ),
        "repository_states": {
            "experiment_code": {
                "revision": _git_revision(project_root),
                "dirty": False,
                "score_safe_sha256": _sha256(Path(__file__)),
            },
            "safe": {"revision": SAFE_REVISION, "dirty": False},
        },
        "monitor": {
            "directory": str(args.monitor_dir.resolve()),
            "kind": monitor_kind,
            "safe_model_name": safe_model_name,
            "checkpoint_sha256": _sha256(monitor_paths["checkpoint"]),
            "configuration_sha256": _sha256(monitor_paths["configuration"]),
            "split_manifest_sha256": _sha256(monitor_paths["split_manifest"]),
            "clean_score_archive_sha256": _sha256(monitor_paths["scores"]),
            "primary_alpha": monitor["primary_alpha"],
        },
        "safe_revision": SAFE_REVISION,
        "alarm_rule": "score >= frozen time-varying upper band",
        "nonfinite_score_policy": {
            "alarms": "Use SAFE's threshold comparison; NaN does not trigger.",
            "finite_guard_alarms": "Trigger on a non-finite score or a threshold crossing.",
        },
        "alarm_windows": ALARM_WINDOWS,
        "score_archive": {
            "path": str(scores_path.resolve()),
            "sha256": _sha256(scores_path),
        },
        "records": records,
    }
    write_json_atomic(json_path, output)
    primary_index = next(
        (
            index
            for index, alpha in enumerate(alphas)
            if math.isclose(
                alpha, float(output["monitor"]["primary_alpha"]), rel_tol=0, abs_tol=1e-8
            )
        ),
        None,
    )
    if primary_index is None:
        raise ValueError("primary SAFE alpha is absent from the frozen threshold band")
    for (label, _rollout, completion), values in zip(indexed_rollouts, scores):
        evidence = completion.get("evidence_graph")
        if not isinstance(evidence, dict):
            continue
        evidence_dir = Path(str(evidence["directory"]))
        if not evidence_dir.is_dir() and evidence.get("directory_relative_to_run"):
            evidence_dir = (
                run_dirs_by_label[label] / evidence["directory_relative_to_run"]
            ).resolve()
        attach_monitor_timeline(
            evidence_dir,
            {
                "kind": output["monitor"]["kind"],
                "safe_model_name": output["monitor"]["safe_model_name"],
                "safe_revision": SAFE_REVISION,
                "checkpoint_sha256": output["monitor"]["checkpoint_sha256"],
                "configuration_sha256": output["monitor"]["configuration_sha256"],
                "primary_alpha": output["monitor"]["primary_alpha"],
                "source_run": label,
            },
            [
                {
                    "policy_step": step,
                    "score": float(score) if math.isfinite(float(score)) else None,
                    "score_finite": math.isfinite(float(score)),
                    "threshold": float(bands[primary_index, step]),
                    "alarm": bool(score >= bands[primary_index, step]),
                }
                for step, score in enumerate(values)
            ],
            monitor_id="safe",
        )
    print(
        json.dumps(
            {
                "fault_rollouts": len(records),
                "failures": sum(not record["success"] for record in records),
                "output": str(json_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
