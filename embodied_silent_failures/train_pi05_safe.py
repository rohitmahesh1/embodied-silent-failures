from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record, write_json_atomic
from embodied_silent_failures.pi05_contract import SAFE_REVISION
from embodied_silent_failures.pi05_safe_data import FEATURE_PROTOCOL
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)


ALPHAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
PRIMARY_ALPHA = 0.1


def _environment_versions() -> dict[str, Any]:
    packages = (
        "hydra-core",
        "natsort",
        "numpy",
        "omegaconf",
        "scikit-learn",
        "torch",
        "wandb",
    )
    return {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in packages
        },
    }


def published_configuration(epochs: int = 1000) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "pizero",
            "data_path_unseen": None,
            "load_to_cuda": False,
            "normalize_hidden_states": False,
            "unseen_task_ratio": 0.3,
            "seen_train_ratio": 0.6,
            "feat_name": "pre_velocity",
            "horizon_idx_rel": 0.0,
            "diff_idx_rel": 1.0,
        },
        "model": {
            "name": "indep",
            "use_time_weighting": False,
            "n_epochs": epochs,
            "batch_size": 512,
            "optimizer": "adam",
            "lr": 3e-5,
            "lr_step_size": 300,
            "lr_gamma": 1.0,
            "weight_decay": 1e-2,
            "warmup_steps": 0,
            "lambda_success": 1.0,
            "lambda_fail": 1.0,
            "lambda_reg": 1e-3,
            "cumsum": True,
            "rmean": False,
            "init_weight_scale": 1.0,
            "grad_max_norm": None,
            "dropout": 0.0,
            "n_layers": 2,
            "hidden_dim": 256,
            "final_act_layer": "sigmoid",
            "n_history_steps": 1,
            "use_threshold": False,
            "threshold": 50.0,
        },
        "train": {"seed": 0},
    }


def _identity(rollout: Any) -> dict[str, Any]:
    return {
        "task_id": int(rollout.task_id),
        "episode_index": int(rollout.episode_idx),
        "success": bool(rollout.episode_success),
        "decisions": int(len(rollout.hidden_states)),
    }


def _split_manifest(splits: dict[str, list[Any]], seed: int) -> dict[str, Any]:
    seen = sorted(
        {
            int(item.task_id)
            for name in ("train", "val_seen")
            for item in splits[name]
        }
    )
    unseen = sorted({int(item.task_id) for item in splits["val_unseen"]})
    return {
        "schema_version": 1,
        "seed": seed,
        "ordering": "SAFE's seeded task shuffle and per-task torch.randperm",
        "seen_task_ids": seen,
        "unseen_task_ids": unseen,
        "counts": {
            name: {
                "rollouts": len(values),
                "successes": sum(bool(item.episode_success) for item in values),
                "failures": sum(not bool(item.episode_success) for item in values),
            }
            for name, values in splits.items()
        },
        "splits": {
            name: [_identity(item) for item in values]
            for name, values in splits.items()
        },
    }


def _load_rollouts(
    feature_dir: Path, torch: Any, Rollout: Any
) -> tuple[list[Any], dict[str, Any]]:
    import numpy as np

    manifest_path = feature_dir / "manifest.json"
    archive_path = feature_dir / "features.npz"
    manifest = load_json(manifest_path)
    if manifest.get("feature_protocol") != FEATURE_PROTOCOL:
        raise ValueError("feature archive does not use the frozen SAFE pi0 protocol")
    if artifact_record(archive_path) != manifest.get("archive"):
        raise ValueError("feature archive disagrees with its manifest")
    archive = np.load(archive_path)
    features = archive["features"]
    offsets = archive["offsets"]
    task_ids = archive["task_ids"]
    episode_indices = archive["episode_indices"]
    successes = archive["successes"]
    count = len(task_ids)
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(features)
    ):
        raise ValueError("feature archive has invalid rollout offsets")
    if len(episode_indices) != count or len(successes) != count:
        raise ValueError("feature archive has inconsistent rollout metadata")

    rollouts = []
    for index in range(count):
        selected = torch.from_numpy(
            features[int(offsets[index]) : int(offsets[index + 1])]
        )
        rollouts.append(
            Rollout(
                hidden_states=selected,
                task_suite_name="libero_10",
                task_id=int(task_ids[index]),
                task_description=f"Task {int(task_ids[index])}",
                episode_idx=int(episode_indices[index]),
                episode_success=int(successes[index]),
                mp4_path="",
                exec_horizon=int(manifest["source"]["replan_steps"]),
            )
        )
    return rollouts, manifest


def _forward_scores(model: Any, rollouts: list[Any], torch: Any) -> list[Any]:
    scores = []
    model.eval()
    with torch.no_grad():
        for rollout in rollouts:
            batch = {"features": rollout.hidden_states[None].to(model.get_device())}
            values = model(batch).squeeze(0).squeeze(-1).detach().cpu().numpy()
            scores.append(values)
    return scores


def _binary_metrics(labels: Any, values: Any) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
    }


def _auc_metrics(rollouts: list[Any], scores: list[Any]) -> dict[str, Any]:
    labels = [1 - int(item.episode_success) for item in rollouts]
    evaluations = {
        "at_earliest_task_stop": [
            score[item.task_min_step - 1] for item, score in zip(rollouts, scores)
        ],
        "maximum_by_earliest_task_stop": [
            max(score[: item.task_min_step])
            for item, score in zip(rollouts, scores)
        ],
        "maximum_by_terminal_outcome": [max(score) for score in scores],
    }
    return {
        name: _binary_metrics(labels, values) for name, values in evaluations.items()
    }


def _clean_alarm_metrics(
    splits: dict[str, list[Any]], scores: dict[str, list[Any]], band: Any
) -> dict[str, Any]:
    result = {}
    for name, rollouts in splits.items():
        alarms = [
            bool((score >= band[: len(score)]).any()) for score in scores[name]
        ]
        successes = [bool(item.episode_success) for item in rollouts]
        result[name] = {
            "rollouts": len(rollouts),
            "alarms": sum(alarms),
            "alarm_rate": sum(alarms) / len(alarms),
            "successful_rollouts": sum(successes),
            "false_alarms_on_successes": sum(
                alarm and success for alarm, success in zip(alarms, successes)
            ),
            "false_alarm_rate_on_successes": (
                sum(alarm and success for alarm, success in zip(alarms, successes))
                / sum(successes)
                if any(successes)
                else None
            ),
        }
    return result


def _functional_band_row(band: Any, maximum_length: int, np: Any) -> Any:
    values = np.asarray(band)
    # SAFE b6036ab, failure_prob/utils/conformal/functional_predictor.py::
    # get_one_sided_prediction_band, returns one regression row and
    # failure_prob/utils/metrics.py broadcasts that (1, time) row over rollouts.
    if values.shape == (1, maximum_length):
        values = values[0]
    if values.shape != (maximum_length,):
        raise ValueError(
            "functional calibration returned an unexpected band shape: "
            f"{values.shape}"
        )
    return values


def _functional_bands(
    calibration_rollouts: list[Any],
    calibration_scores: list[Any],
    *,
    seed: int,
    maximum_length: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from failure_prob.utils.conformal.functional_predictor import (
        FunctionalPredictor,
        ModulationType,
        RegressionType,
    )

    successes = [
        (rollout, score)
        for rollout, score in zip(calibration_rollouts, calibration_scores)
        if bool(rollout.episode_success)
    ]
    if len(successes) < 4:
        raise ValueError("functional calibration requires at least four successful rollouts")
    if maximum_length < max(len(score) for _, score in successes):
        raise ValueError("functional band length is shorter than calibration data")
    padded = np.stack(
        [
            np.pad(score, (0, maximum_length - len(score)), mode="edge")
            for _, score in successes
        ]
    )
    permutation = np.random.RandomState(seed).permutation(len(successes))
    fit_count = int(len(successes) * 0.3)
    if fit_count == 0 or fit_count == len(successes):
        raise ValueError("functional calibration partition is empty")
    fit_indices = permutation[:fit_count]
    conformal_indices = permutation[fit_count:]
    bands = []
    for alpha in ALPHAS:
        predictor = FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)
        band = predictor.get_one_sided_prediction_band(
            padded[fit_indices],
            padded[conformal_indices],
            alpha,
            lower_bound=False,
        )
        band = _functional_band_row(band, maximum_length, np)
        if not np.isfinite(band).all():
            raise ValueError("functional calibration produced a non-finite band")
        bands.append(band)
    partition = {
        "seed": seed,
        "fit_fraction": 0.3,
        "population": "successful val_seen rollouts",
        "band_length": maximum_length,
        "band_length_basis": "longest rollout in the complete clean baseline",
        "fit": [_identity(successes[index][0]) for index in fit_indices],
        "calibration": [
            _identity(successes[index][0]) for index in conformal_indices
        ],
    }
    return np.stack(bands).astype(np.float32), partition


def train(
    safe_root: Path,
    feature_dir: Path,
    output_dir: Path,
    *,
    seed: int = 0,
    calibration_seed: int = 0,
    epochs: int = 1000,
) -> dict[str, Any]:
    safe_root = safe_root.resolve()
    feature_dir = feature_dir.resolve()
    output_dir = output_dir.resolve()
    if git_revision(safe_root) != SAFE_REVISION or git_dirty(safe_root):
        raise RuntimeError(f"SAFE must be clean and pinned at {SAFE_REVISION}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"monitor output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir()

    sys.path.insert(0, str(safe_root))
    import numpy as np
    import torch
    import wandb
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from failure_prob.data import pizero
    from failure_prob.data.utils import Rollout, RolloutDataset, set_task_min_step
    from failure_prob.model import get_model
    from failure_prob.utils.random import seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train SAFE-MLP")
    cfg = OmegaConf.create(published_configuration(epochs))
    cfg.train.seed = seed
    rollouts, feature_manifest = _load_rollouts(feature_dir, torch, Rollout)
    set_task_min_step(rollouts)
    seed_everything(seed)
    splits = pizero.split_rollouts(cfg, rollouts)
    split_manifest = _split_manifest(splits, seed)
    if split_manifest["counts"]["train"]["failures"] == 0:
        raise ValueError("SAFE training split contains no failures")
    split_path = output_dir / "split.json"
    write_json_atomic(split_path, split_manifest)

    datasets = {name: RolloutDataset(cfg, values) for name, values in splits.items()}
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=0,
    )
    input_dim = int(splits["train"][0].hidden_states.shape[-1])
    model = get_model(cfg, input_dim).to("cuda")
    optimizer, scheduler = model.get_optimizer()
    os.environ.setdefault("WANDB_MODE", "disabled")
    wandb.init(mode="disabled")
    losses = []
    try:
        for _ in range(epochs):
            model.train()
            losses.append(float(model.train_epoch(optimizer, train_loader)))
            scheduler.step()
    finally:
        wandb.finish(quiet=True)

    checkpoint_path = artifacts_dir / "model_final.ckpt"
    torch.save(model.state_dict(), checkpoint_path)
    config_path = artifacts_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    scores = {
        name: _forward_scores(model, values, torch) for name, values in splits.items()
    }
    metrics = {
        name: _auc_metrics(splits[name], scores[name]) for name in sorted(splits)
    }
    bands, calibration_partition = _functional_bands(
        splits["val_seen"],
        scores["val_seen"],
        seed=calibration_seed,
        maximum_length=max(len(rollout.hidden_states) for rollout in rollouts),
    )
    primary_index = ALPHAS.index(PRIMARY_ALPHA)
    clean_alarm_metrics = _clean_alarm_metrics(
        splits, scores, bands[primary_index]
    )
    maximum_length = int(bands.shape[1])
    ordered = [
        (name, rollout, score)
        for name in sorted(splits)
        for rollout, score in zip(splits[name], scores[name])
    ]
    padded_scores = np.full((len(ordered), maximum_length), np.nan, dtype=np.float32)
    lengths = []
    for index, (_, _, score) in enumerate(ordered):
        if len(score) > maximum_length:
            raise ValueError("clean score exceeds the functional band length")
        padded_scores[index, : len(score)] = score
        lengths.append(len(score))
    score_path = output_dir / "clean_scores.npz"
    with score_path.open("wb") as file:
        np.savez_compressed(
            file,
            splits=np.asarray([item[0] for item in ordered]),
            task_ids=np.asarray([item[1].task_id for item in ordered], dtype=np.int16),
            episode_indices=np.asarray(
                [item[1].episode_idx for item in ordered], dtype=np.int16
            ),
            successes=np.asarray(
                [item[1].episode_success for item in ordered], dtype=bool
            ),
            lengths=np.asarray(lengths, dtype=np.int16),
            scores=padded_scores,
            alphas=np.asarray(ALPHAS, dtype=np.float32),
            bands=bands,
        )

    monitor = {
        "schema_version": 1,
        "model": "SAFE-MLP",
        "safe_revision": SAFE_REVISION,
        "feature_protocol": FEATURE_PROTOCOL,
        "replan_steps": int(feature_manifest["source"]["replan_steps"]),
        "feature": "first action-horizon feature from the final diffusion step",
        "training_protocol": {
            "paper": "SAFE arXiv:2506.09937v2 Table 10",
            "implementation": (
                "SAFE b6036ab failure_prob/model/indep.py::IndepModel and "
                "failure_prob/model/base.py::BaseModel.train_epoch"
            ),
            "seed": seed,
            "epochs": epochs,
        },
        "environment": _environment_versions(),
        "source_features": {
            "manifest": str((feature_dir / "manifest.json").resolve()),
            "sha256": file_sha256(feature_dir / "manifest.json"),
            "source_run_json_sha256s": [
                item["run_json_sha256"]
                for item in feature_manifest["source"].get(
                    "runs", [feature_manifest["source"]]
                )
            ],
        },
        "checkpoint": artifact_record(checkpoint_path),
        "configuration": artifact_record(config_path),
        "split_manifest": artifact_record(split_path),
        "score_archive": artifact_record(score_path),
        "primary_alpha": PRIMARY_ALPHA,
        "sensitivity_alphas": list(ALPHAS),
        "alarm_comparison": "score >= time-varying upper band",
        "calibration_partition": calibration_partition,
        "clean_metrics": metrics,
        "clean_alarm_metrics": clean_alarm_metrics,
        "training_loss": {"first": losses[0], "final": losses[-1]},
    }
    write_json_atomic(output_dir / "monitor.json", monitor)
    return monitor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and freeze SAFE-MLP on compact clean pi0.5 features."
    )
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    if args.seed < 0 or args.calibration_seed < 0 or args.epochs <= 0:
        raise ValueError("seeds must be non-negative and epochs must be positive")
    return args


def main() -> None:
    args = _arguments()
    result = train(
        args.safe_root,
        args.feature_dir,
        args.output_dir,
        seed=args.seed,
        calibration_seed=args.calibration_seed,
        epochs=args.epochs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
