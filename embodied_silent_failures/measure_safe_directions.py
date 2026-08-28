from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record, write_json_atomic
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
    source_file_record,
)
from embodied_silent_failures.safe_directions import monitor_direction_batch
from embodied_silent_failures.score_safe import (
    SAFE_REVISION,
    _validate_monitor,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how OpenVLA feature faults move the frozen SAFE-MLP score."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _eligible(record: dict[str, Any]) -> bool:
    return bool(
        record.get("status") == "scored"
        and record.get("composition_verified")
        and record.get("control_success")
        and record.get("terminal_success") is not None
        and record.get("monitor_horizon") == "complete_physical_trace"
    )


def _outcome_group(record: dict[str, Any], primary_alpha: str) -> str | None:
    if not _eligible(record):
        return None
    local = record["local_measurements"]
    if bool(local["executed_command"]["exact_equal"]):
        return "command_unchanged"
    if bool(record["terminal_success"]):
        return "changed_command_success"
    alarms = record["alarms"][primary_alpha]["post_fault_any"]
    return "detected_failure" if alarms["triggered"] else "silent_failure"


def _validate_configuration(cfg: Any) -> None:
    expected = {
        "model.name": (str(cfg.model.name), "indep"),
        "model.n_layers": (int(cfg.model.n_layers), 2),
        "model.hidden_dim": (int(cfg.model.hidden_dim), 256),
        "model.final_act_layer": (str(cfg.model.final_act_layer), "sigmoid"),
        "model.n_history_steps": (int(cfg.model.n_history_steps), 1),
        "model.cumsum": (bool(cfg.model.cumsum), True),
        "model.rmean": (bool(cfg.model.rmean), False),
        "dataset.token_idx_rel": (float(cfg.dataset.token_idx_rel), 1.0),
    }
    disagreements = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if disagreements:
        raise ValueError(
            "SAFE direction method does not match monitor: "
            + "; ".join(disagreements)
        )


def main() -> None:
    args = _arguments()
    project_root = Path(__file__).resolve().parents[1]
    if git_revision(args.safe_root) != SAFE_REVISION or git_dirty(args.safe_root):
        raise RuntimeError("SAFE source must be the clean pinned revision")
    if not args.campaign_dir.is_dir():
        raise FileNotFoundError(f"campaign directory is absent: {args.campaign_dir}")

    score_document = load_json(args.scores)
    monitor, monitor_paths = _validate_monitor(args.monitor_dir)
    checkpoint_hash = file_sha256(monitor_paths["checkpoint"])
    if score_document["monitor"]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError(
            "language scores and directional analysis use different SAFE checkpoints"
        )
    declared_campaign = Path(score_document["source_campaign"]["directory"]).resolve()
    if declared_campaign != args.campaign_dir.resolve():
        raise ValueError("language score document names a different campaign directory")

    sys.path.insert(0, str(args.safe_root.resolve()))
    import torch
    from omegaconf import OmegaConf

    from failure_prob.data.utils import process_tensor_idx_rel
    from failure_prob.model import get_model

    cfg = OmegaConf.load(monitor_paths["configuration"])
    _validate_configuration(cfg)
    # SAFE@b6036ab, failure_prob/model/indep.py: get_model builds the frozen
    # 4096 -> 256 -> 1 sigmoid projector whose per-step output is accumulated.
    model = get_model(cfg, 4096)
    state_dict = torch.load(monitor_paths["checkpoint"], map_location="cpu")
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in score_document["records"]:
        records_by_context[str(record["context_id"])].append(record)

    output_records = []
    omitted = Counter()
    maximum_score_delta_error = 0.0
    for context_id, score_records in sorted(records_by_context.items()):
        context_dir = args.campaign_dir / "contexts" / context_id
        feature_path = context_dir / "local_features.pkl"
        local_path = context_dir / "local.json"
        if not feature_path.is_file() or not local_path.is_file():
            omitted["missing_local_feature_archive"] += len(score_records)
            continue
        local_document = load_json(local_path)
        if artifact_record(feature_path) != local_document["feature_archive"]:
            raise ValueError(
                f"local feature archive disagrees with its manifest: {context_id}"
            )
        with feature_path.open("rb") as file:
            features = pickle.load(file)

        clean_raw = features["clean_hidden_states"].float()
        # SAFE@b6036ab, failure_prob/data/utils.py: process_tensor_idx_rel with
        # token_idx_rel=1.0 selects the final action-token feature used for scoring.
        clean_selected = process_tensor_idx_rel(
            clean_raw, float(cfg.dataset.token_idx_rel)
        )
        if clean_selected.ndim != 1 or clean_selected.shape[0] != 4096:
            raise ValueError(
                f"SAFE selector returned {tuple(clean_selected.shape)} for {context_id}"
            )
        usable = []
        faulted_selected = []
        ordered_records = sorted(
            score_records, key=lambda value: int(value["layer_index"])
        )
        for score_record in ordered_records:
            layer_index = int(score_record["layer_index"])
            faulted = features["faulted_hidden_states_by_layer"].get(layer_index)
            if faulted is None:
                omitted["missing_layer_feature"] += 1
                continue
            usable.append(score_record)
            selected = process_tensor_idx_rel(
                faulted.float(), float(cfg.dataset.token_idx_rel)
            )
            if selected.shape != clean_selected.shape:
                raise ValueError(
                    "faulted SAFE input shape differs for "
                    f"{context_id}/layer{layer_index}"
                )
            faulted_selected.append(selected)
        if not usable:
            continue

        clean_batch = clean_selected.unsqueeze(0).expand(len(usable), -1).to(device)
        faulted_batch = torch.stack(faulted_selected).to(device)
        measurements = monitor_direction_batch(model, clean_batch, faulted_batch, torch)
        for score_record, direction in zip(usable, measurements, strict=True):
            stored_delta = score_record.get("score_change_from_control_at_fault")
            computed_delta = direction["monitor_increment_delta"]
            score_delta_error = (
                abs(float(stored_delta) - computed_delta)
                if stored_delta is not None and math.isfinite(float(stored_delta))
                else None
            )
            if score_delta_error is not None:
                maximum_score_delta_error = max(
                    maximum_score_delta_error, score_delta_error
                )
            local = score_record["local_measurements"]
            context = score_record["context"]
            threshold = score_record.get("threshold_at_fault")
            score = score_record.get("score_at_fault")
            output_records.append(
                {
                    "record_id": score_record["record_id"],
                    "worker_shard": int(
                        score_document["source_campaign"]["worker_shard"]
                    ),
                    "context_id": context_id,
                    "analysis_split": context["analysis_split"],
                    "task_id": int(context["task_id"]),
                    "episode_index": int(context["episode_index"]),
                    "phase": context["phase"],
                    "policy_step": int(context["policy_step"]),
                    "action_token_position": int(context["action_token_position"]),
                    "layer_index": int(score_record["layer_index"]),
                    "physical_run": score_record.get("physical_run"),
                    "command_changed": not bool(
                        local["executed_command"]["exact_equal"]
                    ),
                    "eligible_causal_outcome": _eligible(score_record),
                    "outcome_group": _outcome_group(
                        score_record,
                        format(
                            float(score_document["monitor"]["primary_alpha"]), "g"
                        ),
                    ),
                    "terminal_success": score_record.get("terminal_success"),
                    "safe_alarm_at_fault": score_record.get("alarm_at_fault"),
                    "safe_alarm_post_fault_any": score_record.get("alarms", {})
                    .get(
                        format(
                            float(score_document["monitor"]["primary_alpha"]), "g"
                        ),
                        {},
                    )
                    .get("post_fault_any", {})
                    .get("triggered"),
                    "stored_score_delta": stored_delta,
                    "computed_increment_delta": computed_delta,
                    "stored_score_delta_absolute_error": score_delta_error,
                    "threshold_margin_after_fault": (
                        float(threshold) - float(score)
                        if threshold is not None and score is not None
                        else None
                    ),
                    **direction,
                }
            )

    output = {
        "schema_version": 1,
        "analysis": "directional response of the frozen SAFE-MLP at its OpenVLA input",
        "source_campaign": score_document["source_campaign"],
        "method": (
            "Select SAFE's configured final action-token feature, differentiate the "
            "unchanged two-layer sigmoid projector at the clean feature, and compare "
            "that local direction with the exact clean-to-fault score increment."
        ),
        "provenance": {
            "experiment_revision": git_revision(project_root),
            "experiment_dirty": git_dirty(project_root),
            "safe_revision": git_revision(args.safe_root),
            "safe_model_source": source_file_record(model),
            "safe_token_selector_source": {
                "function": "failure_prob.data.utils.process_tensor_idx_rel",
                "path": str(
                    Path(process_tensor_idx_rel.__code__.co_filename).resolve()
                ),
                "sha256": file_sha256(
                    Path(process_tensor_idx_rel.__code__.co_filename)
                ),
            },
            "monitor": {
                "checkpoint_sha256": checkpoint_hash,
                "configuration_sha256": file_sha256(monitor_paths["configuration"]),
                "clean_calibration_sha256": file_sha256(monitor_paths["scores"]),
                "monitor_manifest_sha256": file_sha256(monitor_paths["monitor"]),
                "split_manifest_sha256": file_sha256(monitor_paths["split_manifest"]),
            },
            "source_scores": {
                "path": str(args.scores.resolve()),
                "sha256": file_sha256(args.scores),
            },
            "frozen_monitor_manifest": monitor["checkpoint"],
        },
        "device": device,
        "coverage": {
            "score_records": len(score_document["records"]),
            "direction_records": len(output_records),
            "omitted": dict(sorted(omitted.items())),
            "maximum_stored_score_delta_absolute_error": maximum_score_delta_error,
        },
        "records": output_records,
    }
    write_json_atomic(args.output, output)
    summary = {key: output[key] for key in ("analysis", "device", "coverage")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
