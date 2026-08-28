from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_scoring import (
    physical_score_index,
    primary_band,
    score_context,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)
from embodied_silent_failures.score_safe import SAFE_REVISION, _validate_monitor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score every language-block intervention with the frozen SAFE monitor."
        )
    )
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--physical-scores", required=True, type=Path)
    parser.add_argument("--physical-score-archive", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _same_monitor(physical: dict, monitor: dict) -> bool:
    return all(
        physical[physical_name] == monitor[monitor_name]["sha256"]
        for physical_name, monitor_name in (
            ("checkpoint_sha256", "checkpoint"),
            ("configuration_sha256", "configuration"),
            ("split_manifest_sha256", "split_manifest"),
            ("clean_score_archive_sha256", "scores"),
        )
    )


def main() -> None:
    args = _arguments()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    project_root = Path(__file__).resolve().parents[1]
    if git_revision(args.safe_root) != SAFE_REVISION:
        raise RuntimeError(f"SAFE must be checked out at {SAFE_REVISION}")
    for name, path in (("experiment code", project_root), ("SAFE", args.safe_root)):
        if git_dirty(path):
            raise RuntimeError(f"{name} has uncommitted changes: {path}")

    run = load_json(args.campaign_dir / "run.json")
    if run.get("campaign") != "openvla_language_block_temporal_replacement":
        raise ValueError("campaign directory is not a language-block campaign")
    monitor, monitor_paths = _validate_monitor(args.monitor_dir)
    physical_json = load_json(args.physical_scores)
    if not _same_monitor(physical_json["monitor"], monitor):
        raise ValueError("ordinary physical scores used a different SAFE monitor")

    sys.path.insert(0, str(args.safe_root.resolve()))
    import numpy as np
    import torch
    from failure_prob.data.utils import process_tensor_idx_rel
    from failure_prob.model import get_model
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to score SAFE feature traces")
    cfg = OmegaConf.load(monitor_paths["configuration"])
    if str(cfg.model.name) != "indep":
        raise ValueError("the frozen language campaign uses the SAFE-MLP monitor")
    physical_index, bands, alphas = physical_score_index(
        physical_json, args.physical_score_archive, np
    )
    primary = primary_band(physical_json, bands, alphas)

    # SAFE b6036abe, failure_prob/data/openvla.py::load_rollouts, produces one
    # 4096-value final-token feature per OpenVLA policy decision.
    model = get_model(cfg, 4096)
    state_dict = torch.load(monitor_paths["checkpoint"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()

    records = []
    contexts = []
    score_arrays = {}
    checks = []
    planned_ids = [str(value) for value in run["context_ids"]]
    for context_id in planned_ids:
        context_dir = args.campaign_dir / "contexts" / context_id
        if not (context_dir / "context.complete.json").is_file():
            unresolved_path = context_dir / "context.unresolved.json"
            unresolved = load_json(unresolved_path) if unresolved_path.is_file() else {}
            contexts.append(
                {
                    "context_id": context_id,
                    "status": str(unresolved.get("status", "missing")),
                    "errors": unresolved.get("errors", []),
                    "context": unresolved.get("context"),
                }
            )
            continue
        context, context_records, context_scores, context_checks = score_context(
            campaign_dir=args.campaign_dir,
            context_id=context_id,
            cfg=cfg,
            model=model,
            physical_index=physical_index,
            bands=bands,
            alphas=alphas,
            primary=primary,
            batch_size=args.batch_size,
            torch=torch,
            np=np,
            process_tensor_idx_rel=process_tensor_idx_rel,
        )
        contexts.append(context)
        records.extend(context_records)
        score_arrays.update(context_scores)
        checks.extend(context_checks)

    maximum_length = bands.shape[1]
    ordered_ids = sorted(score_arrays)
    padded = np.full((len(ordered_ids), maximum_length), np.nan, dtype=np.float32)
    for row, record_id in enumerate(ordered_ids):
        values = score_arrays[record_id]
        if len(values) > maximum_length:
            raise ValueError("reconstructed SAFE trace exceeds the frozen monitor band")
        padded[row, : len(values)] = values
    archive_path = args.output_prefix.with_suffix(".npz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            record_ids=np.asarray(ordered_ids),
            lengths=np.asarray([len(score_arrays[value]) for value in ordered_ids]),
            scores=padded,
            alphas=np.asarray(alphas, dtype=np.float32),
            bands=bands.astype(np.float32),
        )
    temporary.replace(archive_path)

    output = {
        "schema_version": 1,
        "analysis": "frozen SAFE-MLP scoring of language-block interventions",
        "repository_states": {
            "experiment_code": {
                "revision": git_revision(project_root),
                "dirty": False,
                "score_language_campaign_sha256": file_sha256(Path(__file__)),
            },
            "safe": {"revision": SAFE_REVISION, "dirty": False},
        },
        "monitor": physical_json["monitor"],
        "alarm_rule": physical_json["alarm_rule"],
        "alarm_windows": physical_json["alarm_windows"],
        "source_campaign": {
            "directory": str(args.campaign_dir.resolve()),
            "run_sha256": file_sha256(args.campaign_dir / "run.json"),
            "worker_shard": run["worker_shard"],
        },
        "ordinary_physical_scores": {
            "json_sha256": file_sha256(args.physical_scores),
            "archive_sha256": file_sha256(args.physical_score_archive),
        },
        "score_archive": {
            "path": str(archive_path.resolve()),
            "sha256": file_sha256(archive_path),
        },
        "coverage": {
            "planned_contexts": len(planned_ids),
            "complete_contexts": sum(value["status"] == "complete" for value in contexts),
            "unresolved_contexts": sum(value["status"] != "complete" for value in contexts),
            "planned_interventions": len(planned_ids) * 32,
            "scored_interventions": sum(value["status"] == "scored" for value in records),
            "composition_unverified_interventions": sum(
                value["status"] == "composition_unverified" for value in records
            ),
        },
        "composition_audit": {
            "groups_checked": len(checks),
            "groups_valid": sum(value["valid"] for value in checks),
            "checks": checks,
        },
        "contexts": contexts,
        "records": records,
    }
    output_path = args.output_prefix.with_suffix(".json")
    write_json_atomic(output_path, output)
    print(
        json.dumps(
            {
                "coverage": output["coverage"],
                "composition_audit": {
                    "groups_checked": len(checks),
                    "groups_valid": sum(value["valid"] for value in checks),
                },
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
