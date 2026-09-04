from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.atlas_scoring import score_context
from embodied_silent_failures.language_scoring import physical_score_index
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)
from embodied_silent_failures.score_safe import SAFE_REVISION, _validate_monitor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score graph-atlas interventions with the frozen SAFE-MLP."
    )
    parser.add_argument("--safe-root", required=True, type=Path)
    parser.add_argument("--monitor-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--physical-scores", required=True, type=Path)
    parser.add_argument("--physical-score-archive", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _same_monitor(physical: dict[str, Any], monitor: dict[str, Any], scores: Path) -> bool:
    return all(
        physical[physical_name] == monitor[monitor_name]["sha256"]
        for physical_name, monitor_name in (
            ("checkpoint_sha256", "checkpoint"),
            ("configuration_sha256", "configuration"),
            ("split_manifest_sha256", "split_manifest"),
        )
    ) and physical["clean_score_archive_sha256"] == file_sha256(scores)


def _write_split(
    *,
    split: str,
    output_prefix: Path,
    contexts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    score_arrays: dict[str, tuple[Any, Any]],
    bands: Any,
    alphas: list[float],
    metadata: dict[str, Any],
    np: Any,
) -> dict[str, int]:
    selected_contexts = [
        value for value in contexts if value["context"]["analysis_split"] == split
    ]
    context_ids = {str(value["context_id"]) for value in selected_contexts}
    selected_checks = [
        value
        for value in metadata["composition_audit"]["checks"]
        if str(value["context_id"]) in context_ids
    ]
    split_metadata = {
        **metadata,
        "composition_audit": {
            "physical_branches_checked": len(selected_checks),
            "feature_exact_branches": sum(
                bool(value["feature_exact_equal"]) for value in selected_checks
            ),
            "alarm_exact_branches": sum(
                bool(value.get("alarm_timeline_exact_equal"))
                for value in selected_checks
            ),
            "checks": selected_checks,
        },
    }
    selected_records = [
        value for value in records if str(value["context_id"]) in context_ids
    ]
    selected_ids = sorted(
        value["record_id"]
        for value in selected_records
        if value["record_id"] in score_arrays
    )
    maximum_length = bands.shape[1]
    faulted = np.full((len(selected_ids), maximum_length), np.nan, dtype=np.float32)
    clean = np.full_like(faulted, np.nan)
    lengths = []
    for row, record_id in enumerate(selected_ids):
        faulted_values, clean_values = score_arrays[record_id]
        if len(faulted_values) != len(clean_values):
            raise ValueError(f"SAFE reconstructions have unequal lengths: {record_id}")
        if len(faulted_values) > maximum_length:
            raise ValueError(f"SAFE reconstruction exceeds frozen band: {record_id}")
        lengths.append(len(faulted_values))
        faulted[row, : len(faulted_values)] = faulted_values
        clean[row, : len(clean_values)] = clean_values

    archive_path = Path(f"{output_prefix}-{split}.npz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            record_ids=np.asarray(selected_ids),
            lengths=np.asarray(lengths, dtype=np.int16),
            faulted_evidence_scores=faulted,
            clean_evidence_same_suffix_scores=clean,
            alphas=np.asarray(alphas, dtype=np.float32),
            bands=bands.astype(np.float32),
        )
    temporary.replace(archive_path)
    output = {
        **split_metadata,
        "analysis_split": split,
        "score_archive": {
            "path": str(archive_path.resolve()),
            "sha256": file_sha256(archive_path),
        },
        "coverage": {
            "contexts": len(selected_contexts),
            "interventions": len(selected_records),
            "scored_interventions": len(selected_ids),
            "primary_eligible_interventions": sum(
                bool(value.get("primary_eligible")) for value in selected_records
            ),
        },
        "contexts": selected_contexts,
        "records": selected_records,
    }
    write_json_atomic(Path(f"{output_prefix}-{split}.json"), output)
    return output["coverage"]


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

    manifest = load_json(args.manifest)
    run = load_json(args.campaign_dir / "run.json")
    manifest_sha256 = file_sha256(args.manifest)
    if run.get("campaign") != manifest.get("campaign"):
        raise ValueError("campaign and manifest identities differ")
    if run["execution"]["manifest_file_sha256"] != manifest_sha256:
        raise ValueError("campaign was not executed from the supplied manifest")
    site_by_id = {str(value["site_id"]): value for value in manifest["sites"]}

    monitor, monitor_paths = _validate_monitor(args.monitor_dir)
    physical_json = load_json(args.physical_scores)
    if not _same_monitor(physical_json["monitor"], monitor, monitor_paths["scores"]):
        raise ValueError("physical scores used a different frozen SAFE monitor")

    sys.path.insert(0, str(args.safe_root.resolve()))
    import numpy as np
    import torch
    from failure_prob.data.utils import process_tensor_idx_rel
    from failure_prob.model import get_model
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to score SAFE atlas features")
    cfg = OmegaConf.load(monitor_paths["configuration"])
    if not (
        str(cfg.model.name) == "indep"
        and int(cfg.model.n_history_steps) == 1
        and bool(cfg.model.cumsum)
        and not bool(cfg.model.rmean)
    ):
        raise ValueError(
            "atlas reconstruction requires the frozen one-step cumulative SAFE-MLP"
        )
    physical_index, bands, alphas = physical_score_index(
        physical_json, args.physical_score_archive, np
    )
    model = get_model(cfg, 4096)
    model.load_state_dict(torch.load(monitor_paths["checkpoint"], map_location="cpu"))
    model.to("cuda")
    model.eval()

    contexts = []
    records = []
    score_arrays = {}
    checks = []
    shard_contexts = [
        value
        for value in manifest["contexts"]
        if int(value["worker_shard"]) == int(run["worker_shard"])
    ]
    for planned in shard_contexts:
        context_id = str(planned["context_id"])
        context_dir = args.campaign_dir / "contexts" / context_id
        if not (context_dir / "context.complete.json").is_file():
            contexts.append(
                {"context_id": context_id, "status": "incomplete", "context": planned}
            )
            continue
        context, context_records, context_scores, context_checks = score_context(
            campaign_dir=args.campaign_dir,
            context_id=context_id,
            site_by_id=site_by_id,
            model=model,
            cfg=cfg,
            physical_index=physical_index,
            physical_score_json=physical_json,
            bands=bands,
            alphas=alphas,
            batch_size=args.batch_size,
            torch=torch,
            np=np,
            process_tensor_idx_rel=process_tensor_idx_rel,
        )
        contexts.append(context)
        records.extend(context_records)
        score_arrays.update(context_scores)
        checks.extend(context_checks)

    metadata = {
        "schema_version": 1,
        "analysis": "frozen SAFE-MLP scoring of graph-atlas interventions",
        "repository_states": {
            "experiment_code": {
                "revision": git_revision(project_root),
                "dirty": False,
                "score_intervention_atlas_sha256": file_sha256(Path(__file__)),
            },
            "safe": {"revision": SAFE_REVISION, "dirty": False},
        },
        "source": {
            "manifest_sha256": manifest_sha256,
            "campaign_run_sha256": file_sha256(args.campaign_dir / "run.json"),
            "physical_score_json_sha256": file_sha256(args.physical_scores),
            "physical_score_archive_sha256": file_sha256(args.physical_score_archive),
            "worker_shard": int(run["worker_shard"]),
        },
        "monitor": physical_json["monitor"],
        "alarm_rule": physical_json["alarm_rule"],
        "alarm_windows": physical_json["alarm_windows"],
        "analysis_contract": manifest["analysis_contract"],
        "composition_audit": {
            "physical_branches_checked": len(checks),
            "feature_exact_branches": sum(
                bool(value["feature_exact_equal"]) for value in checks
            ),
            "alarm_exact_branches": sum(
                bool(value.get("alarm_timeline_exact_equal")) for value in checks
            ),
            "checks": checks,
        },
    }
    coverage = {
        split: _write_split(
            split=split,
            output_prefix=args.output_prefix,
            contexts=contexts,
            records=records,
            score_arrays=score_arrays,
            bands=bands,
            alphas=alphas,
            metadata=metadata,
            np=np,
        )
        for split in ("development", "holdout")
    }
    # Deliberately report coverage only. Outcome counts remain sealed by split.
    print(json.dumps({"coverage": coverage}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
