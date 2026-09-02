from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    write_json_atomic,
    write_npz_atomic,
)
from embodied_silent_failures.language_interface_atlas import context_arrays
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CPU-analysis cuts from exact OpenVLA interface traces."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("development", "holdout", "all"), default="development"
    )
    parser.add_argument("--context-id", action="append", dest="context_ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-source-hash-check", action="store_true")
    return parser.parse_args()


def _selected_contexts(
    score_document: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    contexts = [value["context"] for value in score_document["contexts"]]
    selected = [
        value
        for value in contexts
        if split == "all" or value.get("analysis_split") == split
    ]
    ids = [str(value["context_id"]) for value in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("score input repeats a context")
    return sorted(selected, key=lambda value: str(value["context_id"]))


def _extract_context(
    *,
    np: Any,
    campaign_dir: Path,
    output_dir: Path,
    context: dict[str, Any],
    score_document: dict[str, Any],
    verify_hash: bool,
) -> dict[str, Any]:
    context_id = str(context["context_id"])
    source_dir = campaign_dir / "contexts" / context_id
    local_path = source_dir / "local.json"
    local = load_json(local_path)
    archive_record = local.get("interface_archive", {}).get("artifact")
    if not isinstance(archive_record, dict):
        raise ValueError(f"{context_id} has no interface archive record")
    archive_path = source_dir / str(archive_record["name"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"missing interface archive: {archive_path}")
    actual_hash = file_sha256(archive_path) if verify_hash else None
    if verify_hash and actual_hash != archive_record["sha256"]:
        raise ValueError(f"interface archive hash mismatch: {context_id}")

    feature_record = local.get("feature_archive")
    if not isinstance(feature_record, dict):
        raise ValueError(f"{context_id} has no SAFE feature archive record")
    feature_path = source_dir / str(feature_record["name"])
    if not feature_path.is_file():
        raise FileNotFoundError(f"missing SAFE feature archive: {feature_path}")
    feature_hash = file_sha256(feature_path) if verify_hash else None
    if verify_hash and feature_hash != feature_record["sha256"]:
        raise ValueError(f"SAFE feature archive hash mismatch: {context_id}")
    with feature_path.open("rb") as file:
        features = pickle.load(file)
    clean_feature = features["clean_hidden_states"].detach().float()
    fault_features = features["faulted_hidden_states_by_layer"]
    if sorted(int(layer) for layer in fault_features) != list(range(32)):
        raise ValueError(f"SAFE feature archive does not cover every block: {context_id}")
    safe_features = {
        "clean": clean_feature.numpy(),
        "fault": np.stack(
            [fault_features[layer].detach().float().numpy() for layer in range(32)]
        ),
    }

    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = context_arrays(
            np, archive, local, score_document, safe_features=safe_features
        )

    captured_record = local.get("captured_context_archive")
    captured_artifact = (
        captured_record.get("artifact") if isinstance(captured_record, dict) else None
    )
    if not isinstance(captured_artifact, dict):
        raise ValueError(f"{context_id} has no captured physical-state archive")
    captured_path = source_dir / str(captured_artifact["name"])
    if not captured_path.is_file():
        raise FileNotFoundError(f"missing physical-state archive: {captured_path}")
    captured_hash = file_sha256(captured_path) if verify_hash else None
    if verify_hash and captured_hash != captured_artifact["sha256"]:
        raise ValueError(f"physical-state archive hash mismatch: {context_id}")
    retained_observations = []
    with np.load(captured_path, allow_pickle=False) as captured:
        arrays["physical_simulator_state"] = captured["simulator_state"].copy()
        for observation in captured_record["observations"]:
            if str(observation["dtype"]).endswith("u1"):
                continue
            source_key = str(observation["archive_key"])
            output_key = f"physical_{source_key}"
            arrays[output_key] = captured[source_key].copy()
            retained_observations.append(
                {
                    "name": observation["name"],
                    "source_archive_key": source_key,
                    "atlas_key": output_key,
                    "shape": observation["shape"],
                    "dtype": observation["dtype"],
                    "sha256": observation["sha256"],
                }
            )

    result_dir = output_dir / "contexts" / context_id
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = result_dir / "interface_atlas.npz"
    write_npz_atomic(artifact_path, np, arrays)
    result = {
        "schema_version": 1,
        "status": "complete",
        "context": context,
        "source": {
            "local_json_sha256": file_sha256(local_path),
            "interface_archive": archive_record,
            "interface_archive_hash_verified": verify_hash,
            "interface_archive_actual_sha256": actual_hash,
            "safe_feature_archive": feature_record,
            "safe_feature_archive_hash_verified": verify_hash,
            "safe_feature_archive_actual_sha256": feature_hash,
            "captured_physical_state_archive": captured_artifact,
            "captured_physical_state_hash_verified": verify_hash,
            "captured_physical_state_actual_sha256": captured_hash,
        },
        "retained_physical_observations": retained_observations,
        "artifact": artifact_record(artifact_path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str}
            for name, value in arrays.items()
        },
    }
    write_json_atomic(result_dir / "context.complete.json", result)
    (result_dir / "context.error.json").unlink(missing_ok=True)
    return result


def run(args: argparse.Namespace, np: Any) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_document = load_json(args.scores)
    campaign_run = load_json(args.campaign_dir / "run.json")
    contexts = _selected_contexts(score_document, args.split)
    if args.context_ids:
        requested = set(args.context_ids)
        known = {str(value["context_id"]) for value in contexts}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"requested contexts are not in the selected split: {unknown}")
        contexts = [
            value for value in contexts if str(value["context_id"]) in requested
        ]
    project_root = Path(__file__).resolve().parents[1]
    run_record = {
        "schema_version": 1,
        "analysis": "exact OpenVLA interface atlas extraction",
        "analysis_split": args.split,
        "worker_shard": campaign_run["worker_shard"],
        "context_ids": [str(value["context_id"]) for value in contexts],
        "source_campaign": {
            "directory": str(args.campaign_dir.resolve()),
            "run_sha256": file_sha256(args.campaign_dir / "run.json"),
        },
        "source_scores": {
            "path": str(args.scores.resolve()),
            "sha256": file_sha256(args.scores),
        },
        "code": {
            "revision": git_revision(project_root),
            "dirty": git_dirty(project_root),
            "extractor_sha256": file_sha256(Path(__file__)),
        },
        "execution": {"resumable_unit": "one context"},
        "cut_contract": {
            "injection": "source replacement minus clean post-block residual",
            "immediate": (
                "signed post-block residual and current-token post-rotary key/value "
                "changes one language block after each replacement"
            ),
            "propagation_path": (
                "signed current-token residual and differential key/value state at "
                "every mechanically downstream language-block boundary"
            ),
            "monitor_endpoint": (
                "signed exported OpenVLA hidden-feature change at the final action "
                "token consumed by the frozen SAFE-MLP"
            ),
            "outcomes": (
                "frozen SAFE-MLP scores, alarm windows, executed command, and terminal "
                "task evidence from the scored campaign"
            ),
            "physical_context": (
                "exact pre-fault flattened MuJoCo state and every named numeric "
                "observation; camera arrays are omitted from this reduced artifact"
            ),
        },
    }
    write_json_atomic(args.output_dir / "run.json", run_record)

    def process(context: dict[str, Any]) -> dict[str, Any]:
        context_id = str(context["context_id"])
        result_dir = args.output_dir / "contexts" / context_id
        complete_path = result_dir / "context.complete.json"
        artifact_path = result_dir / "interface_atlas.npz"
        if args.resume and complete_path.is_file() and artifact_path.is_file():
            return {"context_id": context_id, "status": "complete"}
        try:
            _extract_context(
                np=np,
                campaign_dir=args.campaign_dir,
                output_dir=args.output_dir,
                context=context,
                score_document=score_document,
                verify_hash=not args.skip_source_hash_check,
            )
            return {"context_id": context_id, "status": "complete"}
        except Exception as error:
            result_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": 1,
                "status": "error",
                "context": context,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            write_json_atomic(result_dir / "context.error.json", failure)
            return {
                "context_id": context_id,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    statuses = []
    for context in contexts:
        statuses.append(process(context))
        write_json_atomic(
            args.output_dir / "status.json",
            {
                "schema_version": 1,
                "planned_contexts": len(contexts),
                "processed_contexts": len(statuses),
                "status_counts": dict(
                    sorted(Counter(value["status"] for value in statuses).items())
                ),
                "contexts": statuses,
            },
        )

    result = load_json(args.output_dir / "status.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = _arguments()
    import numpy as np

    run(args, np)


if __name__ == "__main__":
    main()
