from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.atlas_path_alignment import (
    ALIGNMENT_WINDOWS,
    HORIZONS,
    STATE_STREAMS,
    extract_pair_alignments,
    load_state_trajectory,
    one_completion,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare atlas continuations with nearby states on their successful "
            "control paths."
        )
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--mechanisms", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _source_record(result: dict[str, Any], path: Path) -> dict[str, Any]:
    artifact = result["trajectory_archive"]["artifact"]
    return {
        "path": str(path.resolve()),
        "declared_sha256": str(artifact["sha256"]),
        "bytes": int(artifact["bytes"]),
    }


def main() -> None:
    args = _arguments()
    import numpy as np

    mechanisms = load_json(args.mechanisms)
    contexts = {
        str(record["context_id"]): record for record in mechanisms["contexts"]
    }
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for physical in mechanisms["physical_pairs"]:
        by_context[str(physical["context_id"])].append(physical)

    records = []
    errors = []
    control_sources = []
    faulted_sources = []
    for context_id in sorted(by_context):
        context = contexts[context_id]
        control_run = f"{context_id}-control"
        try:
            control_result, control_path = one_completion(
                args.campaign_dir / "attempts" / control_run
            )
            control = load_state_trajectory(np, control_result, control_path)
            control_sources.append(
                {"run": control_run, **_source_record(control_result, control_path)}
            )
        except Exception as error:
            errors.append(
                {
                    "context_id": context_id,
                    "run": control_run,
                    "stage": "control",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue

        for physical in sorted(by_context[context_id], key=lambda row: str(row["run"])):
            run = str(physical["run"])
            try:
                faulted_result, faulted_path = one_completion(
                    args.campaign_dir / "attempts" / run
                )
                faulted = load_state_trajectory(np, faulted_result, faulted_path)
                records.append(
                    extract_pair_alignments(
                        np, physical, context, faulted, control
                    )
                )
                faulted_sources.append(
                    {"run": run, **_source_record(faulted_result, faulted_path)}
                )
            except Exception as error:
                errors.append(
                    {
                        "context_id": context_id,
                        "run": run,
                        "stage": "faulted",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    output = {
        "schema_version": 1,
        "analysis": "phase-aligned physical recovery paths in the OpenVLA atlas",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("atlas_path_alignment.py")
            ),
        },
        "extraction_contract": {
            "question": (
                "whether a perturbed branch leaves the successful state path or "
                "mainly occupies a different point along that path"
            ),
            "state_streams": list(STATE_STREAMS),
            "horizons": list(HORIZONS),
            "alignment_windows": list(ALIGNMENT_WINDOWS),
            "candidate_path": (
                "successful-control snapshots from the intervention step through "
                "the symmetric window around each faulted snapshot; no pre-fault "
                "snapshot is eligible"
            ),
            "nearest_state": (
                "minimum symmetric normalized L2 distance, with ties resolved by "
                "smallest absolute then earliest signed policy-step offset"
            ),
            "selection": (
                "all physical continuations in the supplied mechanism artifact; "
                "terminal outcome is not used to align states"
            ),
            "failure_handling": (
                "a malformed branch is retained in errors and does not stop later "
                "contexts"
            ),
        },
        "sources": {
            "campaign_dir": str(args.campaign_dir.resolve()),
            "mechanisms": {
                "path": str(args.mechanisms.resolve()),
                "sha256": file_sha256(args.mechanisms),
            },
            "control_trajectories": control_sources,
            "faulted_trajectories": faulted_sources,
        },
        "coverage": {
            "declared_contexts": len(by_context),
            "loaded_controls": len(control_sources),
            "declared_physical_pairs": sum(map(len, by_context.values())),
            "extracted_physical_pairs": len(records),
            "errors": len(errors),
        },
        "records": records,
        "errors": errors,
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
