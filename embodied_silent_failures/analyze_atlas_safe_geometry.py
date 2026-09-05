from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import file_sha256, git_state, load_json
from embodied_silent_failures.safe_trajectory_analysis import (
    attach_temporal_geometry,
    split_geometry_summary,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare physical outcomes in the frozen SAFE monitor's geometry."
    )
    parser.add_argument("--geometry", action="append", required=True, type=Path)
    parser.add_argument("--arrays", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if len(args.geometry) != len(args.arrays):
        raise ValueError("geometry document and array archive counts differ")
    if args.bootstrap_samples < 1:
        raise ValueError("at least one bootstrap sample is required")

    import numpy as np

    records = []
    errors = []
    sources = []
    monitor = None
    window_steps = None
    seen = set()
    for document_path, array_path in zip(args.geometry, args.arrays, strict=True):
        document = load_json(document_path)
        if file_sha256(array_path) != document["array_archive"]["sha256"]:
            raise ValueError(f"array archive differs from {document_path}")
        with np.load(array_path, allow_pickle=False) as archive:
            physical_runs = archive["physical_runs"].astype(str).tolist()
            temporal_arrays = {
                name: archive[name]
                for name in (
                    "selected_feature_l2",
                    "monitor_increment_delta",
                    "clean_gradient_dot_delta",
                )
            }
        document_runs = [str(record["physical_run"]) for record in document["records"]]
        if physical_runs != document_runs:
            raise ValueError(f"array row order differs from {document_path}")
        overlap = seen.intersection(document_runs)
        if overlap:
            raise ValueError(f"duplicate physical continuations: {len(overlap)}")
        seen.update(document_runs)

        current_monitor = document["provenance"]["monitor"]
        monitor = current_monitor if monitor is None else monitor
        if current_monitor != monitor:
            raise ValueError("geometry documents used different SAFE monitors")
        current_window = int(document["analysis_contract"]["window_steps"])
        window_steps = current_window if window_steps is None else window_steps
        if current_window != window_steps:
            raise ValueError("geometry documents used different trajectory windows")
        records.extend(
            attach_temporal_geometry(record, temporal_arrays, index)
            for index, record in enumerate(document["records"])
        )
        errors.extend(document["error_records"])
        sources.append(
            {
                "geometry": {
                    "path": str(document_path.resolve()),
                    "sha256": file_sha256(document_path),
                },
                "arrays": {
                    "path": str(array_path.resolve()),
                    "sha256": file_sha256(array_path),
                },
            }
        )

    output = {
        "schema_version": 1,
        "analysis": "mechanisms of weak SAFE response after physical divergence",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("safe_trajectory_analysis.py")
            ),
            "geometry_methods_sha256": file_sha256(
                Path(__file__).with_name("safe_trajectory_geometry.py")
            ),
        },
        "analysis_contract": {
            "status": "exploratory post-hoc analysis after opening the holdout",
            "question": (
                "when SAFE has little net response to a consequential fault, did its "
                "input representation barely move, move in a direction the frozen "
                "monitor is locally insensitive to, or produce responses that cancel"
            ),
            "unit": "one distinct non-control physical continuation",
            "window_steps": window_steps,
            "comparisons": (
                "larger-value ROC AUC and trajectory-clustered uncertainty are "
                "reported separately for failure versus success and, among failures, "
                "silent versus eventually detected outcomes"
            ),
            "quiet_failure_audit": (
                "the lowest split-specific quarter of absolute net SAFE response is "
                "described only to separate weak movement, weak alignment, and "
                "temporal cancellation; it is not a fitted decision threshold"
            ),
            "guardrail": (
                "gradient projection is a mechanistic description of the existing "
                "SAFE projector, not a new detector or an independent predictor"
            ),
        },
        "monitor": monitor,
        "sources": sources,
        "coverage": {
            "complete_physical_continuations": len(records),
            "errors": len(errors),
            "error_types": {
                name: sum(record["error_type"] == name for record in errors)
                for name in sorted({record["error_type"] for record in errors})
            },
        },
        "results": split_geometry_summary(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
