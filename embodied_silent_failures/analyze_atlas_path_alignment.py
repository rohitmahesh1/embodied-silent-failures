from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.atlas_path_alignment import (
    ALIGNMENT_WINDOWS,
    HORIZONS,
    STATE_STREAMS,
)
from embodied_silent_failures.atlas_path_analysis import (
    PRIMARY_HORIZON,
    PRIMARY_STREAM,
    PRIMARY_WINDOW,
    analyze_path_alignment,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether atlas failures leave the successful physical state path "
            "or mainly move out of phase along it."
        )
    )
    parser.add_argument("--alignment", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 1:
        raise ValueError("at least one bootstrap sample is required")

    records = []
    errors = []
    seen = set()
    sources = []
    excluded_reference_controls = 0
    for path in args.alignment:
        artifact = load_json(path)
        for row in artifact["records"]:
            run = str(row["run"])
            if run == f"{row['context_id']}-control":
                excluded_reference_controls += 1
                continue
            if run in seen:
                raise ValueError(f"duplicate physical continuation: {run}")
            seen.add(run)
            records.append(row)
        errors.extend(artifact["errors"])
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})

    analysis = analyze_path_alignment(
        records, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    output = {
        "schema_version": 1,
        "analysis": "phase-aligned recovery audit of OpenVLA atlas continuations",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("atlas_path_analysis.py")
            ),
        },
        "analysis_contract": {
            "status": (
                "exploratory post-hoc analysis; both original atlas splits had "
                "already been opened before this question was fixed"
            ),
            "question": (
                "whether successful perturbations remain near the successful state "
                "path with a timing offset while failed perturbations leave it"
            ),
            "outcome": "terminal task failure; outcome is never used for alignment",
            "unit": "one distinct non-control physical continuation",
            "state_streams": list(STATE_STREAMS),
            "horizons": list(HORIZONS),
            "alignment_windows": list(ALIGNMENT_WINDOWS),
            "primary_test": {
                "horizon": PRIMARY_HORIZON,
                "window": PRIMARY_WINDOW,
                "stream": PRIMARY_STREAM,
                "comparison": (
                    "failure ranking by nearest successful-path distance versus "
                    "same-clock control distance"
                ),
            },
            "sensitivity": (
                "repeat the 25-step comparison with 5- and 10-step windows and "
                "with LIBERO's named object and robot state streams"
            ),
            "same_scene_test": (
                "failed and successful branches are ranked only within an identical "
                "saved context; uncertainty resamples complete clean trajectories"
            ),
            "reentry_test": (
                "measure change from 5 to 25 steps in nearest-path distance and "
                "signed phase lag; no binary recovery threshold is chosen"
            ),
            "limits": (
                "distance is geometric rather than task-semantic, a single successful "
                "control supplies each reference path, and windowed nearest-neighbor "
                "alignment does not establish causal recoverability"
            ),
            "posthoc_robustness": (
                "because the unexplained-fraction result was inspected after scoring, "
                "report its value again after removing exact same-clock rejoins"
            ),
        },
        "sources": sources,
        "excluded_reference_controls": excluded_reference_controls,
        "extraction_errors": errors,
        **analysis,
    }
    write_json_atomic(args.output, output)
    primary = output["declared_uncertainty_panels"][
        f"h{PRIMARY_HORIZON}:w{PRIMARY_WINDOW}:{PRIMARY_STREAM}"
    ]
    print(
        json.dumps(
            {"population": output["population"], "primary": primary}, indent=2
        )
    )


if __name__ == "__main__":
    main()
