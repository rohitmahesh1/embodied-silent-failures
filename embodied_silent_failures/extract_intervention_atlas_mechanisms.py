from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.atlas_mechanism_extraction import (
    extract_context_state,
    extract_physical_pair,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize state and recovery measurements from atlas trajectories."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--site-analysis", action="append", required=True, type=Path)
    parser.add_argument("--physical-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    import numpy as np

    site_analyses = [load_json(path) for path in args.site_analysis]
    physical_analysis = load_json(args.physical_analysis)
    physical_index = {
        str(record["run"]): record for record in physical_analysis["records"]
    }
    eligible = {}
    contexts = {}
    for analysis in site_analyses:
        for record in analysis["records"]:
            if not record.get("primary_eligible"):
                continue
            run = str(record["physical_run"])
            eligible[run] = physical_index[run]
            context = dict(record["context"])
            previous = contexts.setdefault(str(record["context_id"]), context)
            if previous != context:
                raise ValueError(f"inconsistent context {record['context_id']}")

    context_records = [
        extract_context_state(args.campaign_dir, contexts[context_id], np)
        for context_id in sorted(contexts)
    ]
    physical_pairs = [
        extract_physical_pair(
            args.campaign_dir,
            eligible[run],
            contexts[str(eligible[run]["run"]).split("-", 1)[0]],
            np,
        )
        for run in sorted(eligible)
    ]
    output = {
        "schema_version": 1,
        "analysis": "mechanical state and recovery summary for atlas interventions",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("atlas_mechanism_extraction.py")
            ),
        },
        "analysis_contract": {
            "context_state": (
                "raw simulator state and the recorder's named object-state and "
                "robot0_proprio-state arrays immediately before intervention"
            ),
            "recovery_comparison": (
                "faulted and successful-control trajectories compared at the same "
                "policy step; unavailable horizons are retained as absent"
            ),
            "excluded": (
                "camera arrays are not loaded; no handwritten object groups or task "
                "success geometry are introduced"
            ),
        },
        "sources": {
            "campaign_dir": str(args.campaign_dir.resolve()),
            "site_analyses": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in args.site_analysis
            ],
            "physical_analysis": {
                "path": str(args.physical_analysis.resolve()),
                "sha256": file_sha256(args.physical_analysis),
            },
        },
        "coverage": {
            "contexts": len(context_records),
            "physical_pairs": len(physical_pairs),
        },
        "contexts": context_records,
        "physical_pairs": physical_pairs,
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
