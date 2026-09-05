from __future__ import annotations

import argparse
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import file_sha256, git_state, load_json
from embodied_silent_failures.recovery_evidence import (
    HORIZONS,
    monitor_window_features,
)
from embodied_silent_failures.recovery_observability import (
    analyze_recovery_observability,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test when policy, physical-recovery, and SAFE evidence first distinguish "
            "terminal failures in the OpenVLA intervention atlas."
        )
    )
    parser.add_argument("--mechanisms", action="append", required=True, type=Path)
    parser.add_argument("--safe-geometry", action="append", required=True, type=Path)
    parser.add_argument("--safe-arrays", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def _load_shard(np, mechanism_path: Path, geometry_path: Path, array_path: Path):
    mechanisms = load_json(mechanism_path)
    geometry = load_json(geometry_path)
    if file_sha256(array_path) != geometry["array_archive"]["sha256"]:
        raise ValueError(f"SAFE arrays differ from {geometry_path}")

    pairs = {str(row["run"]): row for row in mechanisms["physical_pairs"]}
    contexts = {
        str(row["context_id"]): row for row in mechanisms["contexts"]
    }
    geometry_records = {
        str(row["physical_run"]): row for row in geometry["records"]
    }
    with np.load(array_path, allow_pickle=False) as archive:
        runs = archive["physical_runs"].astype(str).tolist()
        if runs != [str(row["physical_run"]) for row in geometry["records"]]:
            raise ValueError(f"SAFE array order differs from {geometry_path}")
        arrays = {
            name: archive[name]
            for name in (
                "monitor_increment_delta",
                "absolute_monitor_increment_delta",
                "selected_feature_normalized_l2",
            )
        }
        rows = []
        for index, run in enumerate(runs):
            pair = pairs[run]
            context = contexts[str(pair["context_id"])]
            geometry_record = geometry_records[run]
            if bool(pair["faulted_success"]) == bool(geometry_record["policy_failure"]):
                raise ValueError(f"physical outcome disagrees for {run}")
            rows.append(
                {
                    "physical_run": run,
                    "context_id": str(pair["context_id"]),
                    "task_id": int(pair["task_id"]),
                    "episode_index": int(pair["episode_index"]),
                    "analysis_split": str(pair["analysis_split"]),
                    "phase": str(pair["phase"]),
                    "phase_fraction": float(context["phase_fraction"]),
                    "fault_step": int(pair["policy_step"]),
                    "faulted_length": int(pair["faulted_length"]),
                    "control_length": int(pair["control_length"]),
                    "policy_failure": not bool(pair["faulted_success"]),
                    "comparisons": pair["comparisons"],
                    "monitor_features": {
                        str(horizon): monitor_window_features(
                            arrays, index, horizon
                        )
                        for horizon in HORIZONS
                    },
                }
            )
    return rows, {
        "mechanisms": {
            "path": str(mechanism_path.resolve()),
            "sha256": file_sha256(mechanism_path),
        },
        "safe_geometry": {
            "path": str(geometry_path.resolve()),
            "sha256": file_sha256(geometry_path),
        },
        "safe_arrays": {
            "path": str(array_path.resolve()),
            "sha256": file_sha256(array_path),
        },
    }


def main() -> None:
    args = _arguments()
    if not (
        len(args.mechanisms) == len(args.safe_geometry) == len(args.safe_arrays)
    ):
        raise ValueError("mechanism and SAFE shard counts differ")
    if args.folds < 2:
        raise ValueError("at least two folds are required")
    if args.bootstrap_samples < 1:
        raise ValueError("at least one bootstrap sample is required")

    import numpy as np

    rows = []
    sources = []
    seen = set()
    for paths in zip(
        args.mechanisms,
        args.safe_geometry,
        args.safe_arrays,
        strict=True,
    ):
        shard_rows, source = _load_shard(np, *paths)
        runs = {row["physical_run"] for row in shard_rows}
        overlap = seen.intersection(runs)
        if overlap:
            raise ValueError(f"duplicate physical continuations: {len(overlap)}")
        seen.update(runs)
        rows.extend(shard_rows)
        sources.append(source)

    analysis = analyze_recovery_observability(
        rows,
        folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = {
        "schema_version": 1,
        "analysis": "early recovery and monitor observability after an internal fault",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "evidence_methods_sha256": file_sha256(
                Path(__file__).with_name("recovery_evidence.py")
            ),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("recovery_observability.py")
            ),
        },
        "analysis_contract": {
            "status": (
                "exploratory post-hoc analysis; the original holdout had already "
                "been opened before this question was fixed"
            ),
            "question": (
                "when does a one-step internal fault become distinguishable as an "
                "eventual task failure, and do policy behavior, physical recovery, "
                "or SAFE evidence reveal it first"
            ),
            "unit": "one distinct non-control physical continuation",
            "outcome": "terminal task failure; no terminal outcome enters a feature",
            "horizons": list(HORIZONS),
            "model": (
                "fixed L2 logistic regression with C=1; development cross-validation "
                "and uncertainty keep complete task/episode trajectories together"
            ),
            "context_control": "task identity and continuous fault phase",
            "policy_evidence": (
                "command distance, changed action-token fraction, and action-entropy "
                "difference at mechanically recorded checkpoints"
            ),
            "physical_evidence": (
                "named object, robot proprioceptive, and full simulator-state distance "
                "from the successful control at each checkpoint"
            ),
            "recovery_test": (
                "compare the complete checkpoint path with the current physical distance; "
                "the model sees successive values rather than a hand-written recovery score"
            ),
            "safe_evidence": (
                "paired SAFE-input displacement, signed response, absolute response, "
                "and temporal cancellation over the same early window"
            ),
            "same_scene_test": (
                "rank failed and successful branches launched from the identical saved "
                "context; bootstrap uncertainty resamples whole clean trajectories"
            ),
            "direct_diagnostics": (
                "report every declared policy, physical-distance, physical-growth, "
                "and SAFE scalar in both original splits; these diagnostics were "
                "added after the multivariate result and are not a model-selection step"
            ),
            "limits": (
                "same-time state distance cannot distinguish phase delay from departure "
                "from the successful state path; this analysis diagnoses the existing "
                "atlas and cannot provide an untouched confirmatory result"
            ),
        },
        "sources": sources,
        **analysis,
    }
    write_json_atomic(args.output, output)


if __name__ == "__main__":
    main()
