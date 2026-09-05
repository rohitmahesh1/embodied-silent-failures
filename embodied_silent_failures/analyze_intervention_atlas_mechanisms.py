from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.intervention_atlas_mechanisms import (
    alarm_horizon_audit,
    analyze_mechanism_models,
    context_outcome_audit,
    evidence_substitution_audit,
    monitor_process_audit,
    physical_failure_rows,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why SAFE misses atlas-induced policy failures."
    )
    parser.add_argument("--site-analysis", action="append", required=True, type=Path)
    parser.add_argument("--site-scores", action="append", required=True, type=Path)
    parser.add_argument("--mechanisms", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def _site_data(analysis_paths: list[Path], score_paths: list[Path]):
    import numpy as np

    if len(analysis_paths) != len(score_paths):
        raise ValueError("site analysis and score archive counts differ")
    records = []
    score_index = {}
    sources = []
    alpha = None
    for analysis_path, score_path in zip(
        analysis_paths, score_paths, strict=True
    ):
        analysis = load_json(analysis_path)
        if file_sha256(score_path) != analysis["score_archive"]["sha256"]:
            raise ValueError(f"site score hash differs from {analysis_path}")
        with np.load(score_path, allow_pickle=False) as archive:
            alphas = archive["alphas"].astype(float)
            record_ids = archive["record_ids"]
            lengths = archive["lengths"]
            faulted_scores = archive["faulted_evidence_scores"]
            clean_scores = archive["clean_evidence_same_suffix_scores"]
            bands = archive["bands"]
        primary = float(analysis["monitor"]["primary_alpha"])
        alpha_index = int(np.argmin(np.abs(alphas - primary)))
        if not np.isclose(alphas[alpha_index], primary):
            raise ValueError("primary SAFE alpha is absent from score archive")
        alpha = primary if alpha is None else alpha
        if not math.isclose(alpha, primary):
            raise ValueError("site analyses use different primary SAFE alphas")
        for index, record_id in enumerate(record_ids):
            identity = str(record_id)
            if identity in score_index:
                raise ValueError(f"duplicate score record {identity}")
            score_index[identity] = {
                "length": int(lengths[index]),
                "faulted": faulted_scores[index],
                "clean_same_suffix": clean_scores[index],
                "band": bands[alpha_index],
            }
        records.extend(analysis["records"])
        sources.append(
            {
                "analysis": {
                    "path": str(analysis_path.resolve()),
                    "sha256": file_sha256(analysis_path),
                },
                "scores": {
                    "path": str(score_path.resolve()),
                    "sha256": file_sha256(score_path),
                },
            }
        )
    return records, score_index, float(alpha), sources


def _mechanism_data(paths: list[Path]):
    contexts = {}
    pairs = {}
    sources = []
    for path in paths:
        artifact = load_json(path)
        for context in artifact["contexts"]:
            identity = str(context["context_id"])
            if identity in contexts:
                raise ValueError(f"duplicate mechanism context {identity}")
            contexts[identity] = context
        for pair in artifact["physical_pairs"]:
            run = str(pair["run"])
            if run in pairs:
                raise ValueError(f"duplicate mechanism physical run {run}")
            pairs[run] = pair
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return contexts, pairs, sources


def main() -> None:
    args = _arguments()
    if args.folds < 2:
        raise ValueError("at least two trajectory folds are required")
    if args.bootstrap_samples < 1:
        raise ValueError("at least one bootstrap sample is required")

    records, scores, alpha, site_sources = _site_data(
        args.site_analysis, args.site_scores
    )
    contexts, pairs, mechanism_sources = _mechanism_data(args.mechanisms)
    physical_rows = physical_failure_rows(
        records, contexts, pairs, alpha=alpha
    )
    output = {
        "schema_version": 1,
        "analysis": "post-hoc mechanism audit of SAFE misses in the intervention atlas",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("intervention_atlas_mechanisms.py")
            ),
        },
        "analysis_contract": {
            "status": "exploratory post-hoc analysis after opening the holdout",
            "primary_unit": (
                "one distinct failed physical continuation; site-level evidence "
                "substitution is reported separately"
            ),
            "model_question": (
                "whether exact pre-fault state, early recovery, or observation "
                "horizon adds held-out information beyond task and fault timing"
            ),
            "monitor_process_question": (
                "whether eventual alarm status is explained by immediate fault "
                "evidence or by the later task-dependent cumulative score process"
            ),
            "context_question": (
                "whether alarm outcomes cluster within exact intervention contexts "
                "beyond task and early/middle/late fault phase"
            ),
            "state_representation": (
                "all dimensions of LIBERO's named object-state and "
                "robot0_proprio-state observations, separated by task and scaled "
                "from development data; no hand-selected semantic dimensions"
            ),
            "recovery_representation": (
                "mechanically recorded object, robot, command, action-token, and "
                "entropy differences from the matched successful control at 0, 1, "
                "5, 10, and 25 policy steps"
            ),
            "limits": (
                "the state model excludes camera pixels and the recovery model uses "
                "post-fault information for explanation, not intervention-time "
                "prediction; confirmation requires new trajectories"
            ),
        },
        "sources": {
            "site_data": site_sources,
            "mechanisms": mechanism_sources,
        },
        "evidence_substitution": evidence_substitution_audit(
            records, scores, alpha=alpha
        ),
        "alarm_horizon": alarm_horizon_audit(physical_rows),
        "monitor_process": monitor_process_audit(
            records, scores, contexts, alpha=alpha
        ),
        "context_outcomes": context_outcome_audit(
            physical_rows,
            permutations=args.bootstrap_samples,
            seed=args.seed,
        ),
        "state_and_recovery": analyze_mechanism_models(
            physical_rows,
            folds=args.folds,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
