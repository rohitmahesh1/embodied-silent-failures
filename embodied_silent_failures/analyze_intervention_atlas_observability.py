from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.intervention_atlas_observability import (
    PRIMARY_WINDOW_STEPS,
    observability_audit,
    observability_rows,
    physical_divergence_audit,
    split_auc_difference_bootstrap,
    split_summary,
)
from embodied_silent_failures.language_scoring import physical_score_index
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure fixed-window SAFE evidence after atlas interventions."
    )
    parser.add_argument("--site-analysis", action="append", required=True, type=Path)
    parser.add_argument("--site-scores", action="append", required=True, type=Path)
    parser.add_argument(
        "--physical-analysis", action="append", required=True, type=Path
    )
    parser.add_argument("--physical-scores", action="append", required=True, type=Path)
    parser.add_argument("--mechanisms", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def _load_sites(
    analysis_paths: list[Path], score_paths: list[Path], np: Any
) -> tuple[
    list[tuple[str, list[dict[str, Any]]]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if len(analysis_paths) != len(score_paths):
        raise ValueError("site analysis and score archive counts differ")
    groups = []
    score_index = {}
    sources = []
    monitor = None
    for analysis_path, score_path in zip(
        analysis_paths, score_paths, strict=True
    ):
        analysis = load_json(analysis_path)
        if file_sha256(score_path) != analysis["score_archive"]["sha256"]:
            raise ValueError(f"site score hash differs from {analysis_path}")
        current_monitor = analysis["monitor"]
        monitor = current_monitor if monitor is None else monitor
        if current_monitor != monitor:
            raise ValueError("site analyses used different SAFE monitors")
        with np.load(score_path, allow_pickle=False) as archive:
            record_ids = archive["record_ids"].astype(str)
            lengths = archive["lengths"].astype(int)
            scores = archive["faulted_evidence_scores"]
        for index, record_id in enumerate(record_ids):
            if record_id in score_index:
                raise ValueError(f"duplicate site score record {record_id}")
            score_index[record_id] = scores[index, : lengths[index]]
        records = analysis["records"]
        missing = {
            str(record["record_id"])
            for record in records
            if str(record["record_id"]) not in score_index
        }
        if missing:
            raise ValueError(f"site score archive is missing {len(missing)} records")
        groups.append((str(analysis["analysis_split"]), records))
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
    return groups, score_index, sources, monitor


def _load_physical(
    analysis_paths: list[Path], score_paths: list[Path], np: Any
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(analysis_paths) != len(score_paths):
        raise ValueError("physical analysis and score archive counts differ")
    combined = {}
    sources = []
    monitor = None
    for analysis_path, score_path in zip(
        analysis_paths, score_paths, strict=True
    ):
        analysis = load_json(analysis_path)
        current_monitor = analysis["monitor"]
        monitor = current_monitor if monitor is None else monitor
        if current_monitor != monitor:
            raise ValueError("physical analyses used different SAFE monitors")
        indexed, _bands, _alphas = physical_score_index(analysis, score_path, np)
        overlap = set(combined).intersection(indexed)
        if overlap:
            raise ValueError(f"duplicate physical SAFE runs: {len(overlap)}")
        combined.update(indexed)
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
    return combined, sources, monitor


def _load_mechanisms(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    indexed = {}
    sources = []
    for path in paths:
        artifact = load_json(path)
        for pair in artifact["physical_pairs"]:
            run = str(pair["run"])
            if run in indexed:
                raise ValueError(f"duplicate mechanism record {run}")
            indexed[run] = pair
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return indexed, sources


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 1:
        raise ValueError("at least one bootstrap sample is required")

    import numpy as np

    site_groups, site_scores, site_sources, site_monitor = _load_sites(
        args.site_analysis, args.site_scores, np
    )
    physical_scores, physical_sources, physical_monitor = _load_physical(
        args.physical_analysis, args.physical_scores, np
    )
    mechanisms, mechanism_sources = _load_mechanisms(args.mechanisms)
    if site_monitor != physical_monitor:
        raise ValueError("site and physical traces used different SAFE monitors")
    site_rows, physical_rows = observability_rows(
        site_groups, site_scores, physical_scores
    )
    output = {
        "schema_version": 1,
        "analysis": "fixed-window fault-specific SAFE evidence audit",
        "analysis_code": {
            **git_state(Path(__file__).resolve().parents[1]),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("intervention_atlas_observability.py")
            ),
        },
        "analysis_contract": {
            "status": "exploratory post-hoc analysis after opening the holdout",
            "question": (
                "does SAFE accumulate fault-specific evidence before terminal "
                "cumulative alarms become confounded by observation horizon"
            ),
            "primary_unit": (
                "one distinct non-control physical continuation; this avoids "
                "counting graph sites that selected the same robot command as "
                "independent physical outcomes"
            ),
            "secondary_unit": (
                "eligible sampled graph-site interventions, including sites whose "
                "physical behavior matched control; uncertainty is clustered by "
                "clean-rollout trajectory"
            ),
            "site_unit_audit": (
                "all eligible site units and the subset mapped to non-control robot "
                "behavior are reported separately because numerous no-command-change "
                "sites can make zero evidence shift appear predictive"
            ),
            "primary_window_steps": PRIMARY_WINDOW_STEPS,
            "primary_window_basis": (
                "SAFE scoring already declared a 25-step post-fault alarm window; "
                "1, 5, 10, 50, and 100 steps are reported as temporal diagnostics"
            ),
            "response": (
                "SAFE score newly accumulated from the fault step through the end "
                "of the fixed window in the faulted rollout, minus the same quantity "
                "from its matched successful control"
            ),
            "score_provenance": (
                "SAFE commit b6036abe, failure_prob/model/indep.py::IndepModel.forward, "
                "projects each one-step feature and applies torch.cumsum because the "
                "frozen configuration has cumsum=true and rmean=false"
            ),
            "interpretation": (
                "positive response means the faulted branch accumulated more SAFE "
                "anomaly evidence than control; subtracting each trace at the step "
                "before injection removes inherited cumulative history without "
                "changing or retraining SAFE"
            ),
            "limits": (
                "post-fault evidence can reflect downstream physical consequences, "
                "not only immediate feature corruption; terminal policy failure is "
                "used to test separation, and a new trajectory set is required for "
                "confirmatory inference"
            ),
        },
        "monitor": site_monitor,
        "sources": {
            "site": site_sources,
            "physical": physical_sources,
            "mechanisms": mechanism_sources,
        },
        "audit": observability_audit(site_rows, physical_rows),
        "physical_continuations": split_summary(
            physical_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "physical_primary_window_replication": split_auc_difference_bootstrap(
            physical_rows,
            samples=args.bootstrap_samples,
            seed=args.seed + 20,
        ),
        "physical_divergence_association": physical_divergence_audit(
            physical_rows, mechanisms
        ),
        "site_interventions": split_summary(
            site_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 10,
        ),
        "noncontrol_site_interventions": split_summary(
            [row for row in site_rows if not row["physical_run_is_control"]],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 30,
        ),
        "noncontrol_site_primary_window_replication": (
            split_auc_difference_bootstrap(
                [row for row in site_rows if not row["physical_run_is_control"]],
                samples=args.bootstrap_samples,
                seed=args.seed + 40,
            )
        ),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
