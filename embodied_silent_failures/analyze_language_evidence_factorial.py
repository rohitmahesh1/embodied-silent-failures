from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_evidence_factorial import (
    ALARM_HORIZONS,
    factorial_cells,
    paired_detection_summary,
    score_shift_summary,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


FROZEN_SAFE_CONFIG_SHA256 = (
    "2b447944d0218278c47918777dbf5777b5cf29a207a73231e89161abd9dcd4c6"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate the executed-action and SAFE-evidence effects of language faults."
        )
    )
    parser.add_argument("--score-dir", action="append", required=True, type=Path)
    parser.add_argument("--monitor-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def _score_index(
    np: Any,
    json_value: dict[str, Any],
    archive_path: Path,
    *,
    id_array: str,
    json_records: str,
    json_id: str,
) -> tuple[dict[str, Any], Any, Any]:
    if file_sha256(archive_path) != json_value["score_archive"]["sha256"]:
        raise ValueError(f"score archive hash differs from its record: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        identifiers = [str(value) for value in archive[id_array]]
        lengths = archive["lengths"].astype(int)
        scores = archive["scores"].copy()
        alphas = archive["alphas"].astype(float)
        bands = archive["bands"].astype(float)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"score archive repeats identifiers: {archive_path}")
    records = json_value[json_records]
    json_identifiers = [str(record[json_id]) for record in records]
    if identifiers != json_identifiers:
        raise ValueError(f"score archive and JSON ordering differ: {archive_path}")
    indexed = {
        identifier: scores[index, : lengths[index]]
        for index, identifier in enumerate(identifiers)
    }
    return indexed, alphas, bands


def _primary_band(
    np: Any, monitor: dict[str, Any], alphas: Any, bands: Any
) -> Any:
    primary = float(monitor["primary_alpha"])
    matches = np.flatnonzero(np.isclose(alphas, primary, rtol=0, atol=1e-8))
    if len(matches) != 1:
        raise ValueError("SAFE archive does not contain exactly one primary alpha")
    return bands[int(matches[0])]


def _load_worker(np: Any, directory: Path) -> dict[str, Any]:
    language_json_path = directory / "language-safe.json"
    language_npz_path = directory / "language-safe.npz"
    physical_json_path = directory / "physical-safe.json"
    physical_npz_path = directory / "physical-safe.npz"
    language = load_json(language_json_path)
    physical = load_json(physical_json_path)
    language_scores, language_alphas, language_bands = _score_index(
        np,
        language,
        language_npz_path,
        id_array="record_ids",
        json_records="records",
        json_id="record_id",
    )
    physical_scores, physical_alphas, physical_bands = _score_index(
        np,
        physical,
        physical_npz_path,
        id_array="runs",
        json_records="records",
        json_id="run",
    )
    if not np.array_equal(language_alphas, physical_alphas) or not np.array_equal(
        language_bands, physical_bands
    ):
        raise ValueError("language and physical scores use different SAFE bands")
    if language["ordinary_physical_scores"]["json_sha256"] != file_sha256(
        physical_json_path
    ):
        raise ValueError("language scores cite a different physical score JSON")
    if language["ordinary_physical_scores"]["archive_sha256"] != file_sha256(
        physical_npz_path
    ):
        raise ValueError("language scores cite a different physical score archive")
    return {
        "directory": directory,
        "language": language,
        "language_scores": language_scores,
        "physical": physical,
        "physical_scores": physical_scores,
        "alphas": language_alphas,
        "bands": language_bands,
        "artifacts": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in (
                language_json_path,
                language_npz_path,
                physical_json_path,
                physical_npz_path,
            )
        ],
    }


def _record(
    np: Any, source: dict[str, Any], record: dict[str, Any], band: Any
) -> dict[str, Any]:
    context = record["context"]
    step = int(context["policy_step"])
    base = {
        "record_id": str(record["record_id"]),
        "context_id": str(record["context_id"]),
        "analysis_split": str(context["analysis_split"]),
        "task_id": int(context["task_id"]),
        "episode_index": int(context["episode_index"]),
        "policy_step": step,
        "phase": str(context["phase"]),
        "action_token_position": int(context["action_token_position"]),
        "layer_index": int(record["layer_index"]),
        "command_id": record.get("command_id"),
        "command_group_size": int(record.get("command_group_size", 1)),
        "physical_run": record.get("physical_run"),
        "control_success": bool(record.get("control_success")),
        "terminal_success": record.get("terminal_success"),
        "audit_valid": False,
        "audit_reasons": [],
    }
    reasons = base["audit_reasons"]
    if record.get("status") != "scored":
        reasons.append(f"record_status_{record.get('status')}")
    if not record.get("composition_verified"):
        reasons.append("physical_composition_unverified")
    if record.get("monitor_horizon") != "complete_physical_trace":
        reasons.append("incomplete_monitor_horizon")
    if record.get("terminal_success") is None:
        reasons.append("terminal_outcome_unavailable")
    natural = source["language_scores"].get(str(record["record_id"]))
    physical_run = record.get("physical_run")
    control_run = f"{record['context_id']}-control"
    control = source["physical_scores"].get(control_run)
    if natural is None:
        reasons.append("natural_score_trace_missing")
    if control is None:
        reasons.append("control_score_trace_missing")
    if reasons:
        return base

    factorial = factorial_cells(np, natural, control, band, step)
    cells = factorial["cells"]
    natural_score = cells["faulted_action_faulted_evidence"][
        "score_at_intervention"
    ]
    control_score = cells["clean_action_clean_evidence"][
        "score_at_intervention"
    ]
    if natural_score != float(record["score_at_fault"]):
        reasons.append("natural_score_disagrees_with_JSON")
    if control_score != float(record["control_score_at_fault"]):
        reasons.append("control_score_disagrees_with_JSON")
    shared_before = cells["faulted_action_faulted_evidence"][
        "alarm_before_intervention"
    ]
    control_before = cells["clean_action_clean_evidence"][
        "alarm_before_intervention"
    ]
    if shared_before != control_before:
        reasons.append("pre_intervention_alarm_history_differs")
    if reasons:
        return base

    local = record["local_measurements"]
    command_changed = not bool(local["executed_command"]["exact_equal"])
    base.update(
        {
            "audit_valid": True,
            "command_changed": command_changed,
            "task_failure": not bool(record["terminal_success"]),
            "alarm_before_intervention": shared_before,
            "clean_score_at_intervention": control_score,
            "faulted_score_at_intervention": natural_score,
            "clean_evidence_contribution": factorial["evidence_contribution"][
                "clean"
            ],
            "faulted_evidence_contribution": factorial["evidence_contribution"][
                "faulted"
            ],
            "threshold_at_intervention": cells[
                "faulted_action_faulted_evidence"
            ]["threshold_at_intervention"],
            "faulted_minus_clean_score": factorial["evidence_contribution"][
                "faulted_minus_clean"
            ],
            "cells": cells,
        }
    )
    for name in ALARM_HORIZONS:
        base[f"shared_{name}"] = cells["faulted_action_faulted_evidence"][
            "alarms"
        ][name]["triggered"]
        base[f"restored_{name}"] = cells["faulted_action_clean_evidence"][
            "alarms"
        ][name]["triggered"]
        base[f"fault_evidence_clean_action_{name}"] = cells[
            "clean_action_faulted_evidence"
        ]["alarms"][name]["triggered"]
        base[f"control_{name}"] = cells["clean_action_clean_evidence"][
            "alarms"
        ][name]["triggered"]
    return base


def _cohort(rows: list[dict[str, Any]], *, failures: bool) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("audit_valid")
        and row.get("control_success")
        and row.get("command_changed")
        and not row.get("alarm_before_intervention")
    ]
    if failures:
        selected = [row for row in selected if row.get("task_failure")]
    return selected


def _summaries(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    action_changing = _cohort(rows, failures=False)
    failures = _cohort(rows, failures=True)
    result = {
        "action_changing_no_prior_alarm": {
            "score_shift": score_shift_summary(
                action_changing, samples=samples, seed=seed
            ),
            "detection": {
                window: paired_detection_summary(
                    action_changing,
                    window,
                    samples=samples,
                    seed=seed + 100 + index,
                )
                for index, window in enumerate(ALARM_HORIZONS)
            },
        },
        "task_failures_no_prior_alarm": {
            "score_shift": score_shift_summary(
                failures, samples=samples, seed=seed + 10
            ),
            "detection": {
                window: paired_detection_summary(
                    failures,
                    window,
                    samples=samples,
                    seed=seed + 200 + index,
                )
                for index, window in enumerate(ALARM_HORIZONS)
            },
        },
    }
    immediate = result["task_failures_no_prior_alarm"]["detection"][
        "at_intervention"
    ]
    result["primary_result"] = {
        "cohort": "task failures from successful controls with no prior alarm",
        "contrast": (
            "faulted action with clean evidence restored minus the same faulted "
            "action with naturally faulted evidence"
        ),
        "restoration_recovers_detection": immediate[
            "restoration_recovers_detection"
        ],
        "faulted_evidence_adds_detection": immediate[
            "faulted_evidence_adds_detection"
        ],
        "paired_detection_rate_difference": immediate[
            "paired_detection_rate_difference"
        ],
    }
    return result


def main() -> None:
    args = _arguments()
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    if len(set(args.score_dir)) != len(args.score_dir):
        raise ValueError("score directories must be unique")
    monitor_config_sha256 = file_sha256(args.monitor_config)
    if monitor_config_sha256 != FROZEN_SAFE_CONFIG_SHA256:
        raise ValueError("factorial requires the pinned cumulative SAFE-MLP config")

    import numpy as np

    workers = [_load_worker(np, path) for path in args.score_dir]
    monitor_hashes = {
        worker["language"]["monitor"]["checkpoint_sha256"] for worker in workers
    }
    if len(monitor_hashes) != 1:
        raise ValueError("workers used different frozen SAFE checkpoints")
    if any(
        worker["language"]["monitor"]["configuration_sha256"]
        != monitor_config_sha256
        for worker in workers
    ):
        raise ValueError("score workers cite a different SAFE configuration")
    first = workers[0]
    for worker in workers[1:]:
        if not np.array_equal(worker["alphas"], first["alphas"]) or not np.array_equal(
            worker["bands"], first["bands"]
        ):
            raise ValueError("workers used different SAFE calibration bands")
    band = _primary_band(
        np, first["language"]["monitor"], first["alphas"], first["bands"]
    )

    rows = []
    for worker in workers:
        rows.extend(
            _record(np, worker, record, band)
            for record in worker["language"]["records"]
        )
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("workers contain duplicate language interventions")

    summaries = _summaries(rows, samples=args.bootstrap_samples, seed=args.seed)
    split_summaries = {
        split: _summaries(
            [row for row in rows if row["analysis_split"] == split],
            samples=args.bootstrap_samples,
            seed=args.seed + 1_000 * (index + 1),
        )
        for index, split in enumerate(("development", "holdout"))
    }
    project_root = Path(__file__).resolve().parents[1]
    output = {
        "schema_version": 1,
        "analysis": "paired action/evidence factorial for frozen SAFE-MLP",
        "status": "exploratory_existing_outcomes_already_opened",
        "design": {
            "open_question": (
                "Does corruption of SAFE's shared internal evidence cause a harmful "
                "language-layer fault to be missed?"
            ),
            "factorial_cells": {
                "clean_action_clean_evidence": "recorded control trace",
                "clean_action_faulted_evidence": (
                    "control trace with the intervention-step score from the "
                    "faulted feature"
                ),
                "faulted_action_clean_evidence": (
                    "faulted physical trace with the intervention-step clean "
                    "score restored"
                ),
                "faulted_action_faulted_evidence": (
                    "recorded faulted physical trace with its layer-specific "
                    "faulted feature"
                ),
            },
            "primary_cohort": (
                "task failures caused by command-changing interventions from "
                "successful controls, excluding traces with an alarm before "
                "intervention"
            ),
            "primary_endpoint": "SAFE alarm at the intervention step",
            "primary_contrast": (
                "faulted action with clean evidence restored minus the same "
                "faulted action with naturally faulted evidence"
            ),
            "secondary_endpoints": list(ALARM_HORIZONS)[1:]
            + ["faulted-minus-clean SAFE score at intervention"],
            "estimand": (
                "uniform layer-context interventions; uncertainty resamples complete "
                "task/episode trajectories"
            ),
            "sensitivity": (
                "also weight each exact-command group equally because multiple "
                "layers can share one physical continuation"
            ),
            "interpretation_boundary": (
                "restoring the clean SAFE feature isolates evidence corruption; "
                "it does not construct an independent observer or prove that the "
                "clean feature contains enough information to diagnose a downstream "
                "action fault"
            ),
            "why_score_swapping_is_exact": (
                "SAFE commit b6036abe uses the frozen indep/MLP monitor, which "
                "projects each n_history_steps=1 feature independently and then "
                "cumulatively sums those contributions. Replacing one feature "
                "therefore shifts the complete cumulative-score suffix by that "
                "contribution difference. score_language_campaign.py already "
                "records the faulted physical continuation with each layer-specific "
                "intervention contribution."
            ),
        },
        "code": git_state(project_root),
        "monitor": first["language"]["monitor"],
        "inputs": [
            *[artifact for worker in workers for artifact in worker["artifacts"]],
            {
                "path": str(args.monitor_config.resolve()),
                "sha256": monitor_config_sha256,
            },
        ],
        "coverage": {
            "records": len(rows),
            "audit_valid": sum(row["audit_valid"] for row in rows),
            "audit_reasons": dict(
                sorted(
                    Counter(
                        reason for row in rows for reason in row["audit_reasons"]
                    ).items()
                )
            ),
        },
        "combined_discovery": summaries,
        "original_split_sensitivity": split_summaries,
        "records": rows,
    }
    write_json_atomic(args.output, output)
    columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    write_csv_atomic(
        args.records_csv,
        [{key: row.get(key, "") for key in columns} for row in rows],
    )
    print(
        json.dumps(
            {
                "coverage": output["coverage"],
                "primary_result": summaries["primary_result"],
                "split_primary_results": {
                    split: value["primary_result"]
                    for split, value in split_summaries.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
