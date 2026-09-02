from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.analyze_language_campaign import analysis_row
from embodied_silent_failures.artifacts import (
    artifact_record,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
)
from embodied_silent_failures.provenance import file_sha256, load_json


POLICY_ARRAYS = (
    "raw_action",
    "executed_command",
    "action_tokens",
    "sequence_token_ids",
    "action_token_logits",
    "global_top_token_ids",
    "global_top_token_logits",
    "action_log_normalizer",
    "action_entropy",
)


def _one_index(np: Any, condition: Any, label: str) -> int:
    indices = np.flatnonzero(condition)
    if len(indices) != 1:
        raise ValueError(f"expected one {label}, found {len(indices)}")
    return int(indices[0])


def boundary_indices(np: Any, archive: Any, policy_step: int) -> tuple[int, int]:
    steps = archive["policy_step"]
    stages = archive["snapshot_stage"]
    before = _one_index(
        np,
        (steps == policy_step) & (stages == 0),
        f"before-action snapshot at policy step {policy_step}",
    )
    after_candidates = np.flatnonzero(steps == policy_step + 1)
    if len(after_candidates) != 1:
        raise ValueError(
            "expected one snapshot after the intervened action at policy step "
            f"{policy_step + 1}, found {len(after_candidates)}"
        )
    return before, int(after_candidates[0])


def _trajectory_source(
    campaign_dir: Path, run: str
) -> tuple[Path, Path, dict[str, Any]]:
    attempt_dir = campaign_dir / "attempts" / run
    markers = sorted(attempt_dir.glob("*.complete.json"))
    if len(markers) != 1:
        raise ValueError(
            f"expected one completion marker for {run}, found {len(markers)}"
        )
    marker = load_json(markers[0])
    record = marker.get("trajectory_archive", {}).get("artifact")
    if not isinstance(record, dict):
        raise ValueError(f"completion marker has no trajectory artifact: {markers[0]}")
    path = attempt_dir / str(record["name"])
    if not path.is_file():
        raise FileNotFoundError(f"missing trajectory archive: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"trajectory byte count disagrees with its manifest: {path}")
    return path, markers[0], marker


def _pack_ragged(np: Any, values: list[Any]) -> tuple[Any, Any, Any]:
    if not values:
        raise ValueError("cannot pack an empty array collection")
    ranks = {value.ndim for value in values}
    if len(ranks) != 1:
        raise ValueError(f"ragged values have inconsistent ranks: {sorted(ranks)}")
    dtype = np.result_type(*(value.dtype for value in values))
    flattened = [np.asarray(value, dtype=dtype).reshape(-1) for value in values]
    offsets = np.zeros(len(flattened) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([value.size for value in flattened], dtype=np.int64)
    shapes = np.asarray([value.shape for value in values], dtype=np.int64)
    return np.concatenate(flattened), offsets, shapes


def _safe_archive(
    score_json_path: Path, score_document: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    record = score_document.get("score_archive")
    if not isinstance(record, dict):
        raise ValueError("physical SAFE scores have no archive record")
    path = score_json_path.parent / Path(str(record["path"])).name
    if not path.is_file():
        raise FileNotFoundError(f"missing physical SAFE score archive: {path}")
    actual = artifact_record(path)
    if actual["sha256"] != record["sha256"]:
        raise ValueError(f"physical SAFE score archive hash mismatch: {path}")
    return path, actual


def _alarm_fields(
    np: Any, scores: Any, threshold: Any, fault_step: int
) -> dict[str, Any]:
    alarms = np.asarray(scores >= threshold, dtype=np.bool_)
    score = float(scores[fault_step]) if fault_step < len(scores) else None
    band = float(threshold[fault_step]) if fault_step < len(threshold) else None
    return {
        "safe_score_at_fault": score,
        "safe_threshold_at_fault": band,
        "safe_alarm_before_fault": bool(alarms[:fault_step].any()),
        "safe_alarm_at_fault": (
            bool(alarms[fault_step]) if fault_step < len(alarms) else None
        ),
        "safe_alarm_post_fault_any": bool(alarms[fault_step:].any()),
    }


def extract_campaign(
    *,
    np: Any,
    campaign_dir: Path,
    language_scores_path: Path,
    physical_scores_path: Path,
    output_dir: Path,
    verify_source_hashes: bool,
) -> dict[str, Any]:
    language_scores = load_json(language_scores_path)
    physical_scores = load_json(physical_scores_path)
    primary_alpha = float(physical_scores["monitor"]["primary_alpha"])
    primary_alpha_key = format(primary_alpha, "g")
    safe_path, safe_artifact = _safe_archive(physical_scores_path, physical_scores)
    campaign_run = load_json(campaign_dir / "run.json")

    with np.load(safe_path, allow_pickle=False) as safe:
        safe_runs = [str(value) for value in safe["runs"]]
        if len(safe_runs) != len(set(safe_runs)):
            raise ValueError("physical SAFE archive repeats a run")
        score_indices = {run: index for index, run in enumerate(safe_runs)}
        alphas = safe["alphas"].copy()
        alpha_index = _one_index(
            np,
            np.isclose(alphas, primary_alpha),
            f"SAFE alpha {primary_alpha_key}",
        )
        bands = safe["bands"].copy()
        all_scores = safe["scores"].copy()
        safe_lengths = safe["lengths"].copy()

    physical_records = physical_scores["records"]
    runs = [str(record["run"]) for record in physical_records]
    if len(runs) != len(set(runs)):
        raise ValueError("physical SAFE records repeat a run")
    missing_scores = sorted(set(runs) - set(score_indices))
    if missing_scores:
        raise ValueError(f"physical SAFE archive is missing runs: {missing_scores}")

    policy_values: dict[str, list[Any]] = {name: [] for name in POLICY_ARRAYS}
    state_pre: list[Any] = []
    state_post: list[Any] = []
    state_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    branch_sources: list[dict[str, Any]] = []
    omitted_images: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    selected_score_indices: list[int] = []

    for record in physical_records:
        run = str(record["run"])
        try:
            trajectory_path, marker_path, marker = _trajectory_source(campaign_dir, run)
            trajectory_record = marker["trajectory_archive"]["artifact"]
            actual_hash = file_sha256(trajectory_path) if verify_source_hashes else None
            if verify_source_hashes and actual_hash != trajectory_record["sha256"]:
                raise ValueError(f"trajectory archive hash mismatch: {trajectory_path}")
            fault = record["fault"]
            fault_step = int(fault["policy_step"])
            score_index = score_indices[run]
            score_length = int(safe_lengths[score_index])
            scores = all_scores[score_index, :score_length]
            threshold = bands[alpha_index, :score_length]

            current_images = []
            with np.load(trajectory_path, allow_pickle=False) as archive:
                before, after = boundary_indices(np, archive, fault_step)
                decision = _one_index(
                    np,
                    archive["decision_policy_step"] == fault_step,
                    f"policy decision at step {fault_step}",
                )
                current_policy_values = {
                    name: archive[name][decision].copy() for name in POLICY_ARRAYS
                }
                current_state: list[tuple[dict[str, Any], Any, Any]] = [
                    (
                        {
                            "name": "simulator_state",
                            "source_archive_key": "simulator_state",
                            "source_dtype": archive["simulator_state"].dtype.str,
                            "source_series_sha256": marker["trajectory_archive"][
                                "simulator_state"
                            ]["sha256"],
                        },
                        archive["simulator_state"][before].copy(),
                        archive["simulator_state"][after].copy(),
                    )
                ]
                for observation in marker["trajectory_archive"]["observations"]:
                    source = {
                        "name": str(observation["name"]),
                        "source_archive_key": str(observation["archive_key"]),
                        "source_dtype": str(observation["dtype"]),
                        "source_series_sha256": str(observation["sha256"]),
                    }
                    if observation["kind"] == "image":
                        current_images.append(
                            {
                                "run": run,
                                **source,
                                "source_shape": observation["shape"],
                            }
                        )
                        continue
                    values = archive[source["source_archive_key"]]
                    if values.dtype.kind == "c":
                        raise ValueError(
                            f"numeric observation is complex and cannot use the "
                            f"real-valued product-state table: {source['name']}"
                        )
                    current_state.append(
                        (source, values[before].copy(), values[after].copy())
                    )

            alarms = _alarm_fields(np, scores, threshold, fault_step)
            failure = not bool(record["success"])
            fault_condition = record["condition"] == "activation_fault"
            branch_index = len(branch_rows)
            current_branch = {
                "branch_index": branch_index,
                "run": run,
                "condition": record["condition"],
                "context_id": run.split("-", 1)[0],
                "task_id": int(record["task_id"]),
                "episode_index": int(record["episode_index"]),
                "policy_step": fault_step,
                "source_policy_step": int(fault["source_policy_step"]),
                "action_token_position": int(fault["action_token_position"]),
                "fault_kind": fault["kind"],
                "layer_index": fault.get("layer_index"),
                "command_id": fault.get("command_group", {}).get("command_id"),
                "success": bool(record["success"]),
                "task_failure": failure,
                "rollout_policy_steps": int(record["length"]),
                **alarms,
                "operational_silent_failure": (
                    fault_condition
                    and failure
                    and not alarms["safe_alarm_before_fault"]
                    and not alarms["safe_alarm_post_fault_any"]
                ),
            }
            current_source = {
                "run": run,
                "completion_marker": str(marker_path.relative_to(campaign_dir)),
                "completion_marker_sha256": file_sha256(marker_path),
                "trajectory": str(trajectory_path.relative_to(campaign_dir)),
                "trajectory_bytes": int(trajectory_record["bytes"]),
                "trajectory_sha256": trajectory_record["sha256"],
                "trajectory_hash_verified": verify_source_hashes,
                "trajectory_actual_sha256": actual_hash,
            }

            # Commit a branch only after every source has been read successfully.
            for name, value in current_policy_values.items():
                policy_values[name].append(value)
            for source, pre_value, post_value in current_state:
                state_pre.append(np.asarray(pre_value, dtype=np.float64))
                state_post.append(np.asarray(post_value, dtype=np.float64))
                state_rows.append(
                    {
                        "state_entry_index": len(state_rows),
                        "branch_index": branch_index,
                        "run": run,
                        **source,
                        "value_shape": json.dumps(list(pre_value.shape)),
                    }
                )
            branch_rows.append(current_branch)
            branch_sources.append(current_source)
            omitted_images.extend(current_images)
            selected_score_indices.append(score_index)
        except Exception as error:
            errors.append(
                {
                    "run": run,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if not branch_rows:
        raise ValueError("no physical branches could be extracted")

    arrays: dict[str, Any] = {
        "safe_scores": all_scores[selected_score_indices],
        "safe_lengths": safe_lengths[selected_score_indices],
        "safe_alphas": alphas,
        "safe_bands": bands,
    }
    for name, values in policy_values.items():
        packed, offsets, shapes = _pack_ragged(np, values)
        arrays[f"{name}_values"] = packed
        arrays[f"{name}_offsets"] = offsets
        arrays[f"{name}_shapes"] = shapes
    pre_values, state_offsets, state_shapes = _pack_ragged(np, state_pre)
    post_values, post_offsets, post_shapes = _pack_ragged(np, state_post)
    if not np.array_equal(state_offsets, post_offsets) or not np.array_equal(
        state_shapes, post_shapes
    ):
        raise ValueError("before and after product-state layouts disagree")
    arrays["numeric_state_before_values"] = pre_values
    arrays["numeric_state_after_values"] = post_values
    arrays["numeric_state_offsets"] = state_offsets
    arrays["numeric_state_shapes"] = state_shapes

    branch_index = {row["run"]: int(row["branch_index"]) for row in branch_rows}
    intervention_rows = []
    for record in language_scores["records"]:
        row = analysis_row(record, primary_alpha_key)
        row["branch_index"] = branch_index.get(str(row.get("physical_run")), "")
        row["product_state_available"] = row["branch_index"] != ""
        intervention_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    branches_path = output_dir / "branches.csv"
    interventions_path = output_dir / "interventions.csv"
    states_path = output_dir / "state-entries.csv"
    arrays_path = output_dir / "product-state.npz"
    write_csv_atomic(branches_path, branch_rows)
    write_csv_atomic(interventions_path, intervention_rows)
    write_csv_atomic(states_path, state_rows)
    write_npz_atomic(arrays_path, np, arrays)

    artifacts = [
        artifact_record(path)
        for path in (branches_path, interventions_path, states_path, arrays_path)
    ]
    result = {
        "schema_version": 1,
        "status": "complete_with_errors" if errors else "complete",
        "analysis": "OpenVLA policy-environment-monitor product-state extraction",
        "source_campaign": {
            "directory": str(campaign_dir.resolve()),
            "run_sha256": file_sha256(campaign_dir / "run.json"),
            "campaign_revision": campaign_run["execution"]["experiment_code"][
                "revision"
            ],
        },
        "source_scores": {
            "language": {
                "path": str(language_scores_path.resolve()),
                "sha256": file_sha256(language_scores_path),
            },
            "physical": {
                "path": str(physical_scores_path.resolve()),
                "sha256": file_sha256(physical_scores_path),
                "archive": safe_artifact,
            },
        },
        "boundary_semantics": {
            "provenance": (
                "Experiment commit "
                f"{campaign_run['execution']['experiment_code']['revision']}, "
                "language_context.py::run_terminal_branch: "
                "the before state is recorded immediately before the intervened "
                "env.step; the after state is the next recorded simulator snapshot."
            ),
            "policy_evidence": "the OpenVLA decision executed at the intervention step",
            "safe_evidence": (
                "the complete frozen SAFE-MLP score trace and its original "
                "time-varying threshold bands"
            ),
        },
        "coverage": {
            "physical_records": len(physical_records),
            "extracted_branches": len(branch_rows),
            "failed_branches": len(errors),
            "intervention_records": len(intervention_rows),
            "interventions_with_product_state": sum(
                bool(row["product_state_available"]) for row in intervention_rows
            ),
            "numeric_state_entries": len(state_rows),
            "omitted_image_series": len(omitted_images),
        },
        "artifacts": artifacts,
        "source_trajectory_archives": branch_sources,
        "omitted_image_series": omitted_images,
        "errors": errors,
    }
    write_json_atomic(output_dir / "manifest.json", result)
    return result
