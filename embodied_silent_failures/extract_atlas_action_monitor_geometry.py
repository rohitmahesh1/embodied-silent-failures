from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from embodied_silent_failures.action_monitor_geometry import (
    ACTION_VOCABULARY_SIZE,
    SAFE_TOKEN_INDEX,
    action_monitor_arrays,
    summarize_action_monitor_arrays,
)
from embodied_silent_failures.artifacts import (
    artifact_record,
    write_json_atomic,
    write_npz_atomic,
)
from embodied_silent_failures.language_policy import action_vocabulary_bounds
from embodied_silent_failures.openvla_runtime import (
    CHECKPOINT_REVISION,
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_state,
    load_json,
    source_file_record,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover the correct OpenVLA action vocabulary and compare its response "
            "with SAFE on paired atlas trajectories."
        )
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--geometry-arrays", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--array-output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--physical-run", action="append")
    return parser.parse_args()


def _one_completion(attempt_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    completions = list(attempt_dir.glob("*.complete.json"))
    if len(completions) != 1:
        raise ValueError(
            f"expected one completion in {attempt_dir}, found {len(completions)}"
        )
    result = load_json(completions[0])
    trajectory = result.get("files", {}).get("trajectory")
    feature_pickle = result.get("files", {}).get("pickle")
    if not trajectory or not feature_pickle:
        raise ValueError(f"completion omits trajectory or feature archive: {completions[0]}")
    return result, attempt_dir / str(trajectory), attempt_dir / str(feature_pickle)


def _trajectory_window(np: Any, path: Path, steps: list[int]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        positions = {
            int(step): index
            for index, step in enumerate(archive["decision_policy_step"].tolist())
        }
        missing = [step for step in steps if step not in positions]
        if missing:
            raise ValueError(f"trajectory omits requested decision steps: {missing}")
        rows = np.asarray([positions[step] for step in steps], dtype=np.int64)
        return {
            name: archive[name][rows].copy()
            for name in (
                "raw_action",
                "executed_command",
                "action_tokens",
                "action_token_logits",
                "action_log_normalizer",
            )
        }


def _hidden_window(path: Path, steps: list[int], torch: Any) -> Any:
    with path.open("rb") as file:
        values = pickle.load(file)["hidden_states"]
    if isinstance(values, list):
        values = torch.stack(values, dim=0)
    if values.ndim != 3 or tuple(values.shape[1:]) != (7, 4096):
        raise ValueError(f"unexpected OpenVLA hidden-state shape: {tuple(values.shape)}")
    if min(steps) < 0 or max(steps) >= len(values):
        raise ValueError("feature archive does not cover the requested absolute steps")
    return values[steps].contiguous()


def _fault_provenance(completion: dict[str, Any]) -> dict[str, Any]:
    fault = completion.get("fault", {})
    local = fault.get("representative_local_measurements", {})
    identity = local.get("identity")
    topologies = [str(value) for value in local.get("topologies", [])]
    stale_logit_source_step = None
    if (
        isinstance(identity, dict)
        and identity.get("kind") == "module_output"
        and identity.get("module_path") == "policy"
        and identity.get("output_port") == "value.logits"
    ):
        stale_logit_source_step = int(fault["source_policy_step"])
    return {
        "identity": identity,
        "topologies": topologies,
        "same_feature_comparable": topologies
        == ["shared_action_and_monitor_evidence"],
        "stale_logit_source_step": stale_logit_source_step,
    }


def _project_rows(model: Any, hidden: Any, start: int, stop: int, torch: Any) -> Any:
    if start == stop:
        return torch.empty((*hidden.shape[:-1], 0), dtype=torch.float32)
    head = model.language_model.lm_head
    weight = head.weight[start:stop]
    bias = head.bias[start:stop] if head.bias is not None else None
    with torch.inference_mode():
        projected = torch.nn.functional.linear(
            hidden.to(device=weight.device, dtype=weight.dtype), weight, bias
        )
    return projected.detach().to(torch.float32).cpu()


def _project_safe_token_like_generation(
    model: Any, hidden: Any, start: int, stop: int, torch: Any
) -> Any:
    """Apply the complete head one row at a time, matching generation calls 1-6."""
    head = model.language_model.lm_head
    rows = []
    with torch.inference_mode():
        for step in range(hidden.shape[0]):
            value = hidden[step, SAFE_TOKEN_INDEX].to(
                device=head.weight.device, dtype=head.weight.dtype
            )
            rows.append(head(value[None, None, :])[0, 0, start:stop].cpu())
    return torch.stack(rows).to(torch.float32)


def recover_action_logits(
    np: Any,
    torch: Any,
    model: Any,
    hidden: Any,
    archived: Any,
    trajectory_metadata: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    archived = np.asarray(archived, dtype=np.float32)
    if archived.shape != (*hidden.shape[:2], ACTION_VOCABULARY_SIZE):
        raise ValueError("archived action-logit slice has unexpected shape")
    model_output_size = int(model.language_model.lm_head.weight.shape[0])
    correct_start, correct_stop = action_vocabulary_bounds(model, model_output_size)
    recorded_start = trajectory_metadata.get("action_token_start")
    if recorded_start is not None:
        if int(recorded_start) != correct_start:
            raise ValueError("trajectory declares a different action-vocabulary boundary")
        return archived, {
            "method": "declared_correct_action_vocabulary",
            "reconstructed_entries": 0,
            "preserved_archived_entries": ACTION_VOCABULARY_SIZE,
            "same_feature_overlap_maximum_absolute_error": None,
            "secondary_token_overlap_maximum_absolute_error": None,
        }

    # Campaign revision 61ef417 used language_policy.py::generation_logit_trace,
    # which archived logits[:, -256:]. OpenVLA 300dce26 decodes with
    # model.vocab_size after removing the 64 padded outputs. Recover only the
    # action entries that the old slice omitted and preserve its exact overlap.
    legacy_start = model_output_size - ACTION_VOCABULARY_SIZE
    if legacy_start < correct_start or legacy_start > correct_stop:
        raise ValueError("legacy and decoded action vocabularies do not overlap as expected")
    missing = legacy_start - correct_start
    overlap = ACTION_VOCABULARY_SIZE - missing
    recovered = np.empty_like(archived)
    audit_count = min(16, overlap)
    projected_stop = legacy_start + audit_count
    batched = _project_rows(
        model, hidden, correct_start, projected_stop, torch
    ).numpy()
    selected = _project_safe_token_like_generation(
        model, hidden, correct_start, projected_stop, torch
    ).numpy()
    if missing:
        recovered[..., :missing] = batched[..., :missing]
        recovered[:, SAFE_TOKEN_INDEX, :missing] = selected[..., :missing]
    recovered[..., missing:] = archived[..., :overlap]

    selected_overlap_error = None
    secondary_overlap_error = None
    if audit_count:
        expected = archived[..., :audit_count]
        selected_overlap_error = float(
            np.max(
                np.abs(
                    selected[..., missing:]
                    - expected[:, SAFE_TOKEN_INDEX, :]
                )
            )
        )
        secondary = np.delete(batched[..., missing:], SAFE_TOKEN_INDEX, axis=1)
        secondary_expected = np.delete(expected, SAFE_TOKEN_INDEX, axis=1)
        secondary_overlap_error = float(
            np.max(np.abs(secondary - secondary_expected))
        )
    return recovered, {
        "method": "recover_legacy_padded_vocabulary_omission",
        "reconstructed_entries": missing,
        "preserved_archived_entries": overlap,
        "overlap_audit_entries": audit_count,
        "same_feature_overlap_maximum_absolute_error": selected_overlap_error,
        "secondary_token_overlap_maximum_absolute_error": secondary_overlap_error,
    }


def _bfloat16_words(torch: Any, values: Any) -> Any:
    tensor = torch.as_tensor(values).to(torch.bfloat16).contiguous()
    return tensor.view(torch.int16).numpy().view("uint16")


def _branch_data(
    *,
    np: Any,
    torch: Any,
    model: Any,
    attempt_dir: Path,
    steps: list[int],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    completion, trajectory_path, feature_path = _one_completion(attempt_dir)
    trajectory = _trajectory_window(np, trajectory_path, steps)
    fault_provenance = _fault_provenance(completion)
    hidden_steps = list(steps)
    if fault_provenance["stale_logit_source_step"] is not None:
        # The graph records policy.value.logits as an action-only boundary. At
        # this site x_t <- x_(t-1) changes generated logits after the current
        # hidden feature is computed, so recover its missing logits from the
        # mechanically recorded source step rather than the unchanged x_t feature.
        hidden_steps[0] = int(fault_provenance["stale_logit_source_step"])
    hidden = _hidden_window(feature_path, hidden_steps, torch)
    logits, recovery = recover_action_logits(
        np,
        torch,
        model,
        hidden,
        trajectory["action_token_logits"],
        completion.get("trajectory_archive", {}).get("policy_evidence", {}),
    )
    return trajectory, logits, {**recovery, **fault_provenance}


def main() -> None:
    args = _arguments()
    config_file = args.libero_config / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"LIBERO configuration is missing: {config_file}")
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    project_root = Path(__file__).resolve().parents[1]
    validate_pinned_runtime(
        args.checkpoint,
        args.openvla_root,
        args.libero_root,
        project_root=project_root,
    )
    geometry = load_json(args.geometry)
    if file_sha256(args.geometry_arrays) != geometry["array_archive"]["sha256"]:
        raise ValueError("SAFE geometry array archive differs from its manifest")
    records = list(geometry["records"])
    if args.physical_run:
        requested = set(args.physical_run)
        records = [
            record for record in records if str(record["physical_run"]) in requested
        ]
        missing = requested - {str(record["physical_run"]) for record in records}
        if missing:
            raise ValueError(f"requested physical runs are absent: {sorted(missing)}")
    if args.limit is not None:
        records = records[: args.limit]

    runtime = load_runtime(args.openvla_root, args.libero_root)
    runtime.set_seed_everywhere(20260905)
    model = runtime.get_model(model_config(args.checkpoint, "libero_10")).eval()
    model.requires_grad_(False)
    np, torch = runtime.np, runtime.torch
    model_output_size = int(model.language_model.lm_head.weight.shape[0])
    action_start, action_stop = action_vocabulary_bounds(model, model_output_size)

    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_context[str(record["context_id"])].append(record)

    control_arrays: dict[str, list[Any]] = defaultdict(list)
    branch_arrays: dict[str, list[Any]] = defaultdict(list)
    output_records = []
    errors = []
    same_feature_integrity_errors = []
    secondary_integrity_errors = []
    recovery_methods = Counter()
    control_context_ids = []
    for context_number, (context_id, members) in enumerate(sorted(by_context.items()), 1):
        fault_step = int(members[0]["fault_step"])
        window_steps = int(members[0]["window_steps"])
        if any(
            int(value["fault_step"]) != fault_step
            or int(value["window_steps"]) != window_steps
            for value in members
        ):
            raise ValueError(f"context {context_id} has inconsistent windows")
        steps = list(range(fault_step, fault_step + window_steps))
        try:
            clean, clean_logits, clean_recovery = _branch_data(
                np=np,
                torch=torch,
                model=model,
                attempt_dir=args.campaign_dir / "attempts" / f"{context_id}-control",
                steps=steps,
            )
            control_index = len(control_arrays["correct_action_logits_bfloat16_words"])
            control_context_ids.append(context_id)
            control_arrays["correct_action_logits_bfloat16_words"].append(
                _bfloat16_words(torch, clean_logits)
            )
            for name in (
                "raw_action",
                "executed_command",
                "action_tokens",
                "action_log_normalizer",
            ):
                control_arrays[name].append(clean[name])
            recovery_methods[clean_recovery["method"]] += 1
            if clean_recovery["same_feature_overlap_maximum_absolute_error"] is not None:
                same_feature_integrity_errors.append(
                    float(clean_recovery["same_feature_overlap_maximum_absolute_error"])
                )
                secondary_integrity_errors.append(
                    float(clean_recovery["secondary_token_overlap_maximum_absolute_error"])
                )
        except Exception as error:
            for record in members:
                errors.append(
                    {
                        "physical_run": record["physical_run"],
                        "context_id": context_id,
                        "status": "error",
                        "stage": "control",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            continue

        for record in members:
            try:
                faulted, faulted_logits, recovery = _branch_data(
                    np=np,
                    torch=torch,
                    model=model,
                    attempt_dir=args.campaign_dir
                    / "attempts"
                    / str(record["physical_run"]),
                    steps=steps,
                )
                comparison = action_monitor_arrays(
                    np,
                    clean_logits=clean_logits,
                    faulted_logits=faulted_logits,
                    clean_tokens=clean["action_tokens"],
                    faulted_tokens=faulted["action_tokens"],
                    clean_raw_action=clean["raw_action"],
                    faulted_raw_action=faulted["raw_action"],
                    clean_command=clean["executed_command"],
                    faulted_command=faulted["executed_command"],
                    clean_full_log_normalizer=clean["action_log_normalizer"],
                    faulted_full_log_normalizer=faulted["action_log_normalizer"],
                )
                array_index = len(output_records)
                branch_arrays["correct_action_logits_bfloat16_words"].append(
                    _bfloat16_words(torch, faulted_logits)
                )
                for name in (
                    "raw_action",
                    "executed_command",
                    "action_tokens",
                    "action_log_normalizer",
                ):
                    branch_arrays[name].append(faulted[name])
                for name, values in comparison.items():
                    branch_arrays[name].append(values)
                output_records.append(
                    {
                        **record,
                        "status": "complete",
                        "array_index": array_index,
                        "control_array_index": control_index,
                        "action_logit_recovery": recovery,
                        "representative_topologies": recovery["topologies"],
                        "representative_identity": recovery["identity"],
                        "same_feature_comparable": recovery[
                            "same_feature_comparable"
                        ],
                        **summarize_action_monitor_arrays(comparison),
                    }
                )
                recovery_methods[recovery["method"]] += 1
                if recovery["same_feature_overlap_maximum_absolute_error"] is not None:
                    same_feature_integrity_errors.append(
                        float(recovery["same_feature_overlap_maximum_absolute_error"])
                    )
                    secondary_integrity_errors.append(
                        float(recovery["secondary_token_overlap_maximum_absolute_error"])
                    )
            except Exception as error:
                errors.append(
                    {
                        "physical_run": record["physical_run"],
                        "context_id": context_id,
                        "status": "error",
                        "stage": "faulted",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        if context_number % 10 == 0 or context_number == len(by_context):
            print(
                f"contexts {context_number}/{len(by_context)}; "
                f"branches {len(output_records)}; errors {len(errors)}",
                flush=True,
            )

    if not output_records:
        raise RuntimeError("no action-monitor trajectory geometry was extracted")
    archive = {
        "physical_runs": np.asarray(
            [record["physical_run"] for record in output_records]
        ),
        "control_context_ids": np.asarray(control_context_ids),
        **{
            f"control_{name}": np.stack(values)
            for name, values in sorted(control_arrays.items())
        },
        **{
            f"faulted_{name}": np.stack(values)
            for name, values in sorted(branch_arrays.items())
        },
    }
    write_npz_atomic(args.array_output, np, archive)
    output = {
        "schema_version": 1,
        "analysis": "OpenVLA action evidence and SAFE response at matched steps",
        "analysis_contract": {
            "unit": "one distinct non-control physical continuation",
            "same_feature_comparison": (
                "the seventh generated action token, whose final 4096-dimensional "
                "feature is consumed both by OpenVLA's action head and SAFE-MLP; "
                "primary comparisons retain only graph sites mechanically marked "
                "shared_action_and_monitor_evidence"
            ),
            "policy_measurement": (
                "Jensen-Shannon divergence conditional on OpenVLA's 256 decoded "
                "action tokens; all seven token positions and executed commands are retained"
            ),
            "temporal_separation": (
                "fault-step values describe the immediate policy decision; later values "
                "describe the resulting closed-loop trajectory and are kept separate"
            ),
            "legacy_recovery": (
                "the original campaign omitted the first 64 decoded action logits because "
                "the model head has 64 padded outputs; only those missing entries are "
                "recomputed from saved final features, while the 192-entry overlap is preserved; "
                "the SAFE-aligned seventh token matches the original one-row head call, while "
                "missing entries for secondary tokens use a recorded-error batched reconstruction"
            ),
            "error_policy": (
                "branch errors are retained as records and do not terminate extraction"
            ),
        },
        "provenance": {
            "analysis_code": git_state(project_root),
            "entrypoint_sha256": file_sha256(Path(__file__)),
            "methods_sha256": file_sha256(
                Path(__file__).with_name("action_monitor_geometry.py")
            ),
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint": str(args.checkpoint.resolve()),
            "openvla_model_source": source_file_record(model),
            "geometry": {
                "path": str(args.geometry.resolve()),
                "sha256": file_sha256(args.geometry),
            },
            "geometry_arrays": {
                "path": str(args.geometry_arrays.resolve()),
                "sha256": file_sha256(args.geometry_arrays),
            },
        },
        "model_vocabulary": {
            "decoder_vocabulary_size": int(model.vocab_size),
            "model_output_size": model_output_size,
            "action_token_start": action_start,
            "action_token_stop_exclusive": action_stop,
        },
        "coverage": {
            "declared_physical_continuations": len(records),
            "complete": len(output_records),
            "errors": len(errors),
            "recovery_methods": dict(sorted(recovery_methods.items())),
        },
        "maximum_same_feature_recomputed_overlap_error": (
            max(same_feature_integrity_errors)
            if same_feature_integrity_errors
            else None
        ),
        "maximum_secondary_token_recomputed_overlap_error": (
            max(secondary_integrity_errors) if secondary_integrity_errors else None
        ),
        "array_archive": artifact_record(args.array_output),
        "records": output_records,
        "error_records": errors,
    }
    write_json_atomic(args.output, output)
    print(
        json.dumps(
            {
                "coverage": output["coverage"],
                "model_vocabulary": output["model_vocabulary"],
                "maximum_same_feature_recomputed_overlap_error": output[
                    "maximum_same_feature_recomputed_overlap_error"
                ],
                "maximum_secondary_token_recomputed_overlap_error": output[
                    "maximum_secondary_token_recomputed_overlap_error"
                ],
                "array_archive": output["array_archive"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
