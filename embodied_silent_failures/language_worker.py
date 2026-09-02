from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    write_json_atomic,
    write_pickle_atomic,
)
from embodied_silent_failures.language_context import (
    capture_context,
    run_terminal_branch,
    write_captured_context_archive,
    write_terminal_branch,
)
from embodied_silent_failures.language_fault import LanguageBlockInjector
from embodied_silent_failures.language_interface import (
    boundary_replay_record,
    boundary_replay_targets,
    cache_replay_inputs,
    trace_repeatability,
)
from embodied_silent_failures.language_interface_archive import InterfaceArchiveBuilder
from embodied_silent_failures.language_policy import (
    intervention_record,
    policy_decision,
)
from embodied_silent_failures.openvla_runtime import array_sha256
from embodied_silent_failures.provenance import load_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def error_record(reason: str, error: Exception, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "error",
        "reason": reason,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(limit=16),
        "updated_at": now(),
        **extra,
    }


def _command_groups(runtime: Any, candidates: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    # SAFE OpenVLA 300dce26, modeling_prismatic.py::predict_action, returns the
    # action consumed once by language_context.py::run_terminal_branch. Later
    # decisions are unmodified, so exact executed-command identity defines the
    # shared physical suffix; each member's distinct SAFE feature is kept.
    groups: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        layer_index, decision, local_record = candidate
        command_id = array_sha256(runtime, decision.command)
        group = groups.setdefault(
            command_id,
            {
                "command_id": command_id,
                "executed_command": decision.command.tolist(),
                "members": [],
            },
        )
        group["members"].append((layer_index, decision, local_record))
    return list(groups.values())


def _select_terminal_groups(
    groups: list[dict[str, Any]],
    control_result: dict[str, Any],
    maximum_branches: int | None,
) -> tuple[list[dict[str, Any]], str | None]:
    if control_result.get("status") != "complete":
        return [], "control_unresolved"
    if control_result.get("success") is not True:
        return [], "control_failed"
    if maximum_branches is None:
        return groups, None
    return groups[:maximum_branches], None


def _group_record(group: dict[str, Any]) -> dict[str, Any]:
    members = group["members"]
    return {
        "command_id": group["command_id"],
        "executed_command": group["executed_command"],
        "representative_layer_index": int(members[0][0]),
        "member_layer_indices": [int(member[0]) for member in members],
        "member_site_ids": [str(member[2]["site_id"]) for member in members],
    }


def _run_branch(
    *,
    output_dir: Path,
    runtime: Any,
    policy_config: Any,
    model: Any,
    processor: Any,
    env: Any,
    task_description: str,
    initial_state: Any,
    context: dict[str, Any],
    captured: Any,
    decision: Any,
    fault: dict[str, Any],
    execution: dict[str, Any],
    condition: str,
    wait_steps: int,
    restore_directly: bool,
) -> dict[str, Any]:
    completion = output_dir / (
        f"task{context['task_id']}--ep{context['episode_index']}.complete.json"
    )
    if completion.is_file():
        return load_json(completion)
    errors = []
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            branch = run_terminal_branch(
                runtime,
                policy_config,
                model,
                processor,
                env,
                task_description,
                initial_state,
                context,
                captured,
                decision,
                wait_steps=wait_steps,
                restore_directly=restore_directly,
            )
            result = write_terminal_branch(
                output_dir,
                runtime,
                context,
                task_description,
                branch,
                fault,
                execution,
                condition=condition,
                elapsed_seconds=time.perf_counter() - started,
            )
            (output_dir / "branch.unresolved.json").unlink(missing_ok=True)
            return result
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            write_json_atomic(
                output_dir / f"attempt-{attempt}.error.json",
                error_record("terminal_branch_exception", error, attempt=attempt),
            )
    write_json_atomic(
        output_dir / "branch.unresolved.json",
        {
            "schema_version": 1,
            "status": "unresolved",
            "reason": "terminal branch failed twice",
            "errors": errors,
            "updated_at": now(),
        },
    )
    return {"status": "unresolved", "errors": errors}


def run_context(
    *,
    output_dir: Path,
    wait_steps: int,
    maximum_faulted_terminal_branches: int | None,
    branch_state_restoration: str,
    instrumentation: dict[str, Any] | None,
    context: dict[str, Any],
    sites: dict[tuple[int, int], dict[str, Any]],
    runtime: Any,
    policy_config: Any,
    model: Any,
    processor: Any,
    injector: LanguageBlockInjector,
    env: Any,
    task_description: str,
    initial_state: Any,
    execution: dict[str, Any],
) -> dict[str, Any]:
    if branch_state_restoration not in {"prefix-replay", "direct"}:
        raise ValueError(
            f"unknown branch state restoration: {branch_state_restoration}"
        )
    restore_directly = branch_state_restoration == "direct"
    instrumentation = instrumentation or {}
    interface_enabled = bool(instrumentation.get("full_language_interfaces", False))
    context_interfaces = bool(
        instrumentation.get("context_conditioned_interfaces", False)
    )
    terminal_branches = bool(instrumentation.get("terminal_branches", True))
    replay_kinds = list(instrumentation.get("boundary_replays", []))
    context_dir = output_dir / "contexts" / str(context["context_id"])
    context_dir.mkdir(parents=True, exist_ok=True)
    complete_path = context_dir / "context.complete.json"
    if complete_path.is_file():
        return load_json(complete_path)

    runtime.set_seed_everywhere(int(context["trial_seed"]))
    captured = capture_context(
        runtime,
        policy_config,
        model,
        processor,
        injector,
        env,
        task_description,
        initial_state,
        context,
        wait_steps=wait_steps,
        capture_internal_state=context_interfaces,
        capture_context_state=context_interfaces,
    )
    captured_context_path = context_dir / "captured_context.npz"
    captured_context_archive = write_captured_context_archive(
        captured_context_path,
        runtime,
        captured,
    )
    token_position = int(context["action_token_position"])
    clean = policy_decision(
        runtime,
        policy_config,
        model,
        processor,
        captured.observation,
        task_description,
        injector=injector,
        action_token_position=token_position,
        capture_internal_state=context_interfaces,
        capture_context_state=context_interfaces,
    )
    local_records = []
    candidates = []
    faulted_features = {}
    if interface_enabled and captured.source_decision is None:
        raise RuntimeError("instrumented context has no captured source decision")
    interface_archive = (
        InterfaceArchiveBuilder(runtime, captured.source_decision, clean)
        if interface_enabled
        else None
    )
    boundary_replays = []
    for layer_index in range(32):
        site = sites[(layer_index, token_position)]
        try:
            faulted = policy_decision(
                runtime,
                policy_config,
                model,
                processor,
                captured.observation,
                task_description,
                injector=injector,
                action_token_position=token_position,
                replacement_layer=layer_index,
                sources=captured.source_trace.block_values,
                capture_internal_state=context_interfaces,
            )
            record = intervention_record(
                runtime,
                site=site,
                source=captured.source_trace,
                clean=clean,
                faulted=faulted,
            )
            local_records.append(record)
            faulted_features[layer_index] = faulted.hidden_states
            if interface_archive is not None:
                interface_archive.add_fault(layer_index, faulted)
                for boundary_kind, boundary_layer in boundary_replay_targets(
                    layer_index, replay_kinds
                ):
                    try:
                        cache_layers, cache_sources = cache_replay_inputs(
                            faulted.trace,
                            injection_layer=layer_index,
                            boundary_layer=boundary_layer,
                        )
                        replayed = policy_decision(
                            runtime,
                            policy_config,
                            model,
                            processor,
                            captured.observation,
                            task_description,
                            injector=injector,
                            action_token_position=token_position,
                            replacement_layer=boundary_layer,
                            sources={
                                boundary_layer: faulted.trace.block_values[
                                    boundary_layer
                                ]
                            },
                            cache_replacement_layers=cache_layers,
                            cache_sources=cache_sources,
                            capture_internal_state=context_interfaces,
                        )
                        interface_archive.add_replay(
                            injection_layer=layer_index,
                            boundary_layer=boundary_layer,
                            boundary_kind=boundary_kind,
                            decision=replayed,
                        )
                        boundary_replays.append(
                            boundary_replay_record(
                                runtime,
                                original=faulted,
                                replay=replayed,
                                injection_layer=layer_index,
                                boundary_layer=boundary_layer,
                                boundary_kind=boundary_kind,
                            )
                        )
                    except Exception as error:
                        boundary_replays.append(
                            error_record(
                                "boundary_replay_exception",
                                error,
                                injection_layer=layer_index,
                                boundary_layer=boundary_layer,
                                boundary_kind=boundary_kind,
                            )
                        )
            if not record["executed_command"]["exact_equal"]:
                candidates.append((layer_index, faulted, record))
        except Exception as error:
            local_records.append(
                error_record(
                    "local_intervention_exception",
                    error,
                    layer_index=layer_index,
                    site_id=site["site_id"],
                )
            )

    repeated_clean = policy_decision(
        runtime,
        policy_config,
        model,
        processor,
        captured.observation,
        task_description,
        injector=injector,
        action_token_position=token_position,
        capture_internal_state=context_interfaces,
        capture_context_state=context_interfaces,
    )
    repeated_clean_record = {
        "command_exact_equal": bool(
            runtime.np.array_equal(clean.command, repeated_clean.command)
        ),
        "raw_action_exact_equal": bool(
            runtime.np.array_equal(clean.raw_action, repeated_clean.raw_action)
        ),
        "action_tokens_exact_equal": clean.action_tokens == repeated_clean.action_tokens,
        "safe_feature_exact_equal": bool(
            runtime.torch.equal(clean.hidden_states, repeated_clean.hidden_states)
        ),
        "hook_anomalies": list(repeated_clean.trace.anomalies),
    }
    if interface_enabled:
        try:
            repeated_clean_record["full_trace"] = trace_repeatability(
                runtime.torch, clean, repeated_clean
            )
        except Exception as error:
            repeated_clean_record["full_trace"] = error_record(
                "clean_trace_repeatability_exception", error
            )
    feature_path = context_dir / "local_features.pkl"
    write_pickle_atomic(
        feature_path,
        {
            "context_id": context["context_id"],
            "action_token_position": token_position,
            "clean_hidden_states": clean.hidden_states,
            "faulted_hidden_states_by_layer": faulted_features,
        },
    )
    interface_archive_record = None
    interface_archive_error = None
    if interface_archive is not None:
        try:
            interface_archive_record = interface_archive.write(
                context_dir / "language_interfaces.npz"
            )
        except Exception as error:
            interface_archive_error = f"{type(error).__name__}: {error}"
            write_json_atomic(
                context_dir / "language_interfaces.error.json",
                error_record("language_interface_archive_exception", error),
            )
    write_json_atomic(
        context_dir / "local.json",
        {
            "schema_version": 2 if interface_enabled else 1,
            "status": "complete",
            "context": context,
            "task_description": task_description,
            "captured_simulator_state_sha256": captured.simulator_state_sha256,
            "captured_context_archive": captured_context_archive,
            "source_hook_calls": captured.source_trace.call_counts,
            "source_hook_anomalies": list(captured.source_trace.anomalies),
            "clean_hook_calls": clean.trace.call_counts,
            "clean_hook_anomalies": list(clean.trace.anomalies),
            "clean_inference_seconds": clean.inference_seconds,
            "feature_archive": artifact_record(feature_path),
            "interface_archive": interface_archive_record,
            "interface_archive_error": interface_archive_error,
            "boundary_replays": boundary_replays,
            "repeated_clean": repeated_clean_record,
            "interventions": local_records,
        },
    )

    branch_results = []
    control_fault = {
        "kind": "language_block_current_control",
        "policy_step": int(context["policy_step"]),
        "source_policy_step": int(context["source_policy_step"]),
        "action_token_position": token_position,
        "branch_state_restoration": branch_state_restoration,
    }
    command_groups = _command_groups(runtime, candidates)
    if terminal_branches:
        control_result = _run_branch(
            output_dir=output_dir / "attempts" / f"{context['context_id']}-control",
            runtime=runtime,
            policy_config=policy_config,
            model=model,
            processor=processor,
            env=env,
            task_description=task_description,
            initial_state=initial_state,
            context=context,
            captured=captured,
            decision=clean,
            fault=control_fault,
            execution=execution,
            condition="activation_control",
            wait_steps=wait_steps,
            restore_directly=restore_directly,
        )
        branch_results.append({"branch": "control", "result": control_result})
        selected_groups, terminal_skip_reason = _select_terminal_groups(
            command_groups,
            control_result,
            maximum_faulted_terminal_branches,
        )
    else:
        selected_groups = []
        terminal_skip_reason = "disabled_by_frozen_instrumentation"
    for group in selected_groups:
        layer_index, decision, local_record = group["members"][0]
        group_record = _group_record(group)
        fault = {
            "kind": "language_block_temporal_replacement",
            "operator": "replace final action-token vector at t with its value at t-1",
            "policy_step": int(context["policy_step"]),
            "source_policy_step": int(context["source_policy_step"]),
            "action_token_position": token_position,
            "layer_index": layer_index,
            "site_id": local_record["site_id"],
            "command_group": group_record,
            "local_measurements": local_record,
            "branch_state_restoration": branch_state_restoration,
        }
        result = _run_branch(
            output_dir=output_dir
            / "attempts"
            / f"{context['context_id']}-command-{group['command_id'][:12]}",
            runtime=runtime,
            policy_config=policy_config,
            model=model,
            processor=processor,
            env=env,
            task_description=task_description,
            initial_state=initial_state,
            context=context,
            captured=captured,
            decision=decision,
            fault=fault,
            execution=execution,
            condition="activation_fault",
            wait_steps=wait_steps,
            restore_directly=restore_directly,
        )
        branch_results.append(
            {
                "branch": f"command-{group['command_id'][:12]}",
                "command_group": group_record,
                "result": result,
            }
        )

    selected_interventions = sum(
        len(group["members"]) for group in selected_groups
    )
    deferred_groups = (
        len(command_groups) - len(selected_groups)
        if terminal_skip_reason is None
        else 0
    )
    deferred_interventions = (
        len(candidates) - selected_interventions
        if terminal_skip_reason is None
        else 0
    )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "context": context,
        "branch_state_restoration": branch_state_restoration,
        "local_interventions": len(local_records),
        "local_errors": sum(record.get("status") == "error" for record in local_records),
        "command_changing_interventions": len(candidates),
        "unique_faulted_commands": len(command_groups),
        "command_groups": [_group_record(group) for group in command_groups],
        "faulted_terminal_branches_run": len(selected_groups),
        "faulted_terminal_interventions_represented": selected_interventions,
        "faulted_terminal_branches_deferred_by_limit": deferred_groups,
        "faulted_terminal_interventions_deferred_by_limit": deferred_interventions,
        "faulted_terminal_skip_reason": terminal_skip_reason,
        "faulted_terminal_branches_skipped_by_control": (
            len(command_groups) if terminal_skip_reason is not None else 0
        ),
        "faulted_terminal_interventions_skipped_by_control": (
            len(candidates) if terminal_skip_reason is not None else 0
        ),
        "terminal_unresolved": sum(
            item["result"].get("status") == "unresolved" for item in branch_results
        ),
        "branches": branch_results,
        "repeated_clean": repeated_clean_record,
        "interface_archive_complete": interface_archive_record is not None,
        "interface_archive_error": interface_archive_error,
        "boundary_replays_complete": sum(
            record.get("status") == "complete" for record in boundary_replays
        ),
        "boundary_replays_error": sum(
            record.get("status") == "error" for record in boundary_replays
        ),
        "finished_at": now(),
    }
    write_json_atomic(complete_path, summary)
    return summary
