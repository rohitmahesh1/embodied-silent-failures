from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import (
    artifact_record,
    write_json_atomic,
    write_pickle_atomic,
)
from embodied_silent_failures.atlas_context import capture_atlas_context
from embodied_silent_failures.atlas_policy import atlas_policy_decision
from embodied_silent_failures.language_context import (
    write_captured_context_archive,
)
from embodied_silent_failures.language_fault import tensor_change
from embodied_silent_failures.language_policy import array_change
from embodied_silent_failures.language_worker import (
    error_record,
    now,
    run_resilient_terminal_branch,
)
from embodied_silent_failures.openvla_runtime import array_sha256
from embodied_silent_failures.provenance import load_json
from embodied_silent_failures.temporal_fault import (
    TemporalReplacementInjector,
    TemporalReplacementSpec,
)
from embodied_silent_failures.temporal_values import (
    TemporalValueCollector,
    write_temporal_value_archive,
)


ATLAS_STATE_RECONSTRUCTION = (
    "reset to the episode initial state and replay the captured executed-command prefix"
)


def _token_change(clean: tuple[int, ...], faulted: tuple[int, ...]) -> dict[str, Any]:
    if len(clean) != len(faulted):
        raise ValueError("cannot compare action-token sequences with different lengths")
    changed = [index for index, values in enumerate(zip(clean, faulted)) if values[0] != values[1]]
    return {
        "exact_equal": not changed,
        "changed_token_count": len(changed),
        "changed_token_positions": changed,
        "clean": list(clean),
        "faulted": list(faulted),
    }


def _local_record(
    runtime: Any,
    site: dict[str, Any],
    clean: Any,
    faulted: Any,
    fault: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "complete",
        "site_id": site["site_id"],
        "topologies": site["topologies"],
        "sampling": site["sampling"],
        "identity": site["identity"],
        "fault": fault,
        "raw_action": array_change(runtime.np, clean.raw_action, faulted.raw_action),
        "executed_command": array_change(runtime.np, clean.command, faulted.command),
        "action_tokens": _token_change(clean.action_tokens, faulted.action_tokens),
        "action_logits": tensor_change(
            runtime.torch,
            clean.generation_logits.action_token_logits,
            faulted.generation_logits.action_token_logits,
        ),
        "safe_input": tensor_change(
            runtime.torch, clean.hidden_states, faulted.hidden_states
        ),
        "inference_seconds": faulted.inference_seconds,
    }


def _command_groups(
    runtime: Any, candidates: list[tuple[str, Any, dict[str, Any]]]
) -> list[dict[str, Any]]:
    # SAFE OpenVLA 300dce26, modeling_prismatic.py::predict_action, supplies one
    # command to LIBERO at the intervention decision. Because all later policy
    # calls are clean, exact command identity gives an auditable shared physical
    # continuation while each member retains its own SAFE input.
    groups: dict[str, dict[str, Any]] = {}
    for site_id, decision, local_record in candidates:
        command_id = array_sha256(runtime, decision.command)
        group = groups.setdefault(
            command_id,
            {
                "command_id": command_id,
                "executed_command": decision.command.tolist(),
                "members": [],
            },
        )
        group["members"].append((site_id, decision, local_record))
    return list(groups.values())


def _group_record(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_id": group["command_id"],
        "executed_command": group["executed_command"],
        "representative_site_id": str(group["members"][0][0]),
        "member_site_ids": [str(value[0]) for value in group["members"]],
    }


def run_atlas_context(
    *,
    output_dir: Path,
    wait_steps: int,
    maximum_faulted_terminal_branches: int | None,
    context: dict[str, Any],
    sites: list[dict[str, Any]],
    runtime: Any,
    policy_config: Any,
    model: Any,
    processor: Any,
    collector: TemporalValueCollector,
    env: Any,
    task_description: str,
    initial_state: Any,
    execution: dict[str, Any],
) -> dict[str, Any]:
    context_dir = output_dir / "contexts" / str(context["context_id"])
    context_dir.mkdir(parents=True, exist_ok=True)
    complete_path = context_dir / "context.complete.json"
    if complete_path.is_file():
        return load_json(complete_path)

    runtime.set_seed_everywhere(int(context["trial_seed"]))
    captured = capture_atlas_context(
        runtime,
        policy_config,
        model,
        processor,
        collector,
        env,
        task_description,
        initial_state,
        context,
        wait_steps=wait_steps,
    )
    captured_path = context_dir / "captured_context.npz"
    captured_record = write_captured_context_archive(
        captured_path, runtime, captured.rollout
    )

    collector.begin_capture()
    clean = atlas_policy_decision(
        runtime,
        policy_config,
        model,
        processor,
        captured.rollout.observation,
        task_description,
        policy_step=int(context["policy_step"]),
        adapter=collector,
    )
    current_values = collector.values
    current_errors = collector.errors
    current_missing = collector.missing_site_ids()
    values_path = context_dir / "temporal_values.npz"
    values_record = None
    values_error = None
    try:
        values_record = write_temporal_value_archive(
            values_path,
            runtime,
            sites,
            captured.source_values,
            current_values,
        )
    except Exception as error:
        values_error = f"{type(error).__name__}: {error}"
        write_json_atomic(
            context_dir / "temporal_values.error.json",
            error_record("temporal_value_archive_exception", error),
        )

    local_records = []
    candidates = []
    faulted_evidence = {}
    for site in sites:
        site_id = str(site["site_id"])
        if site_id not in captured.source_values or site_id not in current_values:
            local_records.append(
                {
                    "schema_version": 1,
                    "status": "unresolved",
                    "reason": "site_missing_from_source_or_current_decision",
                    "site_id": site_id,
                    "source_error": captured.source_errors.get(site_id),
                    "current_error": current_errors.get(site_id),
                    "source_missing": site_id in captured.source_missing_site_ids,
                    "current_missing": site_id in current_missing,
                }
            )
            continue
        injector = TemporalReplacementInjector(
            runtime.torch,
            runtime.np,
            TemporalReplacementSpec(
                site_id=site_id,
                identity=site["identity"],
                policy_step=int(context["policy_step"]),
                source_policy_step=int(context["source_policy_step"]),
                value_slice=str(site["intervention"]["value_slice"]),
            ),
        )
        try:
            injector.install(model)
            injector.begin_trial(
                int(context["trial_seed"]),
                source_value=captured.source_values[site_id],
            )
            faulted = atlas_policy_decision(
                runtime,
                policy_config,
                model,
                processor,
                captured.rollout.observation,
                task_description,
                policy_step=int(context["policy_step"]),
                adapter=injector,
            )
            fault = injector.require_injected()
            record = _local_record(runtime, site, clean, faulted, fault)
            local_records.append(record)
            faulted_evidence[site_id] = {
                "hidden_states": faulted.hidden_states,
                "action_token_logits": faulted.generation_logits.action_token_logits,
                "action_tokens": faulted.action_tokens,
                "raw_action": faulted.raw_action,
                "executed_command": faulted.command,
            }
            if not record["executed_command"]["exact_equal"]:
                candidates.append((site_id, faulted, record))
        except Exception as error:
            local_records.append(
                error_record(
                    "local_intervention_exception",
                    error,
                    site_id=site_id,
                    identity=site["identity"],
                )
            )
        finally:
            injector.close()

    evidence_path = context_dir / "local_evidence.pkl"
    write_pickle_atomic(
        evidence_path,
        {
            "context_id": context["context_id"],
            "source": {
                "hidden_states": captured.rollout.source_decision.hidden_states,
                "action_token_logits": (
                    captured.rollout.source_decision.generation_logits.action_token_logits
                ),
                "action_tokens": captured.rollout.source_decision.action_tokens,
                "raw_action": captured.rollout.source_decision.raw_action,
                "executed_command": captured.rollout.source_decision.command,
            },
            "clean": {
                "hidden_states": clean.hidden_states,
                "action_token_logits": clean.generation_logits.action_token_logits,
                "action_tokens": clean.action_tokens,
                "raw_action": clean.raw_action,
                "executed_command": clean.command,
            },
            "faulted_by_site": faulted_evidence,
        },
    )
    write_json_atomic(
        context_dir / "local.json",
        {
            "schema_version": 1,
            "status": "complete",
            "context": context,
            "task_description": task_description,
            "captured_simulator_state_sha256": captured.rollout.simulator_state_sha256,
            "captured_context_archive": captured_record,
            "temporal_value_archive": values_record,
            "temporal_value_archive_error": values_error,
            "local_evidence_archive": artifact_record(evidence_path),
            "source_collection": {
                "captured": len(captured.source_values),
                "errors": captured.source_errors,
                "missing_site_ids": list(captured.source_missing_site_ids),
            },
            "current_collection": {
                "captured": len(current_values),
                "errors": current_errors,
                "missing_site_ids": current_missing,
            },
            "interventions": local_records,
        },
    )

    branches = []
    control_fault = {
        "kind": "graph_atlas_current_control",
        "policy_step": int(context["policy_step"]),
        "source_policy_step": int(context["source_policy_step"]),
        "state_reconstruction": ATLAS_STATE_RECONSTRUCTION,
    }
    control = run_resilient_terminal_branch(
        output_dir=output_dir / "attempts" / f"{context['context_id']}-control",
        runtime=runtime,
        policy_config=policy_config,
        model=model,
        processor=processor,
        env=env,
        task_description=task_description,
        initial_state=initial_state,
        context=context,
        captured=captured.rollout,
        decision=clean,
        fault=control_fault,
        execution=execution,
        condition="atlas_control",
        wait_steps=wait_steps,
        restore_directly=False,
    )
    branches.append({"branch": "control", "result": control})
    command_groups = _command_groups(runtime, candidates)
    if control.get("status") != "complete":
        selected_groups = []
        skip_reason = "control_unresolved"
    elif control.get("success") is not True:
        selected_groups = []
        skip_reason = "control_failed"
    else:
        selected_groups = command_groups
        skip_reason = None
        if maximum_faulted_terminal_branches is not None:
            selected_groups = selected_groups[:maximum_faulted_terminal_branches]

    for group in selected_groups:
        site_id, decision, local_record = group["members"][0]
        group_record = _group_record(group)
        fault = {
            "kind": "graph_atlas_temporal_replacement",
            "operator": "replace x_t with the same site's x_(t-1)",
            "policy_step": int(context["policy_step"]),
            "source_policy_step": int(context["source_policy_step"]),
            "site_id": site_id,
            "command_group": group_record,
            "representative_local_measurements": local_record,
            "state_reconstruction": ATLAS_STATE_RECONSTRUCTION,
        }
        result = run_resilient_terminal_branch(
            output_dir=(
                output_dir
                / "attempts"
                / f"{context['context_id']}-command-{group['command_id'][:12]}"
            ),
            runtime=runtime,
            policy_config=policy_config,
            model=model,
            processor=processor,
            env=env,
            task_description=task_description,
            initial_state=initial_state,
            context=context,
            captured=captured.rollout,
            decision=decision,
            fault=fault,
            execution=execution,
            condition="atlas_temporal_fault",
            wait_steps=wait_steps,
            restore_directly=False,
        )
        branches.append(
            {
                "branch": f"command-{group['command_id'][:12]}",
                "command_group": group_record,
                "result": result,
            }
        )

    selected_sites = sum(len(group["members"]) for group in selected_groups)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "context": context,
        "local_interventions": len(local_records),
        "local_complete": sum(value.get("status") == "complete" for value in local_records),
        "local_unresolved": sum(value.get("status") != "complete" for value in local_records),
        "command_changing_interventions": len(candidates),
        "unique_faulted_commands": len(command_groups),
        "command_groups": [_group_record(group) for group in command_groups],
        "faulted_terminal_branches_run": len(selected_groups),
        "faulted_terminal_interventions_represented": selected_sites,
        "faulted_terminal_branches_deferred_by_limit": (
            max(0, len(command_groups) - len(selected_groups))
            if skip_reason is None
            else 0
        ),
        "faulted_terminal_skip_reason": skip_reason,
        "terminal_unresolved": sum(
            value["result"].get("status") == "unresolved" for value in branches
        ),
        "branches": branches,
        "finished_at": now(),
    }
    write_json_atomic(complete_path, summary)
    return summary
