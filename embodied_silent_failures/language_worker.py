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
    write_terminal_branch,
)
from embodied_silent_failures.language_fault import LanguageBlockInjector
from embodied_silent_failures.language_policy import (
    intervention_record,
    policy_decision,
)
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
    )
    local_records = []
    candidates = []
    faulted_features = {}
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
    write_json_atomic(
        context_dir / "local.json",
        {
            "schema_version": 1,
            "status": "complete",
            "context": context,
            "captured_simulator_state_sha256": captured.simulator_state_sha256,
            "source_hook_calls": captured.source_trace.call_counts,
            "source_hook_anomalies": list(captured.source_trace.anomalies),
            "clean_hook_calls": clean.trace.call_counts,
            "clean_hook_anomalies": list(clean.trace.anomalies),
            "clean_inference_seconds": clean.inference_seconds,
            "feature_archive": artifact_record(feature_path),
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
    }
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
    )
    branch_results.append({"branch": "control", "result": control_result})

    selected_candidates = candidates
    if maximum_faulted_terminal_branches is not None:
        selected_candidates = candidates[:maximum_faulted_terminal_branches]
    for layer_index, decision, local_record in selected_candidates:
        fault = {
            "kind": "language_block_temporal_replacement",
            "operator": "replace final action-token vector at t with its value at t-1",
            "policy_step": int(context["policy_step"]),
            "source_policy_step": int(context["source_policy_step"]),
            "action_token_position": token_position,
            "layer_index": layer_index,
            "site_id": local_record["site_id"],
            "local_measurements": local_record,
        }
        result = _run_branch(
            output_dir=output_dir
            / "attempts"
            / f"{context['context_id']}-layer{layer_index:02d}",
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
        )
        branch_results.append({"branch": f"layer{layer_index:02d}", "result": result})

    summary = {
        "schema_version": 1,
        "status": "complete",
        "context": context,
        "local_interventions": len(local_records),
        "local_errors": sum(record.get("status") == "error" for record in local_records),
        "command_changing_interventions": len(candidates),
        "faulted_terminal_branches_run": len(selected_candidates),
        "faulted_terminal_branches_deferred_by_limit": len(candidates)
        - len(selected_candidates),
        "terminal_unresolved": sum(
            item["result"].get("status") == "unresolved" for item in branch_results
        ),
        "branches": branch_results,
        "repeated_clean": repeated_clean_record,
        "finished_at": now(),
    }
    write_json_atomic(complete_path, summary)
    return summary
