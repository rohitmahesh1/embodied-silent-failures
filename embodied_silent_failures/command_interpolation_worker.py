from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.command_interpolation import interpolate_command
from embodied_silent_failures.language_context import (
    capture_context,
    run_terminal_branch,
    write_terminal_branch,
)
from embodied_silent_failures.language_policy import policy_decision
from embodied_silent_failures.language_worker import error_record
from embodied_silent_failures.openvla_runtime import array_sha256
from embodied_silent_failures.provenance import file_sha256, load_json


CONDITION = "command_interpolation_task_boundary"
COMMAND_COLUMNS = (
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
)


def _lambda_label(value: float) -> str:
    return f"lambda-{value:.6f}".replace(".", "p")


def _archived_prefix(
    source_campaign_dir: Path,
    context: dict[str, Any],
    runtime: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # Campaign commit 88a0def, language_context.py::capture_context wrote these
    # commands to the control CSV. Replaying them defines the same causal state
    # without assuming a fresh OpenVLA inference reproduces every prefix bit.
    context_id = str(context["context_id"])
    control_dir = source_campaign_dir / "attempts" / f"{context_id}-control"
    completions = sorted(control_dir.glob("*.complete.json"))
    if len(completions) != 1:
        raise ValueError(
            f"expected one archived control completion for {context_id}, "
            f"found {len(completions)}"
        )
    completion = load_json(completions[0])
    if completion.get("status") != "complete" or not completion.get("success"):
        raise ValueError(f"archived control did not succeed for {context_id}")
    csv_path = control_dir / completion["files"]["csv"]
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    policy_steps = int(context["policy_step"])
    if len(rows) <= policy_steps:
        raise ValueError(f"archived control is too short for {context_id}")
    prefix = []
    for step, row in enumerate(rows[:policy_steps]):
        if int(row["action/timestep"]) != step:
            raise ValueError(f"archived control timestep changed for {context_id}")
        command = runtime.np.asarray(
            [float(row[column]) for column in COMMAND_COLUMNS], dtype=float
        )
        if not bool(runtime.np.isfinite(command).all()):
            raise ValueError(f"archived control command is non-finite for {context_id}")
        prefix.append(command)
    return tuple(prefix), {
        "completion_sha256": file_sha256(completions[0]),
        "csv_sha256": file_sha256(csv_path),
        "commands": len(prefix),
    }


def _run_branch(
    *,
    output_root: Path,
    wait_steps: int,
    branch_plan: dict[str, Any],
    interpolation: float,
    captured: Any,
    clean_decision: Any,
    runtime: Any,
    policy_config: Any,
    model: Any,
    processor: Any,
    env: Any,
    task_description: str,
    initial_state: Any,
    execution: dict[str, Any],
    source_context_sha256: str,
    source_prefix: dict[str, Any],
) -> dict[str, Any]:
    context = branch_plan["context"]
    output_dir = (
        output_root
        / "attempts"
        / str(branch_plan["physical_run"])
        / _lambda_label(interpolation)
    )
    completion = output_dir / (
        f"task{context['task_id']}--ep{context['episode_index']}.complete.json"
    )
    if completion.is_file():
        return load_json(completion)

    clean = [float(value) for value in branch_plan["clean_command"]]
    failed = [float(value) for value in branch_plan["faulted_command"]]
    delta = [
        target - reference
        for reference, target in zip(clean, failed, strict=True)
    ]
    current_clean = runtime.np.asarray(clean_decision.command, dtype=float)
    clean_error = float(
        runtime.np.max(runtime.np.abs(current_clean - runtime.np.asarray(clean)))
    )
    command = current_clean + interpolation * runtime.np.asarray(delta, dtype=float)
    command[-1] = current_clean[-1]
    declared = interpolate_command(clean, failed, interpolation)
    target = replace(clean_decision, command=command)
    fault = {
        "kind": "direct_executed_command_interpolation",
        "policy_step": int(context["policy_step"]),
        "interpolation": interpolation,
        "source_physical_run": branch_plan["physical_run"],
        "source_context_sha256": source_context_sha256,
        "source_prefix": source_prefix,
        "archived_clean_command": clean,
        "archived_failed_command": failed,
        "archived_interpolated_command": declared,
        "executed_command": command.tolist(),
        "maximum_archived_clean_command_error": clean_error,
        "evidence_scope": (
            "Task consequence only. The shared rollout writer retains the clean policy "
            "feature at the intervention step; this canary does not score a monitor."
        ),
    }
    errors = []
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            result = run_terminal_branch(
                runtime,
                policy_config,
                model,
                processor,
                env,
                task_description,
                initial_state,
                context,
                captured,
                target,
                wait_steps=wait_steps,
                restore_directly=True,
            )
            return write_terminal_branch(
                output_dir,
                runtime,
                context,
                task_description,
                result,
                fault,
                execution,
                condition=CONDITION,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            write_json_atomic(
                output_dir / f"attempt-{attempt}.error.json",
                error_record(
                    "command_interpolation_branch_exception",
                    error,
                    context=context,
                    interpolation=interpolation,
                    attempt=attempt,
                ),
            )
            runtime.torch.cuda.empty_cache()
    write_json_atomic(
        output_dir / "branch.unresolved.json",
        {
            "schema_version": 1,
            "status": "unresolved",
            "context": context,
            "interpolation": interpolation,
            "errors": errors,
        },
    )
    return {"status": "unresolved", "errors": errors}


def run_planned_interpolations(
    *,
    output_root: Path,
    source_campaign_dir: Path,
    wait_steps: int,
    branch_plan: dict[str, Any],
    runtime: Any,
    policy_config: Any,
    model: Any,
    processor: Any,
    injector: Any,
    env: Any,
    task_description: str,
    initial_state: Any,
    execution: dict[str, Any],
) -> list[dict[str, Any]]:
    context = branch_plan["context"]
    context_id = str(context["context_id"])
    if array_sha256(runtime, initial_state) != context["initial_state_sha256"]:
        raise ValueError(f"LIBERO initial state changed for {context_id}")
    archived_path = source_campaign_dir / "contexts" / context_id / "local.json"
    archived = load_json(archived_path)
    if archived["context"] != context:
        raise ValueError(f"archived context metadata changed for {context_id}")
    runtime.set_seed_everywhere(int(context["trial_seed"]))
    prefix, prefix_provenance = _archived_prefix(
        source_campaign_dir, context, runtime
    )
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
        executed_prefix=prefix,
    )
    if (
        captured.simulator_state_sha256
        != archived["captured_simulator_state_sha256"]
    ):
        raise ValueError(f"recaptured simulator state changed for {context_id}")
    clean = policy_decision(
        runtime,
        policy_config,
        model,
        processor,
        captured.observation,
        task_description,
        injector=injector,
        action_token_position=int(context["action_token_position"]),
    )
    archived_clean = runtime.np.asarray(branch_plan["clean_command"], dtype=float)
    clean_error = float(
        runtime.np.max(
            runtime.np.abs(
                runtime.np.asarray(clean.command, dtype=float) - archived_clean
            )
        )
    )
    if clean_error > 1e-6:
        raise ValueError(f"recaptured clean command drifted by {clean_error:.3g}")

    return [
        _run_branch(
            output_root=output_root,
            wait_steps=wait_steps,
            branch_plan=branch_plan,
            interpolation=float(interpolation),
            captured=captured,
            clean_decision=clean,
            runtime=runtime,
            policy_config=policy_config,
            model=model,
            processor=processor,
            env=env,
            task_description=task_description,
            initial_state=initial_state,
            execution=execution,
            source_context_sha256=file_sha256(archived_path),
            source_prefix=prefix_provenance,
        )
        for interpolation in branch_plan["lambdas"]
    ]
