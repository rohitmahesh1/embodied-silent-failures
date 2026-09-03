from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from embodied_silent_failures.atlas_policy import atlas_policy_decision
from embodied_silent_failures.language_context import (
    CapturedContext,
    ContextTerminatedEarly,
    action_row,
    start_episode,
)
from embodied_silent_failures.openvla_runtime import array_sha256


@dataclass(frozen=True)
class AtlasContext:
    rollout: CapturedContext
    source_values: dict[str, Any]
    source_errors: dict[str, str]
    source_missing_site_ids: tuple[str, ...]


def capture_atlas_context(
    runtime: Any,
    policy_config: Any,
    model: Any,
    processor: Any,
    collector: Any,
    env: Any,
    task_description: str,
    initial_state: Any,
    context: dict[str, Any],
    *,
    wait_steps: int,
) -> AtlasContext:
    observation = start_episode(runtime, env, initial_state, wait_steps)
    prefix_commands = []
    prefix_hidden_states = []
    prefix_rows = []
    source_decision = None
    source_values = None
    source_errors = None
    source_missing = None
    source_step = int(context["source_policy_step"])
    for policy_step in range(int(context["policy_step"])):
        capture_source = policy_step == source_step
        if capture_source:
            collector.begin_capture()
        decision = atlas_policy_decision(
            runtime,
            policy_config,
            model,
            processor,
            observation,
            task_description,
            policy_step=policy_step,
            adapter=collector if capture_source else None,
        )
        if capture_source:
            source_decision = decision
            source_values = collector.values
            source_errors = collector.errors
            source_missing = tuple(collector.missing_site_ids())
        command = runtime.np.asarray(decision.command).copy()
        row = action_row(policy_step, command, decision)
        observation, reward, done, _ = env.step(command.tolist())
        row["environment/reward"] = reward
        row["environment/done"] = bool(done)
        prefix_commands.append(command)
        prefix_hidden_states.append(decision.hidden_states)
        prefix_rows.append(row)
        if done:
            raise ContextTerminatedEarly(
                f"clean rerun succeeded at step {policy_step} before context "
                f"{context['context_id']}"
            )
    if source_decision is None or source_values is None:
        raise RuntimeError("previous-decision atlas values were not captured")
    simulator_state = runtime.np.asarray(env.get_sim_state()).copy()
    rollout = CapturedContext(
        observation={
            key: runtime.np.asarray(value).copy() for key, value in observation.items()
        },
        simulator_state=simulator_state,
        simulator_state_sha256=array_sha256(runtime, simulator_state),
        prefix_commands=tuple(prefix_commands),
        prefix_hidden_states=tuple(prefix_hidden_states),
        prefix_rows=tuple(prefix_rows),
        source_trace=None,
        source_decision=source_decision,
    )
    return AtlasContext(
        rollout=rollout,
        source_values=source_values,
        source_errors=source_errors or {},
        source_missing_site_ids=source_missing or (),
    )
