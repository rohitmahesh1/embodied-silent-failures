from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import artifact_record, write_npz_atomic
from embodied_silent_failures.openvla_runtime import array_sha256


@dataclass
class TrajectoryRecorder:
    runtime: Any
    policy_steps: list[int] = field(default_factory=list)
    stages: list[int] = field(default_factory=list)
    simulator_states: list[Any] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    decision_policy_steps: list[int] = field(default_factory=list)
    raw_actions: list[Any] = field(default_factory=list)
    executed_commands: list[Any] = field(default_factory=list)
    action_tokens: list[Any] = field(default_factory=list)
    sequence_token_ids: list[Any] = field(default_factory=list)
    action_logits: list[Any] = field(default_factory=list)
    top_token_ids: list[Any] = field(default_factory=list)
    top_token_logits: list[Any] = field(default_factory=list)
    log_normalizers: list[Any] = field(default_factory=list)
    entropies: list[Any] = field(default_factory=list)

    def append(
        self,
        *,
        policy_step: int,
        stage: str,
        observation: dict[str, Any],
        simulator_state: Any,
    ) -> None:
        stage_ids = {"before_action": 0, "after_final_action": 1}
        if stage not in stage_ids:
            raise ValueError(f"unknown trajectory snapshot stage: {stage}")
        self.policy_steps.append(int(policy_step))
        self.stages.append(stage_ids[stage])
        self.simulator_states.append(
            self.runtime.np.asarray(simulator_state).copy()
        )
        self.observations.append(
            {
                str(name): self.runtime.np.asarray(value).copy()
                for name, value in observation.items()
            }
        )

    def append_decision(
        self,
        *,
        policy_step: int,
        decision: Any,
        executed_command: Any,
    ) -> None:
        logits = decision.generation_logits
        self.decision_policy_steps.append(int(policy_step))
        self.raw_actions.append(self.runtime.np.asarray(decision.raw_action).copy())
        self.executed_commands.append(
            self.runtime.np.asarray(executed_command).copy()
        )
        self.action_tokens.append(
            self.runtime.np.asarray(decision.action_tokens, dtype=self.runtime.np.int32)
        )
        self.sequence_token_ids.append(logits.sequence_token_ids.numpy().copy())
        self.action_logits.append(logits.action_token_logits.numpy().copy())
        self.top_token_ids.append(logits.top_token_ids.numpy().copy())
        self.top_token_logits.append(logits.top_token_logits.numpy().copy())
        self.log_normalizers.append(logits.log_normalizer.numpy().copy())
        self.entropies.append(logits.entropy.numpy().copy())

    def write(self, path: Path) -> dict[str, Any]:
        if not self.simulator_states:
            raise ValueError("cannot archive an empty terminal trajectory")
        np = self.runtime.np
        arrays: dict[str, Any] = {
            "policy_step": np.asarray(self.policy_steps, dtype=np.int32),
            "snapshot_stage": np.asarray(self.stages, dtype=np.uint8),
            "simulator_state": np.stack(self.simulator_states, axis=0),
        }
        if self.decision_policy_steps:
            arrays.update(
                {
                    "decision_policy_step": np.asarray(
                        self.decision_policy_steps, dtype=np.int32
                    ),
                    "raw_action": np.stack(self.raw_actions, axis=0),
                    "executed_command": np.stack(self.executed_commands, axis=0),
                    "action_tokens": np.stack(self.action_tokens, axis=0),
                    "sequence_token_ids": np.stack(
                        self.sequence_token_ids, axis=0
                    ),
                    "action_token_logits": np.stack(self.action_logits, axis=0),
                    "global_top_token_ids": np.stack(self.top_token_ids, axis=0),
                    "global_top_token_logits": np.stack(
                        self.top_token_logits, axis=0
                    ),
                    "action_log_normalizer": np.stack(
                        self.log_normalizers, axis=0
                    ),
                    "action_entropy": np.stack(self.entropies, axis=0),
                }
            )
        shared_names = set(self.observations[0])
        for observation in self.observations[1:]:
            shared_names &= set(observation)
        excluded = []
        records = []
        for index, name in enumerate(sorted(shared_names)):
            values = [observation[name] for observation in self.observations]
            first = values[0]
            if first.dtype.hasobject or first.dtype.kind not in "biufc":
                excluded.append({"name": name, "reason": "non-numeric dtype"})
                continue
            if any(
                value.shape != first.shape or value.dtype != first.dtype
                for value in values[1:]
            ):
                excluded.append(
                    {"name": name, "reason": "shape or dtype changed across snapshots"}
                )
                continue
            archive_key = f"observation_{index:04d}"
            stacked = np.stack(values, axis=0)
            arrays[archive_key] = stacked
            records.append(
                {
                    "name": name,
                    "archive_key": archive_key,
                    "shape": list(stacked.shape),
                    "dtype": stacked.dtype.str,
                    "sha256": array_sha256(self.runtime, stacked),
                    "kind": "image" if "image" in name.lower() else "numeric",
                }
            )
        missing_names = sorted(
            set().union(*(set(value) for value in self.observations)) - shared_names
        )
        excluded.extend(
            {"name": name, "reason": "not present in every snapshot"}
            for name in missing_names
        )
        write_npz_atomic(path, np, arrays)
        states = arrays["simulator_state"]
        return {
            "schema_version": 1,
            "artifact": artifact_record(path),
            "snapshot_count": len(self.policy_steps),
            "decision_count": len(self.decision_policy_steps),
            "snapshot_stage_ids": {"0": "before_action", "1": "after_final_action"},
            "simulator_state": {
                "shape": list(states.shape),
                "dtype": states.dtype.str,
                "sha256": array_sha256(self.runtime, states),
            },
            "observations": records,
            "excluded_observations": excluded,
            "policy_evidence": {
                "sequence_token_ids": (
                    "complete prompt and generated token IDs at every decision"
                ),
                "action_token_logits": (
                    "all 256 OpenVLA action-vocabulary logits for all seven "
                    "generated action tokens at every post-intervention decision"
                ),
                "global_top_tokens": 32,
                "normalization": "exact full-vocabulary logsumexp and entropy",
            },
        }
