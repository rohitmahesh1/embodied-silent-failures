import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.audit import audit_graph
from embodied_silent_failures.evidence_graph.openvla import (
    REQUIRED_ENDPOINTS as OPENVLA_ENDPOINTS,
    RecordingProcessor,
    capture_policy,
    contract_issues as openvla_contract_issues,
    executed_command,
    operator_annotations as openvla_annotations,
    record_current_image,
    record_current_observation,
    record_policy_image,
    record_policy_outputs,
    step_environment,
)
from embodied_silent_failures.evidence_graph.record import Recorder
from embodied_silent_failures.evidence_graph.reduce import reduce_graph
from embodied_silent_failures.evidence_graph.safe import monitored_features
from embodied_silent_failures.evidence_graph.torch_trace import (
    contract_issues as torch_trace_contract_issues,
)


ROLLOUT_ENDPOINTS = (
    "rollout.fault",
    "rollout.monitor_timeline",
    "rollout.outcome",
)
SAFE_EVIDENCE_ENDPOINTS = ("safe.monitor_input",)


def prepare_evidence_output(output_dir: Path) -> None:
    """Replace evidence for a pending trial, including a hard-killed trace."""
    if output_dir.exists():
        shutil.rmtree(output_dir)


def attach_monitor_timeline(
    evidence_dir: Path,
    monitor: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach post-hoc monitor results without changing the runtime trace."""
    composition_path = evidence_dir / "composition.json"
    if not composition_path.is_file():
        raise FileNotFoundError(f"evidence composition is missing: {composition_path}")
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    policy_steps = int(composition["policy_steps"])
    if [int(item["policy_step"]) for item in timeline] != list(range(policy_steps)):
        raise ValueError("monitor timeline must contain every policy step in order")
    attachment = {
        "schema_version": 1,
        "evidence_composition_sha256": _file_sha256(composition_path),
        "monitor": monitor,
        "timeline": timeline,
    }
    path = evidence_dir / "monitor-timeline.json"
    write_json_atomic(path, attachment)
    return attachment


def summarize_saturation(compositions: list[dict[str, Any]]) -> dict[str, Any]:
    observed = set()
    discovery = []
    traced_execution_count = 0
    for composition_index, composition in enumerate(compositions):
        for item in composition.get("timeline", []):
            digest = item.get("template_sha256")
            if digest is None:
                continue
            traced_execution_count += 1
            is_new = digest not in observed
            observed.add(digest)
            discovery.append(
                {
                    "composition_index": composition_index,
                    "policy_step": int(item["policy_step"]),
                    "template_sha256": digest,
                    "new_template": is_new,
                }
            )
    last_new = next(
        (
            index
            for index in range(len(discovery) - 1, -1, -1)
            if discovery[index]["new_template"]
        ),
        None,
    )
    return {
        "schema_version": 1,
        "composition_count": len(compositions),
        "traced_execution_count": traced_execution_count,
        "unique_template_count": len(observed),
        "executions_since_last_new_template": (
            len(discovery) - last_new - 1 if last_new is not None else None
        ),
        "discovery": discovery,
        "interpretation": (
            "Describes observed structural-template discovery only; it does not prove "
            "that untraced executions cannot introduce another template."
        ),
    }


class RolloutEvidence:
    """Record cheap rollout boundaries and selected full operator traces."""

    def __init__(
        self,
        output_dir: Path,
        metadata: dict[str, Any],
        traced_steps: set[int],
    ) -> None:
        self.output_dir = output_dir
        self.traced_steps = set(traced_steps)
        self.recorder = Recorder(output_dir / "raw.jsonl", metadata)
        self.templates: dict[str, dict[str, Any]] = {}
        self.timeline: list[dict[str, Any]] = []
        self._step_start = 0
        self._active_step: int | None = None
        self._fault_recorded = False
        self._step_count = 0
        self._closed = False
        self._monitor_inputs: list[Any] = []
        self._next_observations: list[Any] = []
        self._model_inputs: dict[int, Any] = {}
        self._operator_traced_steps: set[int] = set()
        self._capture_mode = "semantic_boundaries"

    def activation_fault_observer(
        self, before: Any, after: Any, record: dict[str, Any]
    ) -> None:
        policy_step = int(record["policy_step"])
        with self.recorder.scope(policy_step=policy_step):
            self.recorder.mark(
                "rollout.fault",
                inputs=before,
                outputs=after,
                basis="protocol:activation-bit-flip-v1:replace-hook-output-at-declared-site",
                region="fault_intervention",
                role="fault",
                lifetime="temporal",
                fault_interface=str(record["site"]),
                details=record,
            )
        self._fault_recorded = True

    def begin_step(
        self,
        policy_step: int,
        observation: dict[str, Any],
        current_image: Any,
        policy_image: Any,
        source_step: int,
        intervention: dict[str, Any] | None,
        *,
        policy_replayed: bool = False,
    ) -> None:
        if self._active_step is not None:
            raise RuntimeError("finish_step must be called before beginning another step")
        self._active_step = policy_step
        self._step_start = len(self.recorder.events)
        self._capture_mode = (
            "counterfactual_replay" if policy_replayed else "semantic_boundaries"
        )
        with self.recorder.scope(policy_step=policy_step):
            record_current_observation(
                self.recorder,
                observation,
                initial=self._step_count == 0,
                disposition=(
                    "policy_inference_replaced_by_counterfactual_replay"
                    if policy_replayed
                    else None
                ),
            )
            if not policy_replayed:
                record_current_image(self.recorder, observation, current_image)
                record_policy_image(
                    self.recorder,
                    current_image,
                    policy_image,
                    policy_step=policy_step,
                    source_step=source_step,
                )
            if intervention is not None:
                self.recorder.mark(
                    "rollout.fault",
                    inputs=self.recorder.lineage(
                        f"policy_step:{policy_step}:selected_image", policy_image
                    ),
                    outputs=self.recorder.lineage(
                        f"policy_step:{policy_step}:selected_image", policy_image
                    ),
                    basis="protocol:rollout-evidence-v1:record-applied-intervention",
                    region="fault_intervention",
                    role="fault",
                    lifetime="temporal",
                    fault_interface=str(intervention.get("kind", "intervention")),
                    details=intervention,
                )
                self._fault_recorded = True

    def processor(
        self,
        processor: Any,
        policy_image: Any,
        task_description: str,
        policy_step: int,
    ) -> Any:
        return RecordingProcessor(
            self.recorder,
            processor,
            policy_image,
            task_description,
            policy_step,
            lambda result: self._record_model_input(policy_step, result),
        )

    def policy_model(self, model: Any, policy_step: int) -> Any:
        if policy_step not in self.traced_steps:
            return model
        return _RecordingPolicy(self, model, policy_step)

    def policy_outputs(
        self,
        model: Any,
        generated: Any,
        raw_action: Any,
        unnorm_key: str,
        policy_step: int,
        torch: Any,
    ) -> None:
        with self.recorder.scope(policy_step=policy_step):
            model_input = self._model_inputs.pop(policy_step, None)
            if model_input is None:
                raise RuntimeError("recorded processor output is missing for policy call")
            if policy_step not in self._operator_traced_steps:
                self.recorder.mark(
                    "openvla.policy_call",
                    kind="opaque",
                    inputs=model_input,
                    outputs={
                        "sequences": generated["sequences"],
                        "final_layer_states": [
                            token_states[-1]
                            for token_states in generated["hidden_states"]
                        ],
                    },
                    basis=(
                        "code:openvla@300dce2:prismatic.extern.hf.modeling_prismatic."
                        "OpenVLAForActionPrediction.predict_action:return-generated-"
                        "sequences-and-hidden-states"
                    ),
                    region="policy_execution",
                    fault_interface="policy_call",
                    details={
                        "opaque_reason": (
                            "operator capture was intentionally disabled for this "
                            "unselected rollout step"
                        )
                    },
                )
            record_policy_outputs(
                self.recorder,
                model,
                generated,
                raw_action,
                unnorm_key,
                action_sink=False,
            )
            _per_token, monitor_input = monitored_features(
                self.recorder, torch, generated
            )
            self._monitor_inputs.append(monitor_input)

    def replayed_evidence(
        self,
        policy_step: int,
        command: Any,
        monitor_input: Any,
    ) -> Any:
        with self.recorder.scope(policy_step=policy_step):
            self.recorder.source(
                "libero.executed_command",
                self.recorder.lineage(
                    f"policy_step:{policy_step}:executed_command", command
                ),
                basis="protocol:counterfactual-replay-v1:executed-command-from-clean-artifact",
                region="command_replay",
                fault_interface="replayed_executed_command",
            )
            self.recorder.source(
                "safe.monitor_input",
                monitor_input,
                basis="protocol:counterfactual-replay-v1:safe-feature-from-clean-artifact",
                region="safe_feature_replay",
                role="monitor_input",
                fault_interface="replayed_safe_feature",
            )
        self._monitor_inputs.append(monitor_input)
        return command

    def command(self, runtime: Any, raw_action: Any, policy_step: int) -> Any:
        with self.recorder.scope(policy_step=policy_step):
            return executed_command(
                self.recorder, runtime, raw_action, command_sink=False
            )

    def environment_step(
        self,
        env: Any,
        command: Any,
        policy_step: int,
        *,
        policy_replayed: bool = False,
    ) -> tuple[Any, ...]:
        with self.recorder.scope(policy_step=policy_step):
            result = step_environment(
                self.recorder,
                env,
                command,
                next_observation_sink=False,
                next_observation_disposition=(
                    "policy_inference_replaced_by_counterfactual_replay"
                    if policy_replayed
                    else None
                ),
            )
        self._next_observations.append(result[0])
        return result

    def finish_step(
        self,
        policy_step: int,
        *,
        fault_applied: bool,
        reward: Any,
        done: bool,
    ) -> None:
        if self._active_step != policy_step:
            raise RuntimeError("rollout evidence step is not active")
        events = self.recorder.events[self._step_start :]
        signature = (
            structural_signature(events)
            if policy_step in self._operator_traced_steps
            else None
        )
        if signature is not None:
            digest = signature["sha256"]
            self.templates.setdefault(digest, signature)
        self.timeline.append(
            {
                "policy_step": policy_step,
                "fault_applied": bool(fault_applied),
                "reward": _plain_scalar(reward),
                "done": bool(done),
                "template_sha256": signature["sha256"] if signature else None,
                "capture": self._capture_mode,
            }
        )
        self._active_step = None
        self._step_count += 1

    def close(
        self,
        *,
        success: bool,
        policy_steps: int,
        fault: dict[str, Any] | None,
        monitor_timeline: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("rollout evidence is already closed")
        if self._active_step is not None:
            raise RuntimeError("cannot close rollout evidence during an active step")
        if self._step_count != policy_steps:
            raise RuntimeError(
                f"recorded {self._step_count} evidence steps for {policy_steps} rollout steps"
            )
        if len(self._monitor_inputs) != policy_steps:
            raise RuntimeError("monitor evidence does not cover every rollout step")
        if self._model_inputs:
            raise RuntimeError("processor outputs remain unmatched to policy calls")
        with self.recorder.scope(policy_step=max(0, policy_steps - 1)):
            if not self._fault_recorded:
                self.recorder.mark(
                    "rollout.fault",
                    outputs=False,
                    basis="protocol:rollout-evidence-v1:no-intervention-applied",
                    region="fault_intervention",
                    role="fault",
                    lifetime="temporal",
                    disposition="not_applicable_clean_rollout",
                    details={"kind": "none"},
                )
            self.recorder.mark(
                "rollout.monitor_timeline",
                inputs=self._monitor_inputs,
                outputs=monitor_timeline or [],
                basis="protocol:rollout-evidence-v1:monitor-results-attached-after-scoring",
                region="monitor_timeline",
                role="sink",
                lifetime="temporal",
                details={"available": monitor_timeline is not None},
            )
            self.recorder.mark(
                "rollout.outcome",
                inputs={
                    "reward": [
                        self.recorder.lineage(
                            f"policy_step:{item['policy_step']}:reward", item["reward"]
                        )
                        for item in self.timeline
                    ],
                    "done": [
                        self.recorder.lineage(
                            f"policy_step:{item['policy_step']}:done", item["done"]
                        )
                        for item in self.timeline
                    ],
                    "terminal_observation": (
                        self._next_observations[-1]
                        if self._next_observations
                        else None
                    ),
                },
                outputs={"success": bool(success), "policy_steps": policy_steps},
                basis="code:embodied-silent-failures:run_openvla._run_trial:record-libero-terminal-outcome",
                region="task_outcome",
                role="sink",
                lifetime="temporal",
                details={"success": bool(success), "policy_steps": policy_steps},
            )
        self.recorder.close(completed=True)
        annotations = [
            *self.recorder.annotations,
            *openvla_annotations(self.recorder.events),
        ]
        graph = reduce_graph(self.recorder.events, annotations)
        contracts = [
            *openvla_contract_issues(self.recorder.events),
            *torch_trace_contract_issues(
                self.recorder.events,
                ("policy",) if self._operator_traced_steps else (),
            ),
        ]
        missing_traces = sorted(self.traced_steps - self._operator_traced_steps)
        if missing_traces:
            contracts.append(
                f"requested policy steps were not operator-traced: {missing_traces}"
            )
        audit = audit_graph(
            self.recorder.events,
            annotations,
            graph,
            required_endpoints=ROLLOUT_ENDPOINTS,
            repeated_endpoints=OPENVLA_ENDPOINTS + SAFE_EVIDENCE_ENDPOINTS,
            contract_issues=contracts,
        )
        write_json_atomic(
            self.output_dir / "annotations.json",
            {"schema_version": 2, "annotations": annotations},
        )
        write_json_atomic(self.output_dir / "graph.json", graph)
        write_json_atomic(self.output_dir / "audit.json", audit)
        if not audit["passed"]:
            write_json_atomic(
                self.output_dir / "incomplete.json",
                {
                    "schema_version": 1,
                    "status": "incomplete",
                    "reason": "evidence_graph_audit_failed",
                },
            )
            self._closed = True
            raise RuntimeError(
                f"evidence graph audit failed; inspect {self.output_dir / 'audit.json'}"
            )
        composition = {
            "schema_version": 1,
            "timeline": self.timeline,
            "templates": sorted(self.templates),
            "template_count": len(self.templates),
            "requested_trace_steps": sorted(self.traced_steps),
            "traced_steps": sorted(self._operator_traced_steps),
            "policy_steps": policy_steps,
            "success": bool(success),
            "fault": fault,
            "monitor_timeline_recorded_inline": monitor_timeline is not None,
        }
        write_json_atomic(self.output_dir / "composition.json", composition)
        for digest, template in self.templates.items():
            write_json_atomic(
                self.output_dir / f"template-{digest}.json",
                {
                    "schema_version": template["schema_version"],
                    "sha256": template["sha256"],
                    "event_count": template["event_count"],
                    "event_kind_counts": template["event_kind_counts"],
                    "first_event_id": template["first_event_id"],
                    "last_event_id": template["last_event_id"],
                    "source": "raw.jsonl",
                },
            )
        self._closed = True
        return {
            "directory": str(self.output_dir),
            "audit_passed": audit["passed"],
            "template_count": len(self.templates),
            "traced_steps": sorted(self._operator_traced_steps),
        }

    def abort(self, reason: str) -> None:
        if self._closed:
            return
        self.recorder.close(completed=False)
        self._closed = True
        write_json_atomic(
            self.output_dir / "incomplete.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "reason": reason,
                "active_policy_step": self._active_step,
                "completed_policy_steps": self._step_count,
            },
        )

    def _record_model_input(self, policy_step: int, value: Any) -> None:
        if policy_step in self._model_inputs:
            raise RuntimeError("processor produced more than one model input in a step")
        self._model_inputs[policy_step] = value


class _RecordingPolicy:
    def __init__(
        self, evidence: RolloutEvidence, model: Any, policy_step: int
    ) -> None:
        self._evidence = evidence
        self._model = model
        self._policy_step = policy_step

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict_action(self, *args: Any, **kwargs: Any) -> Any:
        self._evidence._operator_traced_steps.add(self._policy_step)
        self._evidence._capture_mode = "full_operator_trace"
        with self._evidence.recorder.scope(
            policy_step=self._policy_step
        ), capture_policy(self._evidence.recorder, self._model):
            return self._model.predict_action(*args, **kwargs)


def structural_signature(events: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(b"[")
    event_count = 0
    kind_counts: dict[str, int] = {}
    for event in events:
        if event["kind"] not in {
            "boundary",
            "module",
            "opaque",
            "operator",
            "source",
            "state",
        }:
            continue
        item = {
            "kind": event["kind"],
            "name": event["name"],
            "context": {
                key: value
                for key, value in event.get("context", {}).items()
                if key != "policy_step"
            },
            "inputs": [_reference_shape(value) for value in event.get("inputs", [])],
            "outputs": [_reference_shape(value) for value in event.get("outputs", [])],
            "operator_semantics": event.get("details", {}).get(
                "operator_semantics"
            ),
            "module_path": event.get("details", {}).get("module_path"),
            "module_call_index": event.get("details", {}).get(
                "module_call_index"
            ),
        }
        if event_count:
            digest.update(b",")
        digest.update(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        event_count += 1
        kind_counts[event["kind"]] = kind_counts.get(event["kind"], 0) + 1
    digest.update(b"]")
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "event_count": event_count,
        "event_kind_counts": dict(sorted(kind_counts.items())),
        "first_event_id": events[0]["event_id"] if events else None,
        "last_event_id": events[-1]["event_id"] if events else None,
    }


def _reference_shape(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        key: reference[key]
        for key in ("port", "type", "shape", "dtype", "device")
        if key in reference
    }


def _plain_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
