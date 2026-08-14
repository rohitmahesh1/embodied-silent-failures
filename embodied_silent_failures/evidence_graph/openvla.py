from contextlib import contextmanager
from typing import Any

from embodied_silent_failures.evidence_graph.record import Recorder
from embodied_silent_failures.evidence_graph.torch_trace import (
    capture_torch_operations,
)


OPENVLA_REVISION = "300dce26d44f407c725695d16cd445755c92cbd1"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
OPENVLA_PAPER = "paper:openvla-v2-arxiv-2406.09246:fig2"
REQUIRED_ENDPOINTS = (
    "libero.current_image",
    "policy.selected_image",
    "openvla.processor_output",
    "openvla.raw_action",
    "libero.executed_command",
    "libero.next_observation",
)


# OpenVLA 300dce2, libero_utils.get_libero_image: read agentview_image,
# rotate it 180 degrees, and resize it to the policy's requested resolution.
IMAGE_BASIS = (
    "code:openvla@300dce2:experiments.robot.libero.libero_utils."
    "get_libero_image:rotate-agent-view-180-and-resize"
)

# OpenVLA 300dce2, openvla_utils.get_vla_action: center-crop the selected
# image, construct the task prompt, and pass both through the pinned processor.
PROCESSOR_BASIS = (
    "code:openvla@300dce2:experiments.robot.openvla_utils."
    "get_vla_action:preprocess-selected-image-and-task-prompt"
)

# OpenVLA 300dce2, OpenVLAForActionPrediction.predict_action: generate one
# token per action dimension, map token IDs to bin centers, then unnormalize.
ACTION_DECODE_BASIS = (
    "code:openvla@300dce2:prismatic.extern.hf.modeling_prismatic."
    "OpenVLAForActionPrediction.predict_action:decode-and-unnormalize-action-tokens"
)

# OpenVLA 300dce2, PrismaticVisionBackbone.__init__: replace both TIMM
# featurizer forwards with get_intermediate_layers at the second-to-last block.
VISION_BASIS = (
    "code:openvla@300dce2:prismatic.extern.hf.modeling_prismatic."
    "PrismaticVisionBackbone.__init__:return-second-to-last-vision-block-features"
)

# OpenVLA 300dce2, robot_utils.normalize_gripper_action and
# invert_gripper_action: map the final action coordinate to a binarized LIBERO
# command and reverse its sign; the other six coordinates are unchanged.
COMMAND_BASIS = (
    "code:openvla@300dce2:experiments.robot.robot_utils."
    "normalize_gripper_action+invert_gripper_action:convert-openvla-gripper-"
    "coordinate-to-libero-command"
)

# LIBERO 8f1084e, ControlEnv.step: forward the seven-coordinate command to the
# wrapped simulator and return its next observation, reward, done flag and info.
ENVIRONMENT_STEP_BASIS = (
    "code:libero@8f1084e:libero.libero.envs.env_wrapper.ControlEnv.step:"
    "forward-command-to-simulator-and-return-next-observation"
)

# openvla_rollout.run_trial sets the pinned task state, then advances the simulator
# with the configured dummy command before the first policy observation.
INITIAL_OBSERVATION_BASIS = (
    "code:embodied-silent-failures:openvla_rollout.run_trial:"
    "set-pinned-initial-state-and-apply-configured-wait-steps"
)

# After the first policy step, openvla_rollout.run_trial reuses the observation
# returned by the preceding pinned LIBERO ControlEnv.step call.
ROLLOUT_OBSERVATION_BASIS = (
    "code:libero@8f1084e:libero.libero.envs.env_wrapper.ControlEnv.step:"
    "return-observation-after-prior-policy-command"
)


class RecordingProcessor:
    """Observe the processor used by the pinned OpenVLA rollout call."""

    def __init__(
        self,
        recorder: Recorder,
        processor: Any,
        policy_image: Any,
        task_description: str,
        policy_step: int,
        output_observer: Any | None = None,
    ) -> None:
        self._recorder = recorder
        self._processor = processor
        self._policy_image = policy_image
        self._task_description = task_description
        self._policy_step = policy_step
        self._output_observer = output_observer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, prompt: str, image: Any, *args: Any, **kwargs: Any) -> Any:
        lineage_prompt = self._recorder.lineage(
            f"policy_step:{self._policy_step}:prompt", prompt
        )
        with self._recorder.scope(policy_step=self._policy_step):
            self._recorder.mark(
                "openvla.preprocessed_input",
                inputs={
                    "image": self._recorder.lineage(
                        f"policy_step:{self._policy_step}:selected_image",
                        self._policy_image,
                    ),
                    "instruction": self._task_description,
                },
                outputs={"image": image, "prompt": lineage_prompt},
                basis=PROCESSOR_BASIS,
                region="policy_preprocessing",
                fault_interface="processed_policy_input",
            )
        return RecordingProcessorOutput(
            self._recorder,
            self._processor(prompt, image, *args, **kwargs),
            image,
            lineage_prompt,
            self._policy_step,
            self._output_observer,
        )


class RecordingProcessorOutput:
    def __init__(
        self,
        recorder: Recorder,
        value: Any,
        image: Any,
        prompt: Any,
        policy_step: int,
        output_observer: Any | None,
    ) -> None:
        self._recorder = recorder
        self._value = value
        self._image = image
        self._prompt = prompt
        self._policy_step = policy_step
        self._output_observer = output_observer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def to(self, *args: Any, **kwargs: Any) -> Any:
        result = self._value.to(*args, **kwargs)
        with self._recorder.scope(policy_step=self._policy_step):
            self._recorder.mark(
                "openvla.processor_output",
                inputs={"image": self._image, "prompt": self._prompt},
                outputs=result,
                basis=PROCESSOR_BASIS,
                region="policy_preprocessing",
                fault_interface="model_input",
            )
        if self._output_observer is not None:
            self._output_observer(result)
        return result


@contextmanager
def capture_policy(recorder: Recorder, model: Any):
    with recorder.scope(phase="policy"), capture_torch_operations(
        recorder, {"policy": model}
    ):
        yield


def record_current_image(
    recorder: Recorder,
    observation: dict[str, Any],
    image: Any,
    *,
    disposition: str | None = None,
) -> None:
    policy_step = _policy_step(recorder)
    recorder.mark(
        "libero.current_image",
        inputs=observation["agentview_image"],
        outputs=recorder.lineage(f"policy_step:{policy_step}:current_image", image),
        basis=IMAGE_BASIS,
        region="observation_pipeline",
        fault_interface="observation_image",
        disposition=disposition,
    )


def record_current_observation(
    recorder: Recorder,
    observation: dict[str, Any],
    *,
    initial: bool = True,
    disposition: str | None = None,
) -> None:
    basis = INITIAL_OBSERVATION_BASIS if initial else ROLLOUT_OBSERVATION_BASIS
    if initial:
        recorder.source(
            "libero.current_observation",
            observation,
            basis=basis,
            region="environment",
            fault_interface="environment_observation",
            disposition=disposition,
        )
    else:
        recorder.mark(
            "libero.current_observation",
            inputs=observation,
            outputs=observation,
            basis=basis,
            region="environment",
            fault_interface="environment_observation",
            disposition=disposition,
        )


def record_policy_image(
    recorder: Recorder,
    current_image: Any,
    policy_image: Any,
    *,
    policy_step: int,
    source_step: int,
) -> None:
    current = recorder.lineage(
        f"policy_step:{policy_step}:current_image", current_image
    )
    selected_input = current
    if source_step != policy_step:
        prior = recorder.lineage(
            f"policy_step:{policy_step}:prior_image", policy_image
        )
        recorder.mark(
            "policy.prior_image",
            inputs=recorder.lineage(
                f"policy_step:{source_step}:current_image", policy_image
            ),
            outputs=prior,
            basis="protocol:stale-image-v1:buffered-prior-policy-image",
            region="policy_input_history",
            lifetime="temporal",
            fault_interface="prior_policy_image_buffer",
            details={
                "policy_step": policy_step,
                "source_step": source_step,
                "lag_policy_steps": policy_step - source_step,
                "temporal_relation": "buffer_history",
            },
        )
        selected_input = prior
    recorder.mark(
        "policy.selected_image",
        inputs=selected_input,
        outputs=recorder.lineage(
            f"policy_step:{policy_step}:selected_image", policy_image
        ),
        basis="protocol:evidence-graph-v1:selected-policy-image",
        region="policy_input_buffer",
        fault_interface="policy_image_buffer",
        details={"policy_step": policy_step, "source_step": source_step},
    )


def record_policy_outputs(
    recorder: Recorder,
    model: Any,
    generated: Any,
    action: Any,
    unnorm_key: str,
    *,
    action_sink: bool = True,
) -> None:
    policy_step = _policy_step(recorder)
    action_tokens = generated["sequences"][:, -model.get_action_dim(unnorm_key) :]
    recorder.mark(
        "openvla.action_tokens",
        inputs=generated["sequences"],
        outputs=action_tokens,
        basis=[OPENVLA_PAPER, ACTION_DECODE_BASIS],
        region="action_token_generation",
        fault_interface="action_tokens",
        details={
            "generated_positions": list(range(int(action_tokens.shape[-1]))),
            "generation_mode": "autoregressive",
        },
    )
    recorder.mark(
        "openvla.raw_action",
        inputs=action_tokens,
        outputs=recorder.lineage(f"policy_step:{policy_step}:raw_action", action),
        basis=[OPENVLA_PAPER, ACTION_DECODE_BASIS],
        region="action_decoding",
        role="sink" if action_sink else None,
        fault_interface="raw_action",
    )


def executed_command(
    recorder: Recorder,
    runtime: Any,
    raw_action: Any,
    *,
    command_sink: bool = True,
) -> Any:
    policy_step = _policy_step(recorder)
    command = runtime.normalize_gripper_action(raw_action.copy(), binarize=True)
    command = runtime.invert_gripper_action(command)
    recorder.mark(
        "libero.executed_command",
        inputs=recorder.lineage(f"policy_step:{policy_step}:raw_action", raw_action),
        outputs=recorder.lineage(
            f"policy_step:{policy_step}:executed_command", command
        ),
        basis=COMMAND_BASIS,
        region="command_conversion",
        role="sink" if command_sink else None,
        fault_interface="executed_command",
    )
    return command


def step_environment(
    recorder: Recorder,
    env: Any,
    command: Any,
    *,
    next_observation_sink: bool = True,
    next_observation_disposition: str | None = None,
) -> tuple[Any, ...]:
    policy_step = _policy_step(recorder)
    observation, reward, done, info = env.step(command.tolist())
    recorder.mark(
        "libero.environment_step",
        kind="opaque",
        inputs=recorder.lineage(
            f"policy_step:{policy_step}:executed_command", command
        ),
        outputs={
            "observation": observation,
            "reward": recorder.lineage(
                f"policy_step:{policy_step}:reward", reward
            ),
            "done": recorder.lineage(
                f"policy_step:{policy_step}:done", bool(done)
            ),
        },
        basis=ENVIRONMENT_STEP_BASIS,
        region="environment",
        lifetime="temporal",
        fault_interface="simulator_command",
        details={
            "lag_policy_steps": 1,
            "temporal_relation": "world_feedback",
            "opaque_reason": "MuJoCo simulator call",
        },
    )
    recorder.mark(
        "libero.next_observation",
        inputs=observation,
        outputs=observation,
        basis=ENVIRONMENT_STEP_BASIS,
        region="environment",
        role="sink" if next_observation_sink else None,
        lifetime="temporal",
        disposition=next_observation_disposition,
    )
    return observation, reward, done, info


def _policy_step(recorder: Recorder) -> int:
    try:
        return int(recorder.context("policy_step"))
    except KeyError as error:
        raise RuntimeError(
            "evidence boundary was recorded outside a policy-step scope"
        ) from error


def operator_annotations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations = []
    for event in events:
        if event["kind"] not in {"module", "operator", "state"}:
            continue
        phase = event.get("context", {}).get("phase")
        name = event["name"]
        details = event.get("details", {})
        scopes = details.get("module_scope", [])
        scope = details.get("module_path") or (scopes[-1] if scopes else "")
        registrations = details.get("registrations", [])
        registration_path = registrations[0]["module_path"] if registrations else None
        if registration_path:
            if details.get("root") != "policy":
                continue
            scope = registration_path
        elif phase != "policy":
            continue

        registration_regions = {
            _region_name(item["module_path"], name)[0] for item in registrations
        }
        if len(registration_regions) > 1:
            region = "shared_policy_state"
            basis = [
                OPENVLA_PAPER,
                "observed:torch-model-state:multiple-registration-paths",
            ]
        else:
            region, basis = _region_name(scope, name)
        semantic_key = _semantic_key(region, scope, name, details)
        disposition = _disposition(scope)
        annotations.append(
            {
                "event_id": event["event_id"],
                "region": region,
                "basis": basis,
                "lifetime": "step",
                "semantic_key": semantic_key,
                **(
                    {"fault_interface": "registered_model_state"}
                    if registration_path
                    else {}
                ),
                **({"disposition": disposition} if disposition else {}),
            }
        )
    return annotations


def _region_name(scope: str, name: str) -> tuple[str, list[str]]:
    if ".vision_backbone" in scope or ".vision_backbone" in name:
        region = "vision_encoder"
        basis = [
            OPENVLA_PAPER,
            VISION_BASIS,
            "observed:torch-module:policy.vision_backbone",
        ]
    elif ".projector" in scope or ".projector" in name:
        region = "multimodal_projector"
        basis = [OPENVLA_PAPER, "observed:torch-module:policy.projector"]
    elif ".lm_head" in scope or ".lm_head" in name:
        region = "action_token_logits"
        basis = [OPENVLA_PAPER, "observed:torch-module:policy.language_model.lm_head"]
    elif ".language_model" in scope or ".language_model" in name:
        region = "language_backbone"
        basis = [OPENVLA_PAPER, "observed:torch-module:policy.language_model"]
    else:
        region = "policy_generation_control"
        basis = [ACTION_DECODE_BASIS, "observed:torch-dispatch-outside-module"]
    return region, basis


def _semantic_key(
    region: str, scope: str, name: str, details: dict[str, Any]
) -> str:
    call_index = details.get("module_call_index")
    if call_index is None:
        calls = details.get("module_calls", [])
        if calls:
            call_index = calls[-1].get("call_index")
    if details.get("registrations"):
        paths = sorted(
            item["module_path"]
            for item in details["registrations"]
        )
        return f"{region}/state/{'+'.join(paths)}"
    if region in {"language_backbone", "action_token_logits"} and call_index is not None:
        return f"{region}/action_token_{int(call_index)}"
    return region


def _disposition(scope: str) -> str | None:
    discarded = (
        ".vision_backbone.featurizer.blocks.23",
        ".vision_backbone.fused_featurizer.blocks.26",
    )
    if any(marker in scope for marker in discarded):
        return "intentionally_discarded_final_vision_block"
    return None


def contract_issues(events: list[dict[str, Any]]) -> list[str]:
    action_events = [event for event in events if event["name"] == "openvla.action_tokens"]
    if not action_events:
        return ["OpenVLA action-token boundary is missing"]
    issues = [
        f"OpenVLA generated action-token positions are {positions}, expected 0-6"
        for event in action_events
        if (positions := event.get("details", {}).get("generated_positions"))
        != list(range(7))
    ]
    lm_head_calls = sorted(
        {
            int(event["details"]["module_call_index"])
            for event in events
            if event.get("kind") == "module"
            and event.get("details", {}).get("module_path", "").endswith(".lm_head")
            and "module_call_index" in event.get("details", {})
        }
    )
    if lm_head_calls and lm_head_calls != list(range(7)):
        issues.append(
            f"OpenVLA lm_head call positions are {lm_head_calls}, expected 0-6"
        )
    return issues
