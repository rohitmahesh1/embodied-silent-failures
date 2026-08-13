import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from embodied_silent_failures.evidence_graph.record import LineageValue, Recorder


QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
HIDE_AND_SEEK_PAPER = "paper:hide-and-seek-arxiv-2605.30834:appendix-g.5"
REQUIRED_ENDPOINTS = (
    "qwen.monitor_input",
    "qwen.raw_response",
    "qwen.alarm",
)

SYSTEM_PROMPT = "You are a robot monitoring assistant."
USER_PROMPT = (
    "Goal: Determine whether the robot is currently in a failure state at the "
    "latest frame.\n"
    "Context: You are given a chronological sequence of frames ending at the "
    "current timestep (up to {history_frames} frames).\n"
    "Task instruction:{instruction}\n"
    "Question:At the last frame (current timestep), is the robot in a failure state?\n"
    "Output strictly JSON with keys: failure_now (0 or 1), reason (short string)."
)


# Hide-and-Seek Appendix G.5 supplies Qwen3-VL-8B-Instruct, the chronological
# prefix, the current-failure question, and the two JSON fields. It does not
# supply H, frame spacing, query cadence, processor media encoding, or decoding
# settings; score_qwen.py records those as experiment protocol choices.
PROMPT_BASIS = HIDE_AND_SEEK_PAPER
HISTORY_BASIS = (
    "protocol:qwen-observation-monitor-v1:fixed-backward-frame-window-"
    "ending-at-queried-policy-step"
)

# Hugging Face Transformers, AutoProcessor.apply_chat_template at the recorded
# package version and source-file hash, receives each already selected frame as
# one ordered image item. This avoids an undocumented second sampling pass.
MEDIA_BASIS = (
    "protocol:qwen-observation-monitor-v1:ordered-image-items-without-"
    "processor-video-resampling"
)

# Hide-and-Seek Appendix G.5 requires JSON with exactly failure_now and reason.
# It does not describe extraction from prose or code fences, so malformed output
# remains invalid evidence rather than being repaired by an experimenter parser.
PARSER_BASIS = (
    "protocol:qwen-observation-monitor-v1:strict-two-field-json-without-repair"
)


@dataclass(frozen=True)
class QwenRequest:
    current_step: int
    frame_steps: tuple[int, ...]
    frame_sha256: tuple[str, ...]
    prompt_sha256: str


@dataclass(frozen=True)
class QwenDecision:
    failure_now: int
    reason: str

    @property
    def alarm(self) -> bool:
        return bool(self.failure_now)


def selected_frame_steps(
    current_step: int, history_frames: int, history_stride: int
) -> tuple[int, ...]:
    if current_step < 0:
        raise ValueError("current step must be nonnegative")
    if history_frames <= 0:
        raise ValueError("history frame count must be positive")
    if history_stride <= 0:
        raise ValueError("history stride must be positive")
    newest_first = range(current_step, -1, -history_stride)
    return tuple(reversed(tuple(newest_first)[:history_frames]))


def query_steps(policy_steps: int, query_stride: int) -> tuple[int, ...]:
    if policy_steps <= 0:
        raise ValueError("policy step count must be positive")
    if query_stride <= 0:
        raise ValueError("query stride must be positive")
    return tuple(range(0, policy_steps, query_stride))


def prompt_text(instruction: str, history_frames: int) -> str:
    if not instruction.strip():
        raise ValueError("task instruction must be nonempty")
    if history_frames <= 0:
        raise ValueError("history frame count must be positive")
    return USER_PROMPT.format(
        history_frames=history_frames,
        instruction=instruction,
    )


def prompt_sha256(instruction: str, history_frames: int) -> str:
    value = SYSTEM_PROMPT + "\0" + prompt_text(instruction, history_frames)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_messages(
    frames: Sequence[Any], instruction: str, history_frames: int
) -> list[dict[str, Any]]:
    if not frames or len(frames) > history_frames:
        raise ValueError("frame sequence must be nonempty and no longer than H")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": frame} for frame in frames),
                {
                    "type": "text",
                    "text": prompt_text(instruction, history_frames),
                },
            ],
        },
    ]


def parse_response(raw_response: str) -> QwenDecision:
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen response is not one JSON value") from error
    if not isinstance(value, dict) or set(value) != {"failure_now", "reason"}:
        raise ValueError("Qwen response must contain exactly failure_now and reason")
    failure_now = value["failure_now"]
    if isinstance(failure_now, bool) or not isinstance(failure_now, int):
        raise ValueError("failure_now must be the integer 0 or 1")
    if failure_now not in (0, 1):
        raise ValueError("failure_now must be the integer 0 or 1")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a nonempty string")
    return QwenDecision(failure_now=failure_now, reason=reason)


def trajectory_prediction(alarms: Sequence[bool | None]) -> bool | None:
    if not alarms:
        raise ValueError("trajectory must contain at least one Qwen query")
    if any(alarm is True for alarm in alarms):
        return True
    if any(alarm is None for alarm in alarms):
        return None
    return False


def record_observation_frame(
    recorder: Recorder,
    frame: Any,
    *,
    policy_step: int,
    frame_sha256: str,
    run_sha256: str,
    video_sha256: str,
) -> None:
    # run_openvla.py::_run_trial, behavior checked at 505b350, appends the current
    # get_libero_image result before env.step. Each source run hash fixes its
    # producing revision; the RGB hash fixes the decoded, possibly lossy value.
    recorder.source(
        "qwen.observation_frame",
        frame,
        basis="observed:rollout-video:decoded-current-camera-frame",
        region="qwen_observation_evidence",
        fault_interface="qwen_observation_frame",
        details={
            "policy_step": policy_step,
            "frame_sha256": frame_sha256,
            "source_run_sha256": run_sha256,
            "source_video_sha256": video_sha256,
            "encoding_note": (
                "monitor receives the decoded rollout-video frame, not the "
                "pre-encoding simulator image"
            ),
        },
    )


def record_monitor_input(
    recorder: Recorder,
    frames: Sequence[Any],
    *,
    instruction: str,
    frame_steps: Sequence[int],
    frame_sha256: Sequence[str],
    history_frames: int,
) -> QwenRequest:
    steps = tuple(int(step) for step in frame_steps)
    hashes = tuple(str(digest) for digest in frame_sha256)
    if len(frames) != len(steps) or len(frames) != len(hashes):
        raise ValueError("frames, steps, and hashes must have equal lengths")
    if not steps or tuple(sorted(steps)) != steps or len(set(steps)) != len(steps):
        raise ValueError("frame steps must be a nonempty increasing sequence")
    if len(steps) > history_frames:
        raise ValueError("monitor input contains more than H frames")
    request = QwenRequest(
        current_step=steps[-1],
        frame_steps=steps,
        frame_sha256=hashes,
        prompt_sha256=prompt_sha256(instruction, history_frames),
    )
    recorder.mark(
        "qwen.monitor_input",
        inputs={"frames": list(frames), "instruction": instruction},
        outputs=request,
        basis=[PROMPT_BASIS, HISTORY_BASIS, MEDIA_BASIS],
        region="qwen_monitor_input",
        role="monitor_input",
        fault_interface="qwen_observation_history",
        details={
            "current_step": request.current_step,
            "frame_steps": list(request.frame_steps),
            "frame_sha256": list(request.frame_sha256),
            "history_frames": history_frames,
            "prompt_sha256": request.prompt_sha256,
        },
    )
    return request


def record_processor_output(
    recorder: Recorder,
    request: QwenRequest,
    processed: Any,
    *,
    processor_basis: str,
) -> None:
    if not processor_basis.startswith("code:"):
        raise ValueError("processor basis must identify pinned implementation code")
    recorder.mark(
        "qwen.processor_output",
        inputs=request,
        outputs=processed,
        basis=[HIDE_AND_SEEK_PAPER, MEDIA_BASIS, processor_basis],
        region="qwen_processor",
        fault_interface="qwen_processor_output",
    )


def record_traced_response(
    recorder: Recorder,
    generated: Any,
    generated_token_ids: Sequence[int],
    raw_response: str,
    *,
    model_basis: str,
) -> LineageValue:
    if not model_basis.startswith("code:"):
        raise ValueError("model basis must identify pinned implementation code")
    response = recorder.lineage("qwen:traced:raw_response", raw_response)
    recorder.mark(
        "qwen.raw_response",
        inputs={"generated": generated, "token_ids": list(generated_token_ids)},
        outputs=response,
        basis=[HIDE_AND_SEEK_PAPER, model_basis],
        region="qwen_response_decode",
        fault_interface="qwen_response_decode",
    )
    return response


def record_model_response(
    recorder: Recorder,
    request: QwenRequest,
    raw_response: str,
    *,
    model_basis: str,
) -> LineageValue:
    if not model_basis.startswith("code:"):
        raise ValueError("model basis must identify pinned implementation code")
    response = recorder.lineage(
        f"qwen:policy_step:{request.current_step}:raw_response", raw_response
    )
    recorder.mark(
        "qwen.raw_response",
        kind="opaque",
        inputs=request,
        outputs=response,
        basis=[HIDE_AND_SEEK_PAPER, model_basis],
        region="qwen_private_compute",
        fault_interface="qwen_private_compute",
        details={
            "current_step": request.current_step,
            "opaque_reason": (
                "Qwen internals are outside the current evidence-path fault scope; "
                "the model snapshot and processor implementation are recorded instead"
            ),
        },
    )
    return response


def internal_annotations(
    events: list[dict[str, Any]], *, model_basis: str
) -> list[dict[str, Any]]:
    if not model_basis.startswith("code:"):
        raise ValueError("model basis must identify pinned implementation code")
    annotations = []
    for event in events:
        if event["kind"] not in {"module", "operator", "state"}:
            continue
        details = event.get("details", {})
        registrations = details.get("registrations", [])
        is_model_state = details.get("root") == "qwen_model" and bool(registrations)
        if event.get("context", {}).get("phase") != "qwen_model" and not is_model_state:
            continue

        calls = details.get("module_calls", [])
        module_path = details.get("module_path") or (
            calls[-1].get("path") if calls else ""
        )
        if is_model_state:
            paths = sorted(item["module_path"] for item in registrations)
            owners = {_module_owner(path) for path in paths}
            region = owners.pop() if len(owners) == 1 else "qwen_model_shared_state"
            semantic_key = f"qwen_model/state/{'+'.join(paths)}"
            observed_basis = "observed:torch-model-state:" + "+".join(paths)
            fault_interface = "registered_qwen_model_state"
        else:
            region = _module_owner(module_path)
            semantic_key = f"qwen_model/module/{module_path or event['name']}"
            observed_basis = f"observed:torch-module:{module_path or event['name']}"
            fault_interface = "qwen_internal_compute"
        annotations.append(
            {
                "event_id": event["event_id"],
                "region": region,
                "semantic_key": semantic_key,
                "basis": [HIDE_AND_SEEK_PAPER, model_basis, observed_basis],
                "lifetime": "step",
                "fault_interface": fault_interface,
            }
        )
    return annotations


def _module_owner(module_path: str) -> str:
    parts = module_path.split(".")
    if not parts or parts[0] != "qwen_model":
        return "qwen_model_unscoped"
    if len(parts) == 1:
        return "qwen_model_root"
    if parts[1] == "model" and len(parts) >= 3:
        return f"qwen_model_{parts[2]}"
    return f"qwen_model_{parts[1]}"


def record_decision(
    recorder: Recorder,
    raw_response: LineageValue,
    decision: QwenDecision,
    *,
    policy_step: int,
) -> bool:
    recorder.mark(
        "qwen.parsed_response",
        inputs=raw_response,
        outputs=decision,
        basis=PARSER_BASIS,
        region="qwen_response_parser",
        fault_interface="qwen_response_parser",
        details={
            "policy_step": policy_step,
            "failure_now": decision.failure_now,
            "reason": decision.reason,
        },
    )
    recorder.mark(
        "qwen.alarm",
        inputs=decision,
        outputs=decision.alarm,
        basis=[HIDE_AND_SEEK_PAPER, PARSER_BASIS],
        region="qwen_alarm",
        role="sink",
        details={"policy_step": policy_step, "alarm": decision.alarm},
    )
    return decision.alarm


def contract_issues(events: list[dict[str, Any]]) -> list[str]:
    issues = []
    inputs = [event for event in events if event["name"] == "qwen.monitor_input"]
    for event in inputs:
        details = event.get("details", {})
        steps = details.get("frame_steps", [])
        if not steps or steps != sorted(set(steps)):
            issues.append("Qwen frame steps are not a nonempty increasing sequence")
        elif details.get("current_step") != steps[-1]:
            issues.append("Qwen evidence does not end at the queried policy step")
        if len(steps) > int(details.get("history_frames", 0)):
            issues.append("Qwen evidence exceeds the declared history length")
    return issues
