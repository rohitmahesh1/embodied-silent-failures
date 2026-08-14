import copy
import hashlib
import re
from typing import Any

from embodied_silent_failures.evidence_graph.openvla import IMAGE_BASIS
from embodied_silent_failures.evidence_graph.qwen import (
    HIDE_AND_SEEK_PAPER,
    HISTORY_BASIS,
    MEDIA_BASIS,
    PARSER_BASIS,
)


ROLLOUT_ENDPOINTS = (
    "rollout.fault",
    "rollout.monitor_timeline",
    "rollout.outcome",
)
QWEN_EVIDENCE_ENDPOINTS = (
    "qwen.observation_frame",
    "qwen.monitor_input",
    "qwen.raw_response",
    "qwen.alarm",
)

# run_openvla.py::_run_trial at the source run's recorded experiment commit
# appends the get_libero_image result for each policy step; pinned OpenVLA's
# save_rollout_video_given_path encodes that ordered list. score_qwen.py then
# decodes the same MP4 and records a hash of every RGB frame supplied to Qwen.
VIDEO_FRAME_BASIS = (
    "protocol:qwen-rollout-evidence-v1:recorded-policy-image-then-lossy-"
    "video-encode-and-rgb-decode"
)
TIMELINE_BASIS = (
    "protocol:qwen-rollout-evidence-v1:all-frozen-query-alarms-in-policy-"
    "step-order"
)


def compose_qwen_rollout(
    events: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    trial: dict[str, Any],
    qwen_run: dict[str, Any],
    source_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace SAFE evidence with the frozen Qwen timeline on one rollout."""
    if not events or events[0].get("kind") != "trace_start":
        raise ValueError("source rollout trace has no trace_start")
    if events[-1].get("kind") != "trace_end":
        raise ValueError("source rollout trace has no trace_end")
    if events[-1].get("details", {}).get("completed") is not True:
        raise ValueError("source rollout trace is incomplete")

    removed = {
        event["event_id"]
        for event in events
        if event.get("name", "").startswith("safe.")
        or event.get("name") == "rollout.monitor_timeline"
    }
    retained = [
        copy.deepcopy(event)
        for event in events[1:-1]
        if event["event_id"] not in removed
    ]
    retained_annotations = [
        copy.deepcopy(item)
        for item in annotations
        if item.get("event_id") not in removed
    ]

    next_event = _next_identifier(events, "event_id", "e")
    next_value = _next_reference_identifier(events)

    def value_id() -> str:
        nonlocal next_value
        result = f"v{next_value:08d}"
        next_value += 1
        return result

    def append(
        name: str,
        *,
        kind: str = "boundary",
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        policy_step: int | None = None,
        details: dict[str, Any] | None = None,
        region: str,
        basis: list[str],
        role: str | None = None,
        lifetime: str = "step",
        fault_interface: str | None = None,
    ) -> dict[str, Any]:
        nonlocal next_event
        event = {
            "event_id": f"e{next_event:08d}",
            "kind": kind,
            "name": name,
        }
        next_event += 1
        if inputs:
            event["inputs"] = inputs
        if outputs:
            event["outputs"] = outputs
        if policy_step is not None:
            event["context"] = {"policy_step": policy_step}
        if details:
            event["details"] = details
        retained.append(event)
        annotation = {
            "event_id": event["event_id"],
            "region": region,
            "basis": basis,
            "lifetime": lifetime,
        }
        if role is not None:
            annotation["role"] = role
        if fault_interface is not None:
            annotation["fault_interface"] = fault_interface
        retained_annotations.append(annotation)
        return event

    observations = _current_observations(retained)
    protocol = qwen_run["configuration"]["protocol"]
    model_basis = _model_basis(qwen_run)
    frame_outputs: dict[int, dict[str, Any]] = {}
    alarm_outputs = []

    timeline = trial.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("Qwen trial has no query timeline")
    query_steps = [int(query["policy_step"]) for query in timeline]
    if query_steps != sorted(set(query_steps)):
        raise ValueError("Qwen queries are not in unique policy-step order")

    for query in timeline:
        policy_step = int(query["policy_step"])
        frame_steps = [int(step) for step in query.get("frame_steps", [])]
        frame_hashes = [str(value) for value in query.get("frame_sha256", [])]
        if len(frame_steps) != len(frame_hashes) or not frame_steps:
            raise ValueError(f"Qwen query {policy_step} has invalid frame evidence")
        if query.get("parse_error") is not None or query.get("alarm") is None:
            raise ValueError(f"Qwen query {policy_step} has no valid alarm")

        for frame_step, frame_hash in zip(frame_steps, frame_hashes, strict=True):
            existing = frame_outputs.get(frame_step)
            if existing is not None:
                if existing["frame_sha256"] != frame_hash:
                    raise ValueError(f"Qwen frame hash changed at policy step {frame_step}")
                continue
            source = observations.get(frame_step)
            if source is None:
                raise ValueError(
                    f"rollout has no current camera observation at policy step {frame_step}"
                )
            frame_reference = {
                "port": "value",
                "value_id": value_id(),
                "type": "numpy.ndarray",
                "shape": [
                    int(trial["video_metadata"]["height"]),
                    int(trial["video_metadata"]["width"]),
                    3,
                ],
                "dtype": "uint8",
            }
            append(
                "qwen.observation_frame",
                inputs=[{**source, "port": "value.source_agentview_image"}],
                outputs=[frame_reference],
                policy_step=frame_step,
                details={
                    "policy_step": frame_step,
                    "frame_sha256": frame_hash,
                    "source_video_sha256": trial["video_sha256"],
                    "source_experiment_revision": source_revision,
                    "encoding_note": (
                        "This value is the decoded rollout-video frame derived from "
                        "the recorded camera observation; lossy encoding means it is "
                        "not asserted to equal the pre-encoding image byte for byte."
                    ),
                },
                region="qwen_observation_evidence",
                basis=[IMAGE_BASIS, VIDEO_FRAME_BASIS],
                fault_interface="qwen_observation_frame",
            )
            frame_outputs[frame_step] = {
                **frame_reference,
                "frame_sha256": frame_hash,
            }

        request = {
            "port": "value",
            "value_id": value_id(),
            "type": "embodied_silent_failures.evidence_graph.qwen.QwenRequest",
        }
        append(
            "qwen.monitor_input",
            inputs=[
                {
                    **_public_reference(frame_outputs[step]),
                    "port": f"value.frames[{index}]",
                }
                for index, step in enumerate(frame_steps)
            ]
            + [
                {
                    "port": "value.instruction",
                    "value_id": value_id(),
                    "type": "builtins.str",
                    "value": trial["task_description"],
                }
            ],
            outputs=[request],
            policy_step=policy_step,
            details={
                "current_step": policy_step,
                "frame_steps": frame_steps,
                "frame_sha256": frame_hashes,
                "history_frames": int(protocol["history_frames"]),
                "prompt_sha256": query["prompt_sha256"],
            },
            region="qwen_monitor_input",
            basis=[HIDE_AND_SEEK_PAPER, HISTORY_BASIS, MEDIA_BASIS],
            role="monitor_input",
            fault_interface="qwen_observation_history",
        )

        response = {
            "port": "value",
            "value_id": value_id(),
            "type": "builtins.str",
            "value": query["raw_response"],
        }
        append(
            "qwen.raw_response",
            kind="opaque",
            inputs=[request],
            outputs=[response],
            policy_step=policy_step,
            details={
                "current_step": policy_step,
                "raw_response_sha256": hashlib.sha256(
                    query["raw_response"].encode("utf-8")
                ).hexdigest(),
                "opaque_reason": (
                    "The primary comparison treats monitor inference as opaque, "
                    "matching the published SAFE rollout graph."
                ),
            },
            region="qwen_private_compute",
            basis=[HIDE_AND_SEEK_PAPER, model_basis],
            fault_interface="qwen_private_compute",
        )
        decision = {
            "port": "value",
            "value_id": value_id(),
            "type": "embodied_silent_failures.evidence_graph.qwen.QwenDecision",
        }
        append(
            "qwen.parsed_response",
            inputs=[response],
            outputs=[decision],
            policy_step=policy_step,
            details={"policy_step": policy_step, **query["parsed_response"]},
            region="qwen_response_parser",
            basis=[PARSER_BASIS],
            fault_interface="qwen_response_parser",
        )
        alarm = {
            "port": "value",
            "value_id": value_id(),
            "type": "builtins.bool",
            "value": bool(query["alarm"]),
        }
        append(
            "qwen.alarm",
            inputs=[decision],
            outputs=[alarm],
            policy_step=policy_step,
            details={"policy_step": policy_step, "alarm": bool(query["alarm"])},
            region="qwen_alarm",
            basis=[HIDE_AND_SEEK_PAPER, PARSER_BASIS],
            fault_interface="qwen_alarm",
        )
        alarm_outputs.append(alarm)

    append(
        "rollout.monitor_timeline",
        inputs=[
            {**alarm, "port": f"value.alarms[{index}]"}
            for index, alarm in enumerate(alarm_outputs)
        ],
        policy_step=query_steps[-1],
        details={
            "monitor": "qwen3_vl_observation_monitor",
            "query_count": len(timeline),
            "query_steps": query_steps,
            "available": True,
        },
        region="monitor_timeline",
        basis=[HIDE_AND_SEEK_PAPER, TIMELINE_BASIS],
        role="sink",
        lifetime="temporal",
    )

    start = copy.deepcopy(events[0])
    start.setdefault("details", {})["composed_monitor"] = "qwen3_vl"
    end = copy.deepcopy(events[-1])
    end["event_id"] = f"e{next_event:08d}"
    return [start, *retained, end], retained_annotations


def _current_observations(
    events: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    result = {}
    for event in events:
        if event.get("name") != "libero.current_observation":
            continue
        step = int(event.get("context", {}).get("policy_step", -1))
        matches = [
            reference
            for reference in event.get("outputs", [])
            if reference.get("port") == "value.agentview_image"
        ]
        if step < 0 or len(matches) != 1 or step in result:
            raise ValueError("rollout current-observation boundaries are ambiguous")
        result[step] = _public_reference(matches[0])
    return result


def _model_basis(run: dict[str, Any]) -> str:
    revision = str(run["configuration"]["model"]["revision"])
    implementation = run["runtime"]["model_implementation"]
    digest = str(implementation["sha256"])
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Qwen model revision is not a full commit hash")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Qwen model implementation hash is invalid")
    return (
        f"code:qwen@{revision}:{implementation['class']}.generate:"
        f"file-sha256-{digest}"
    )


def _public_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in reference.items()
        if key != "frame_sha256"
    }


def _next_identifier(
    events: list[dict[str, Any]], field: str, prefix: str
) -> int:
    values = []
    for event in events:
        value = event.get(field)
        if isinstance(value, str) and re.fullmatch(fr"{prefix}\d+", value):
            values.append(int(value[1:]))
    return max(values, default=-1) + 1


def _next_reference_identifier(events: list[dict[str, Any]]) -> int:
    values = []
    for event in events:
        for reference in [*event.get("inputs", []), *event.get("outputs", [])]:
            value = reference.get("value_id")
            if isinstance(value, str) and re.fullmatch(r"v\d+", value):
                values.append(int(value[1:]))
    return max(values, default=-1) + 1
