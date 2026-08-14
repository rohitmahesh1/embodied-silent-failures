import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.audit import audit_graph
from embodied_silent_failures.evidence_graph.qwen import (
    REQUIRED_ENDPOINTS,
    contract_issues,
    parse_response,
    record_decision,
    record_model_response,
    record_monitor_input,
    record_observation_frame,
)
from embodied_silent_failures.evidence_graph.record import Recorder
from embodied_silent_failures.evidence_graph.reduce import reduce_graph
from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.qwen_artifacts import (
    decode_selected_frames,
    frame_sha256,
)


CONSTRUCTION = "post_hoc_lineage_reconstruction_from_frozen_qwen_scoring_artifact"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one audited Qwen evidence path from frozen scoring output."
    )
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--query-step", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _query(trial: dict[str, Any], policy_step: int) -> dict[str, Any]:
    if trial.get("status") != "complete":
        raise ValueError("Qwen trial is not complete")
    matches = [
        item for item in trial.get("timeline", []) if item.get("policy_step") == policy_step
    ]
    if len(matches) != 1:
        raise ValueError(f"Qwen trial has {len(matches)} queries at policy step {policy_step}")
    query = matches[0]
    if query.get("parse_error") is not None or query.get("parsed_response") is None:
        raise ValueError("cannot build an alarm path from an invalid Qwen response")
    return query


def _model_basis(run: dict[str, Any]) -> str:
    revision = run["configuration"]["model"]["revision"]
    implementation = run["runtime"]["model_implementation"]
    if len(revision) != 40 or len(implementation["sha256"]) != 64:
        raise ValueError("Qwen run does not pin its model and implementation")
    # The model revision identifies the Qwen snapshot; the source hash remains
    # in composition.json because a package file hash is not a Git commit.
    return (
        f"code:qwen@{revision}:Qwen3VLForConditionalGeneration.generate:"
        "pinned-greedy-response-from-frozen-scoring-artifact"
    )


def _record_graph(
    *,
    trial: dict[str, Any],
    run: dict[str, Any],
    query: dict[str, Any],
    frames: Sequence[Any],
    output_dir: Path,
    trial_sha256: str,
    run_sha256: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Qwen evidence output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_steps = [int(value) for value in query["frame_steps"]]
    frame_hashes = [str(value) for value in query["frame_sha256"]]
    if len(frames) != len(frame_steps):
        raise ValueError("decoded frame count does not match the frozen Qwen query")
    instruction = trial.get("task_description")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Qwen trial has no task instruction")
    protocol = run["configuration"]["protocol"]
    if trial.get("configuration_sha256") != run.get("configuration_sha256"):
        raise ValueError("Qwen trial and run configuration hashes disagree")

    raw_path = output_dir / "raw.jsonl"
    metadata = {
        "schema_version": 1,
        "scope": CONSTRUCTION,
        "task_id": int(trial["task_id"]),
        "episode_index": int(trial["episode_index"]),
        "source": trial["source"],
        "policy_step": int(query["policy_step"]),
        "trial_sha256": trial_sha256,
        "qwen_run_sha256": run_sha256,
    }
    with Recorder(raw_path, metadata) as recorder:
        for frame, policy_step, digest in zip(
            frames, frame_steps, frame_hashes, strict=True
        ):
            record_observation_frame(
                recorder,
                frame,
                policy_step=policy_step,
                frame_sha256=digest,
                run_sha256=trial["run_sha256"],
                video_sha256=trial["video_sha256"],
            )
        request = record_monitor_input(
            recorder,
            frames,
            instruction=instruction,
            frame_steps=frame_steps,
            frame_sha256=frame_hashes,
            history_frames=int(protocol["history_frames"]),
        )
        if request.prompt_sha256 != query["prompt_sha256"]:
            raise ValueError("reconstructed prompt hash disagrees with Qwen scoring output")
        response = record_model_response(
            recorder,
            request,
            query["raw_response"],
            model_basis=_model_basis(run),
        )
        decision = parse_response(query["raw_response"])
        if {
            "failure_now": decision.failure_now,
            "reason": decision.reason,
        } != query["parsed_response"]:
            raise ValueError("reparsed Qwen response disagrees with scoring output")
        alarm = record_decision(
            recorder,
            response,
            decision,
            policy_step=int(query["policy_step"]),
        )
        if alarm is not query["alarm"]:
            raise ValueError("reconstructed Qwen alarm disagrees with scoring output")

    events = recorder.events
    annotations = recorder.annotations
    graph = reduce_graph(events, annotations)
    audit = audit_graph(
        events,
        annotations,
        graph,
        required_endpoints=REQUIRED_ENDPOINTS,
        contract_issues=contract_issues(events),
    )
    audit["construction"] = {
        "kind": CONSTRUCTION,
        "established": (
            "Exact frozen frame values, prompt hash, response, parser output, alarm, "
            "and declared lineage boundaries were checked."
        ),
        "not_established": (
            "Qwen internal operators were not retraced and model inference was not rerun."
        ),
    }
    if not audit["passed"]:
        raise RuntimeError("reconstructed Qwen evidence graph did not pass its audit")

    composition = {
        "schema_version": 1,
        "construction": CONSTRUCTION,
        "source": trial["source"],
        "task_id": int(trial["task_id"]),
        "episode_index": int(trial["episode_index"]),
        "policy_step": int(query["policy_step"]),
        "frame_steps": frame_steps,
        "frame_sha256": frame_hashes,
        "prompt_sha256": query["prompt_sha256"],
        "raw_response_sha256": hashlib.sha256(
            query["raw_response"].encode("utf-8")
        ).hexdigest(),
        "alarm": bool(query["alarm"]),
        "trial_sha256": trial_sha256,
        "qwen_run_sha256": run_sha256,
        "configuration_sha256": run["configuration_sha256"],
        "model": run["configuration"]["model"],
        "model_implementation": run["runtime"]["model_implementation"],
        "processor_implementation": run["runtime"]["processor_implementation"],
        "experiment_revision": run["repository_state"]["revision"],
    }
    write_json_atomic(
        output_dir / "annotations.json",
        {"schema_version": 1, "annotations": annotations},
    )
    write_json_atomic(output_dir / "graph.json", graph)
    write_json_atomic(output_dir / "audit.json", audit)
    write_json_atomic(output_dir / "composition.json", composition)
    return {
        "audit_passed": True,
        "regions": len(graph["regions"]),
        "edges": len(graph["edges"]),
        "sinks": [item["name"] for item in graph["sinks"]],
    }


def build_query_graph(
    trial_path: Path,
    run_path: Path,
    policy_step: int,
    output_dir: Path,
) -> dict[str, Any]:
    trial_path = trial_path.resolve()
    run_path = run_path.resolve()
    trial = load_json(trial_path)
    run = load_json(run_path)
    query = _query(trial, policy_step)

    import cv2
    import numpy as np

    frame_steps = {int(value) for value in query["frame_steps"]}
    decoded, _metadata = decode_selected_frames(
        Path(trial["video_path"]),
        frame_steps,
        int(trial["policy_steps"]),
        cv2=cv2,
        np=np,
    )
    frames = [decoded[int(step)] for step in query["frame_steps"]]
    hashes = [frame_sha256(frame) for frame in frames]
    if hashes != query["frame_sha256"]:
        raise ValueError("decoded frames disagree with frozen Qwen frame hashes")
    return _record_graph(
        trial=trial,
        run=run,
        query=query,
        frames=frames,
        output_dir=output_dir.resolve(),
        trial_sha256=file_sha256(trial_path),
        run_sha256=file_sha256(run_path),
    )


def main() -> None:
    args = _parse_arguments()
    result = build_query_graph(
        args.trial, args.run, args.query_step, args.output_dir
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
