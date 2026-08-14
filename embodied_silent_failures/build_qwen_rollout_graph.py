import argparse
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.audit import audit_graph
from embodied_silent_failures.evidence_graph.openvla import (
    REQUIRED_ENDPOINTS as OPENVLA_ENDPOINTS,
    contract_issues as openvla_contract_issues,
)
from embodied_silent_failures.evidence_graph.qwen import (
    contract_issues as qwen_contract_issues,
)
from embodied_silent_failures.evidence_graph.qwen_rollout import (
    QWEN_EVIDENCE_ENDPOINTS,
    ROLLOUT_ENDPOINTS,
    compose_qwen_rollout,
)
from embodied_silent_failures.evidence_graph.record import read_events
from embodied_silent_failures.evidence_graph.reduce import reduce_graph
from embodied_silent_failures.evidence_graph.torch_trace import (
    contract_issues as torch_trace_contract_issues,
)
from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.qwen_artifacts import (
    decode_selected_frames,
    frame_sha256,
)


CONSTRUCTION = "qwen_monitor_composed_with_exact_video_backed_openvla_rollout"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose one whole-rollout OpenVLA and Qwen evidence graph."
    )
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--qwen-trial", required=True, type=Path)
    parser.add_argument("--qwen-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _source_revision(source_run: dict[str, Any]) -> str:
    repositories = source_run.get("repository_states")
    repository = (
        repositories.get("experiment_code")
        if isinstance(repositories, dict)
        else None
    )
    if not isinstance(repository, dict):
        raise ValueError("source rollout run does not record repository state")
    revision = repository.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("source rollout run does not pin its experiment revision")
    if repository.get("dirty") is not False:
        raise ValueError("source rollout used uncommitted experiment code")
    if source_run.get("upstream_revisions", {}).get("experiment_code") != revision:
        raise ValueError("source rollout records conflicting experiment revisions")
    return revision


def _matching_source(
    qwen_run: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        item
        for item in qwen_run["configuration"].get("sources", [])
        if item.get("run_sha256") == trial.get("run_sha256")
        and item.get("completion_sha256") == trial.get("completion_sha256")
        and item.get("video_sha256") == trial.get("video_sha256")
    ]
    if len(matches) != 1:
        raise ValueError("Qwen run does not contain exactly one matching source artifact")
    return matches[0]


def _validate_artifacts(
    *,
    evidence_dir: Path,
    completion_path: Path,
    trial_path: Path,
    qwen_run_path: Path,
) -> dict[str, Any]:
    required = (
        "raw.jsonl",
        "annotations.json",
        "graph.json",
        "audit.json",
        "composition.json",
    )
    missing = [name for name in required if not (evidence_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source evidence is missing {missing}: {evidence_dir}")

    source_audit = load_json(evidence_dir / "audit.json")
    if source_audit.get("passed") is not True:
        raise ValueError("source rollout evidence audit did not pass")
    source_composition = load_json(evidence_dir / "composition.json")
    completion = load_json(completion_path)
    trial = load_json(trial_path)
    qwen_run = load_json(qwen_run_path)
    source_run_path = completion_path.parent / "run.json"
    source_run = load_json(source_run_path)

    if completion.get("status") != "complete" or trial.get("status") != "complete":
        raise ValueError("source rollout and Qwen scoring trial must both be complete")
    if qwen_run.get("status") != "complete":
        raise ValueError("Qwen scoring run is incomplete")
    if trial.get("configuration_sha256") != qwen_run.get("configuration_sha256"):
        raise ValueError("Qwen trial and run configurations disagree")

    video_name = completion.get("files", {}).get("video")
    if not isinstance(video_name, str) or Path(video_name).name != video_name:
        raise ValueError("source completion has no local rollout video")
    video_path = completion_path.parent / video_name
    if not video_path.is_file():
        raise FileNotFoundError(f"source rollout video is missing: {video_path}")

    hashes = {
        "completion": file_sha256(completion_path),
        "source_run": file_sha256(source_run_path),
        "video": file_sha256(video_path),
        "qwen_trial": file_sha256(trial_path),
        "qwen_run": file_sha256(qwen_run_path),
        "source_raw": file_sha256(evidence_dir / "raw.jsonl"),
        "source_annotations": file_sha256(evidence_dir / "annotations.json"),
        "source_graph": file_sha256(evidence_dir / "graph.json"),
        "source_audit": file_sha256(evidence_dir / "audit.json"),
        "source_composition": file_sha256(evidence_dir / "composition.json"),
    }
    expected_hashes = {
        "completion": trial.get("completion_sha256"),
        "source_run": trial.get("run_sha256"),
        "video": trial.get("video_sha256"),
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"Qwen trial {name} hash disagrees with the source rollout")
    _matching_source(qwen_run, trial)

    scalar_fields = ("task_id", "episode_index", "policy_steps", "success", "fault")
    for field in scalar_fields:
        if trial.get(field) != completion.get(field):
            raise ValueError(f"Qwen trial and source completion disagree on {field}")
    for field in ("policy_steps", "success", "fault"):
        if source_composition.get(field) != completion.get(field):
            raise ValueError(f"source evidence and completion disagree on {field}")
    if trial.get("task_description") != completion.get("task_description"):
        raise ValueError("Qwen trial and source completion disagree on task description")

    evidence_record = completion.get("evidence_graph")
    if not isinstance(evidence_record, dict) or evidence_record.get("audit_passed") is not True:
        raise ValueError("source completion does not identify passing rollout evidence")
    relative = evidence_record.get("directory_relative_to_run")
    if isinstance(relative, str):
        expected_evidence = (completion_path.parent / relative).resolve()
        if expected_evidence != evidence_dir.resolve():
            raise ValueError("source completion refers to a different evidence directory")

    return {
        "completion": completion,
        "trial": trial,
        "qwen_run": qwen_run,
        "source_run": source_run,
        "source_composition": source_composition,
        "source_audit": source_audit,
        "video_path": video_path,
        "hashes": hashes,
    }


def _verify_frozen_frames(trial: dict[str, Any], video_path: Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    expected: dict[int, str] = {}
    for query in trial["timeline"]:
        for step, digest in zip(
            query.get("frame_steps", []), query.get("frame_sha256", []), strict=True
        ):
            step = int(step)
            prior = expected.setdefault(step, str(digest))
            if prior != digest:
                raise ValueError(f"Qwen frame hash changes at policy step {step}")
    decoded, metadata = decode_selected_frames(
        video_path,
        set(expected),
        int(trial["policy_steps"]),
        cv2=cv2,
        np=np,
    )
    actual = {step: frame_sha256(frame) for step, frame in decoded.items()}
    if actual != expected:
        raise ValueError("decoded rollout frames disagree with frozen Qwen evidence")
    if metadata != trial.get("video_metadata"):
        raise ValueError("decoded rollout video metadata changed after Qwen scoring")
    return {"verified_frame_count": len(expected), "video_metadata": metadata}


def build_qwen_rollout_graph(
    *,
    evidence_dir: Path,
    completion_path: Path,
    trial_path: Path,
    qwen_run_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    completion_path = completion_path.resolve()
    trial_path = trial_path.resolve()
    qwen_run_path = qwen_run_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Qwen rollout graph output is not empty: {output_dir}")

    artifacts = _validate_artifacts(
        evidence_dir=evidence_dir,
        completion_path=completion_path,
        trial_path=trial_path,
        qwen_run_path=qwen_run_path,
    )
    frame_verification = _verify_frozen_frames(
        artifacts["trial"], artifacts["video_path"]
    )
    source_annotations = load_json(evidence_dir / "annotations.json")["annotations"]
    events, annotations = compose_qwen_rollout(
        read_events(evidence_dir / "raw.jsonl"),
        source_annotations,
        trial=artifacts["trial"],
        qwen_run=artifacts["qwen_run"],
        source_revision=_source_revision(artifacts["source_run"]),
    )
    graph = reduce_graph(events, annotations)
    traced_policy = bool(artifacts["source_composition"].get("traced_steps"))
    issues = [
        *openvla_contract_issues(events),
        *qwen_contract_issues(events),
        *torch_trace_contract_issues(events, ("policy",) if traced_policy else ()),
    ]
    audit = audit_graph(
        events,
        annotations,
        graph,
        required_endpoints=ROLLOUT_ENDPOINTS,
        repeated_endpoints=OPENVLA_ENDPOINTS + QWEN_EVIDENCE_ENDPOINTS,
        contract_issues=issues,
    )
    audit["construction"] = {
        "kind": CONSTRUCTION,
        "established": (
            "The audited OpenVLA rollout, its exact encoded video, every decoded Qwen "
            "input frame, frozen Qwen response and alarm, and task outcome were matched "
            "by recorded hashes and composed into one lineage graph."
        ),
        "not_established": (
            "Lossy video frames are not asserted to be byte-identical to pre-encoding "
            "simulator images. Qwen inference is opaque here to match the primary SAFE "
            "rollout graph; separate traces cover Qwen internals."
        ),
    }
    if not audit["passed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir / "audit.json", audit)
        raise RuntimeError(f"composed Qwen rollout graph audit failed: {output_dir}")

    composition = {
        "schema_version": 1,
        "construction": CONSTRUCTION,
        "source": artifacts["trial"]["source"],
        "task_id": int(artifacts["trial"]["task_id"]),
        "episode_index": int(artifacts["trial"]["episode_index"]),
        "condition": artifacts["completion"]["condition"],
        "policy_steps": int(artifacts["trial"]["policy_steps"]),
        "success": bool(artifacts["trial"]["success"]),
        "fault": artifacts["trial"]["fault"],
        "query_count": len(artifacts["trial"]["timeline"]),
        "frame_verification": frame_verification,
        "source_hashes": artifacts["hashes"],
        "source_evidence_audit_passed": True,
        "source_evidence_traced_steps": artifacts["source_composition"].get(
            "traced_steps", []
        ),
        "qwen_configuration_sha256": artifacts["qwen_run"]["configuration_sha256"],
        "qwen_model": artifacts["qwen_run"]["configuration"]["model"],
        "qwen_model_implementation": artifacts["qwen_run"]["runtime"][
            "model_implementation"
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "raw.jsonl").open("x", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
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
        "queries": len(artifacts["trial"]["timeline"]),
    }


def main() -> None:
    args = _parse_arguments()
    result = build_qwen_rollout_graph(
        evidence_dir=args.evidence_dir,
        completion_path=args.completion,
        trial_path=args.qwen_trial,
        qwen_run_path=args.qwen_run,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
