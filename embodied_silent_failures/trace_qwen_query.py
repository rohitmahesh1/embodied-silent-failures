import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.audit import audit_graph
from embodied_silent_failures.evidence_graph.qwen import (
    QWEN_MODEL_ID,
    REQUIRED_ENDPOINTS,
    build_messages,
    contract_issues as qwen_contract_issues,
    internal_annotations,
    parse_response,
    record_decision,
    record_monitor_input,
    record_observation_frame,
    record_processor_output,
    record_traced_response,
)
from embodied_silent_failures.evidence_graph.record import Recorder
from embodied_silent_failures.evidence_graph.reduce import reduce_graph
from embodied_silent_failures.evidence_graph.torch_trace import (
    capture_torch_operations,
    contract_issues as trace_contract_issues,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    load_json,
    source_file_record,
)
from embodied_silent_failures.qwen_artifacts import (
    decode_selected_frames,
    frame_sha256,
    select_trace_query,
    snapshot_manifest,
)
from embodied_silent_failures.qwen_saturation import coverage_record


CONSTRUCTION = "live_qwen_internal_trace_from_frozen_query"


@dataclass(frozen=True)
class TraceRuntime:
    reference_run: dict[str, Any]
    revision: str
    model: Any
    processor: Any
    torch: Any
    cv2: Any
    np: Any
    model_implementation: dict[str, Any]
    processor_implementation: dict[str, Any]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace one mechanically selected frozen Qwen monitor query."
    )
    parser.add_argument("--trials-dir", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def _implementation_basis(kind: str, revision: str, record: dict[str, Any]) -> str:
    return (
        f"code:qwen@{revision}:{record['class']}:{kind}:"
        f"file-sha256-{record['sha256']}"
    )


def _verify_implementation(
    name: str, actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key in ("class", "sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{name} {key} differs from the frozen scoring runtime")


def _processed_input_records(inputs: Any, torch: Any) -> list[dict[str, Any]]:
    records = []
    for name, value in sorted(inputs.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Qwen processor output {name} is not a tensor")
        contiguous = value.detach().contiguous()
        raw = contiguous.view(torch.uint8).cpu().numpy().tobytes()
        digest = hashlib.sha256()
        digest.update(str(tuple(int(size) for size in value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(raw)
        records.append(
            {
                "name": name,
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
                "sha256": digest.hexdigest(),
            }
        )
    return records


def load_trace_runtime(run_path: Path, cache_dir: Path | None = None) -> TraceRuntime:
    project_root = Path(__file__).resolve().parents[1]
    if git_dirty(project_root):
        raise ValueError("Qwen internal tracing requires a clean experiment repository")
    revision = git_revision(project_root)
    run = load_json(run_path.resolve())
    protocol = run["configuration"]["protocol"]
    model_record = run["configuration"]["model"]
    if model_record.get("id") != QWEN_MODEL_ID:
        raise ValueError("Qwen trace run uses an unexpected model")

    import cv2
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    snapshot = Path(
        snapshot_download(
            repo_id=QWEN_MODEL_ID,
            revision=model_record["revision"],
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    actual_snapshot = snapshot_manifest(snapshot)
    if actual_snapshot["sha256"] != model_record["snapshot_sha256"]:
        raise ValueError("Qwen snapshot differs from the frozen scoring campaign")
    if actual_snapshot["files"] != model_record["snapshot_files"]:
        raise ValueError("Qwen snapshot file manifest differs from the scoring campaign")

    dtype = getattr(torch, protocol["torch_dtype"])
    processor = AutoProcessor.from_pretrained(str(snapshot))
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot),
        dtype=dtype,
        device_map="auto",
        attn_implementation=protocol["attention_implementation"],
    )
    model.eval()
    model_implementation = source_file_record(model)
    processor_implementation = source_file_record(processor)
    _verify_implementation(
        "Qwen model", model_implementation, run["runtime"]["model_implementation"]
    )
    _verify_implementation(
        "Qwen processor",
        processor_implementation,
        run["runtime"]["processor_implementation"],
    )
    if torch.__version__ != run["runtime"]["torch_version"]:
        raise ValueError("PyTorch version differs from the frozen scoring runtime")
    if importlib.metadata.version("transformers") != run["configuration"][
        "transformers_version"
    ]:
        raise ValueError("Transformers version differs from the frozen scoring runtime")
    return TraceRuntime(
        reference_run=run,
        revision=revision,
        model=model,
        processor=processor,
        torch=torch,
        cv2=cv2,
        np=np,
        model_implementation=model_implementation,
        processor_implementation=processor_implementation,
    )


def _verify_compatible_run(runtime: TraceRuntime, run: dict[str, Any]) -> None:
    reference = runtime.reference_run
    for key in ("model", "protocol", "transformers_version"):
        if run["configuration"][key] != reference["configuration"][key]:
            raise ValueError(f"Qwen batch source has incompatible {key}")
    for key in ("torch_version", "model_implementation", "processor_implementation"):
        if run["runtime"][key] != reference["runtime"][key]:
            raise ValueError(f"Qwen batch source has incompatible runtime {key}")


def trace_selected_query(
    trials_dir: Path,
    run_path: Path,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    run_path = run_path.resolve()
    run = load_json(run_path)
    protocol = run["configuration"]["protocol"]
    model_record = run["configuration"]["model"]
    if model_record.get("id") != QWEN_MODEL_ID:
        raise ValueError("Qwen trace run uses an unexpected model")
    selection = select_trace_query(
        trials_dir.resolve(),
        configuration_sha256=run["configuration_sha256"],
        history_frames=int(protocol["history_frames"]),
    )
    selection.update(
        {
            "run": run,
            "run_path": run_path,
            "run_sha256": file_sha256(run_path),
        }
    )
    runtime = load_trace_runtime(run_path, cache_dir)
    return trace_query(runtime, selection, output_dir)


def trace_query(
    runtime: TraceRuntime,
    selection: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    run = selection["run"]
    _verify_compatible_run(runtime, run)
    protocol = run["configuration"]["protocol"]
    model_record = run["configuration"]["model"]
    trial = selection["trial"]
    query = selection["query"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Qwen trace output is not empty: {output_dir}")

    model = runtime.model
    processor = runtime.processor
    torch = runtime.torch

    frame_steps = [int(value) for value in query["frame_steps"]]
    decoded, video_metadata = decode_selected_frames(
        Path(trial["video_path"]),
        set(frame_steps),
        int(trial["policy_steps"]),
        cv2=runtime.cv2,
        np=runtime.np,
    )
    frames = [decoded[step] for step in frame_steps]
    frame_hashes = [frame_sha256(frame) for frame in frames]
    if frame_hashes != query["frame_sha256"]:
        raise ValueError("decoded Qwen trace frames disagree with frozen query hashes")

    model_basis = _implementation_basis(
        "generate", model_record["revision"], runtime.model_implementation
    )
    processor_basis = _implementation_basis(
        "apply_chat_template", model_record["revision"], runtime.processor_implementation
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    metadata = {
        "schema_version": 1,
        "construction": CONSTRUCTION,
        "experiment_revision": runtime.revision,
        "trial_sha256": selection["trial_sha256"],
        "qwen_run_sha256": selection["run_sha256"],
        "policy_step": selection["policy_step"],
    }
    started = time.perf_counter()
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
            instruction=trial["task_description"],
            frame_steps=frame_steps,
            frame_sha256=frame_hashes,
            history_frames=int(protocol["history_frames"]),
        )
        messages = build_messages(
            frames, trial["task_description"], int(protocol["history_frames"])
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        processed_inputs = _processed_input_records(inputs, torch)
        record_processor_output(
            recorder, request, inputs, processor_basis=processor_basis
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with recorder.scope(phase="qwen_model"), capture_torch_operations(
            recorder, {"qwen_model": model}
        ), torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(protocol["max_new_tokens"]),
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        input_ids = inputs["input_ids"]
        generated_token_ids = (
            generated[0, input_ids.shape[1] :].detach().cpu().tolist()
        )
        raw_response = processor.batch_decode(
            [generated_token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if generated_token_ids != query["generated_token_ids"]:
            raise ValueError("traced Qwen token IDs differ from the frozen query")
        if raw_response != query["raw_response"]:
            raise ValueError("traced Qwen response differs from the frozen query")
        response = record_traced_response(
            recorder,
            generated,
            generated_token_ids,
            raw_response,
            model_basis=model_basis,
        )
        decision = parse_response(raw_response)
        if {
            "failure_now": decision.failure_now,
            "reason": decision.reason,
        } != query["parsed_response"]:
            raise ValueError("traced Qwen parser result differs from the frozen query")
        alarm = record_decision(
            recorder, response, decision, policy_step=selection["policy_step"]
        )
        if alarm is not query["alarm"]:
            raise ValueError("traced Qwen alarm differs from the frozen query")

    events = recorder.events
    annotations = [
        *recorder.annotations,
        *internal_annotations(events, model_basis=model_basis),
    ]
    graph = reduce_graph(events, annotations)
    issues = [
        *qwen_contract_issues(events),
        *trace_contract_issues(events, ("qwen_model",)),
    ]
    audit = audit_graph(
        events,
        annotations,
        graph,
        required_endpoints=(*REQUIRED_ENDPOINTS, "qwen.processor_output"),
        contract_issues=issues,
    )
    audit["construction"] = {
        "kind": CONSTRUCTION,
        "established": (
            "The selected decoded frames, processor tensors, model state, dispatched "
            "operators, generated tokens, response, parser output, and alarm were observed."
        ),
        "not_established": (
            "One query does not establish complete Qwen runtime-path coverage, physical "
            "hardware placement, or fault prevalence."
        ),
    }

    selection_metadata = {
        **selection.get("selection", {}),
        **(
            {
                "basis": selection.get("selection_basis"),
                "phase": selection.get("phase"),
                "stratum": selection.get("stratum"),
                "selection_rule": selection.get("selection_rule"),
                "alarm_used_for_selection": selection.get(
                    "alarm_used_for_selection", True
                ),
            }
            if selection.get("phase")
            else {}
        ),
        "trial": selection["trial_path"].name,
        "policy_step": selection["policy_step"],
        "generated_tokens": selection["generated_tokens"],
    }
    selection_metadata = {
        key: value for key, value in selection_metadata.items() if value is not None
    }
    composition = {
        "schema_version": 1,
        "construction": CONSTRUCTION,
        "selection": selection_metadata,
        "source": trial["source"],
        "task_id": int(trial["task_id"]),
        "episode_index": int(trial["episode_index"]),
        "frame_steps": frame_steps,
        "frame_sha256": frame_hashes,
        "video_metadata": video_metadata,
        "processed_inputs": processed_inputs,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "generated_token_ids": generated_token_ids,
        "raw_response_sha256": hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest(),
        "alarm": alarm,
        "equivalent_to_frozen_query": True,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "trial_sha256": selection["trial_sha256"],
        "qwen_run_sha256": selection["run_sha256"],
        "configuration_sha256": run["configuration_sha256"],
        "source_scoring_revision": run["repository_state"]["revision"],
        "trace_revision": runtime.revision,
        "model": model_record,
        "model_implementation": runtime.model_implementation,
        "processor_implementation": runtime.processor_implementation,
    }
    coverage = coverage_record(events, graph, processed_inputs)
    write_json_atomic(
        output_dir / "annotations.json",
        {"schema_version": 1, "annotations": annotations},
    )
    write_json_atomic(output_dir / "graph.json", graph)
    write_json_atomic(output_dir / "audit.json", audit)
    write_json_atomic(output_dir / "composition.json", composition)
    write_json_atomic(output_dir / "coverage.json", coverage)
    if not audit["passed"]:
        raise RuntimeError("traced Qwen evidence graph did not pass its audit")
    return {
        "audit_passed": True,
        "trial": selection["trial_path"].name,
        "policy_step": selection["policy_step"],
        "generated_tokens": selection["generated_tokens"],
        "raw_events": graph["raw_event_count"],
        "operator_events": graph["operator_event_count"],
        "regions": len(graph["regions"]),
        "edges": len(graph["edges"]),
        "coverage": {
            kind: len(coverage[kind])
            for kind in ("regions", "edges", "operators", "processor_shapes")
        },
        "inference_seconds": inference_seconds,
    }


def main() -> None:
    args = _parse_arguments()
    result = trace_selected_query(
        args.trials_dir, args.run, args.output_dir.resolve(), args.cache_dir
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
