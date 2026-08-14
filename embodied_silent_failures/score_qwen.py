import argparse
import importlib.metadata
import json
import re
import time
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.qwen import (
    HIDE_AND_SEEK_PAPER,
    MEDIA_BASIS,
    PARSER_BASIS,
    QWEN_MODEL_ID,
    SYSTEM_PROMPT,
    USER_PROMPT,
    build_messages,
    parse_response,
    prompt_sha256,
    query_steps,
    selected_frame_steps,
    trajectory_prediction,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    json_sha256,
    source_file_record,
)
from embodied_silent_failures.qwen_artifacts import (
    TrialSource,
    decode_selected_frames,
    frame_sha256,
    load_trial_manifest,
    prepare_output,
    snapshot_manifest,
    trial_checkpoint,
)


MODEL_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score selected rollout videos with a pinned Qwen observation monitor."
    )
    parser.add_argument("--trial-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument("--history-frames", required=True, type=int)
    parser.add_argument("--history-stride", required=True, type=int)
    parser.add_argument("--query-stride", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument(
        "--torch-dtype", required=True, choices=("bfloat16", "float16")
    )
    parser.add_argument(
        "--attention-implementation",
        required=True,
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _processor_configuration(processor: Any) -> dict[str, Any]:
    result = {}
    for name in ("image_processor", "video_processor", "tokenizer"):
        component = getattr(processor, name, None)
        if component is None:
            continue
        to_dict = getattr(component, "to_dict", None)
        configuration = to_dict() if callable(to_dict) else {}
        result[name] = {
            "class": f"{type(component).__module__}.{type(component).__qualname__}",
            "configuration": configuration,
            "configuration_sha256": json_sha256(configuration),
        }
    return result


def _score_trial(
    source: TrialSource,
    *,
    output_path: Path,
    configuration_sha256: str,
    history_frames: int,
    history_stride: int,
    query_stride: int,
    max_new_tokens: int,
    resume: bool,
    model: Any,
    processor: Any,
    torch: Any,
    cv2: Any,
    np: Any,
) -> dict[str, Any]:
    policy_steps = int(source.completion["policy_steps"])
    instruction = source.completion.get("task_description")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"rollout has no task description: {source.completion_path}")
    queries = query_steps(policy_steps, query_stride)
    steps_by_query = {
        step: selected_frame_steps(step, history_frames, history_stride)
        for step in queries
    }
    needed_steps = {item for steps in steps_by_query.values() for item in steps}
    frames, video_metadata = decode_selected_frames(
        source.video_path,
        needed_steps,
        policy_steps,
        cv2=cv2,
        np=np,
    )
    frame_hashes = {step: frame_sha256(frame) for step, frame in frames.items()}
    result = trial_checkpoint(
        output_path, source, configuration_sha256, queries, resume
    )
    if result.get("video_metadata") not in (None, video_metadata):
        raise ValueError(f"video metadata changed while resuming: {source.video_path}")
    result["video_metadata"] = video_metadata

    for current_step in queries[len(result["timeline"]) :]:
        frame_steps = steps_by_query[current_step]
        selected = [frames[step] for step in frame_steps]
        messages = build_messages(selected, instruction, history_frames)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        input_ids = inputs["input_ids"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        generated_token_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
        raw_response = processor.batch_decode(
            [generated_token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            decision = parse_response(raw_response)
            parsed = {
                "failure_now": decision.failure_now,
                "reason": decision.reason,
            }
            alarm = decision.alarm
            parse_error = None
        except ValueError as error:
            parsed = None
            alarm = None
            parse_error = str(error)
        result["timeline"].append(
            {
                "policy_step": current_step,
                "frame_steps": list(frame_steps),
                "frame_sha256": [frame_hashes[step] for step in frame_steps],
                "prompt_sha256": prompt_sha256(instruction, history_frames),
                "input_token_count": int(input_ids.shape[1]),
                "generated_token_ids": [int(token) for token in generated_token_ids],
                "raw_response": raw_response,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "alarm": alarm,
                "inference_seconds": inference_seconds,
            }
        )
        write_json_atomic(output_path, result)

    result["status"] = "complete"
    result["trajectory_failure_prediction"] = trajectory_prediction(
        [item["alarm"] for item in result["timeline"]]
    )
    result["invalid_response_count"] = sum(
        item["alarm"] is None for item in result["timeline"]
    )
    result["total_inference_seconds"] = sum(
        float(item["inference_seconds"]) for item in result["timeline"]
    )
    write_json_atomic(output_path, result)
    return result


def main() -> None:
    args = _parse_arguments()
    if MODEL_REVISION_PATTERN.fullmatch(args.model_revision) is None:
        raise ValueError("model revision must be a full 40-character commit hash")
    for name in ("history_frames", "history_stride", "query_stride", "max_new_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    installed_transformers = importlib.metadata.version("transformers")
    if installed_transformers != args.transformers_version:
        raise RuntimeError(
            f"Transformers is {installed_transformers}, expected {args.transformers_version}"
        )
    project_root = Path(__file__).resolve().parents[1]
    if git_dirty(project_root):
        raise RuntimeError(f"experiment code has uncommitted changes: {project_root}")
    manifest, trials = load_trial_manifest(args.trial_manifest.resolve())

    # huggingface_hub.snapshot_download resolves the exact requested Hub commit;
    # hashing every resolved file also fixes weights and processor assets even if
    # a cache is moved or the repository's default branch changes later.
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=QWEN_MODEL_ID,
            revision=args.model_revision,
            cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        )
    ).resolve()
    snapshot_record = snapshot_manifest(snapshot)
    configuration = {
        "paper_basis": HIDE_AND_SEEK_PAPER,
        "selection_basis": manifest["selection_basis"],
        "trial_manifest_sha256": file_sha256(args.trial_manifest),
        "model": {
            "id": QWEN_MODEL_ID,
            "revision": args.model_revision,
            "snapshot_sha256": snapshot_record["sha256"],
            "snapshot_files": snapshot_record["files"],
        },
        "protocol": {
            "history_frames": args.history_frames,
            "history_stride": args.history_stride,
            "query_stride": args.query_stride,
            "queries_every_paper_timestep": args.query_stride == 1,
            "media_basis": MEDIA_BASIS,
            "parser_basis": PARSER_BASIS,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "torch_dtype": args.torch_dtype,
            "attention_implementation": args.attention_implementation,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT,
        },
        "sources": [
            {
                "key": trial.key,
                "run_sha256": trial.run_sha256,
                "completion_sha256": trial.completion_sha256,
                "video_sha256": trial.video_sha256,
            }
            for trial in trials
        ],
        "transformers_version": installed_transformers,
    }
    configuration_sha256 = json_sha256(configuration)
    run_record = {
        "schema_version": 1,
        "status": "initializing",
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "repository_state": {
            "revision": git_revision(project_root),
            "dirty": False,
            "score_qwen_sha256": file_sha256(Path(__file__)),
        },
    }
    run_record = prepare_output(args.output_dir, run_record, args.resume)

    import cv2
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to score the Qwen monitor")
    dtype = getattr(torch, args.torch_dtype)
    processor = AutoProcessor.from_pretrained(str(snapshot))
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot),
        dtype=dtype,
        device_map="auto",
        attn_implementation=args.attention_implementation,
    )
    model.eval()
    runtime = {
        "torch_version": torch.__version__,
        "accelerate_version": importlib.metadata.version("accelerate"),
        "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "cuda_device": torch.cuda.get_device_name(model.device),
        "model_implementation": source_file_record(model),
        "processor_implementation": source_file_record(processor),
        "processor_configuration": _processor_configuration(processor),
    }
    existing_runtime = run_record.get("runtime")
    if existing_runtime not in (None, runtime):
        raise ValueError("Qwen runtime implementation changed while resuming")
    run_record["runtime"] = runtime
    run_record["status"] = "running"
    write_json_atomic(args.output_dir / "run.json", run_record)

    trial_dir = args.output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for trial in trials:
        results.append(
            _score_trial(
                trial,
                output_path=trial_dir / f"{trial.key}.json",
                configuration_sha256=configuration_sha256,
                history_frames=args.history_frames,
                history_stride=args.history_stride,
                query_stride=args.query_stride,
                max_new_tokens=args.max_new_tokens,
                resume=args.resume,
                model=model,
                processor=processor,
                torch=torch,
                cv2=cv2,
                np=np,
            )
        )

    summary = {
        "schema_version": 1,
        "configuration_sha256": configuration_sha256,
        "trial_count": len(results),
        "task_failures": sum(not item["success"] for item in results),
        "monitor_failure_predictions": sum(
            item["trajectory_failure_prediction"] is True for item in results
        ),
        "indeterminate_monitor_predictions": sum(
            item["trajectory_failure_prediction"] is None for item in results
        ),
        "invalid_response_count": sum(
            int(item["invalid_response_count"]) for item in results
        ),
        "trial_files": [str((trial_dir / f"{trial.key}.json").resolve()) for trial in trials],
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    run_record["status"] = "complete"
    run_record["summary_sha256"] = file_sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run.json", run_record)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
