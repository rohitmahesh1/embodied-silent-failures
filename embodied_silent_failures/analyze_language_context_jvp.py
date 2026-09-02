from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_context_jvp import (
    analyze_intervention,
    clean_full_prompt_states,
    context_tensors,
    load_context_arrays,
    sparse_rows,
)
from embodied_silent_failures.openvla_runtime import (
    CHECKPOINT_REVISION,
    load_runtime,
    model_config,
    validate_pinned_runtime,
)
from embodied_silent_failures.provenance import file_sha256, git_state, load_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test first-order composition through pinned OpenVLA language blocks."
        )
    )
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--libero-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--analysis-split", choices=("development", "holdout"), default="development"
    )
    parser.add_argument(
        "--frozen-design",
        type=Path,
        help="Required provenance record before reading the holdout split.",
    )
    parser.add_argument("--context-id", action="append", default=[])
    parser.add_argument("--source-layer", action="append", type=int, default=[])
    parser.add_argument(
        "--scale",
        action="append",
        type=float,
        default=[],
        help=(
            "Repeat to compare finite perturbation scales. Scaled sources are "
            "rounded to BF16 before replay and differentiation."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _error(error: Exception, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(limit=16),
        "updated_at": _now(),
        **extra,
    }


def _context_source(
    campaign_root: Path, context: dict[str, Any]
) -> tuple[Path, dict[str, Any], Path]:
    context_dir = (
        campaign_root
        / f"shard-{int(context['worker_shard'])}"
        / "contexts"
        / str(context["context_id"])
    )
    local_path = context_dir / "local.json"
    if not local_path.is_file():
        raise FileNotFoundError(f"context has no local record: {local_path}")
    local = load_json(local_path)
    interface = local.get("interface_archive")
    if not isinstance(interface, dict) or interface.get("schema_version") != 3:
        raise ValueError("context does not contain a schema-3 context archive")
    record = interface["artifact"]
    archive = context_dir / str(record["name"])
    if not archive.is_file() or archive.stat().st_size != int(record["bytes"]):
        raise FileNotFoundError("context interface archive is missing or truncated")
    return context_dir, local, archive


def _run_identity(args: argparse.Namespace, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    method = {
        "linearization_point": "each clean context",
        "perturbation": "the complete finite t-1 post-block residual replacement",
        "operator": "PyTorch reverse-over-reverse Jacobian-vector product",
        "factorization": (
            "pinned Llama attention and MLP residual sublayers, followed by "
            "the final norm and language-model head"
        ),
        "scope": (
            "continuous propagation within the selected action-token call; "
            "autoregressive token selection is an explicit discrete boundary"
        ),
        "execution_shape": (
            "generation call zero preserves the complete fused prompt sequence; "
            "calls one through six preserve their original one-token shape"
        ),
    }
    analysis = "openvla_context_conditioned_jvp"
    if args.scale:
        analysis = "openvla_context_conditioned_jvp_scale_sweep"
        method["finite_scale_sweep"] = {
            "requested_scales": args.scale,
            "source_realization": (
                "interpolate in float32, round once to the model's BF16 input, "
                "and differentiate along that realized displacement"
            ),
        }
    return {
        "schema_version": 1,
        "analysis": analysis,
        "analysis_split": args.analysis_split,
        "context_ids": [str(context["context_id"]) for context in contexts],
        "source_layers": args.source_layer or list(range(32)),
        "method": method,
        "code": git_state(project_root),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": file_sha256(args.manifest),
        },
        "frozen_design": (
            {
                "path": str(args.frozen_design.resolve()),
                "sha256": file_sha256(args.frozen_design),
            }
            if args.frozen_design is not None
            else None
        ),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "runtime_packages": {
            name: importlib.metadata.version(name)
            for name in ("flash-attn", "torch", "transformers")
        },
        "archive_validation": (
            "check the artifact byte count against its creation-time record; "
            "retain that record's SHA-256 without rereading each 300-500 MiB file"
        ),
        "started_at": _now(),
    }


def main() -> None:
    args = _arguments()
    if args.analysis_split == "holdout" and args.frozen_design is None:
        raise ValueError("holdout analysis requires a frozen design record")
    if args.frozen_design is not None and not args.frozen_design.is_file():
        raise FileNotFoundError(f"frozen design is missing: {args.frozen_design}")
    if any(not 0.0 < scale <= 1.0 for scale in args.scale):
        raise ValueError("perturbation scales must be greater than zero and at most one")
    if len(set(args.scale)) != len(args.scale):
        raise ValueError("perturbation scales must be unique")
    source_layers = args.source_layer or list(range(32))
    if len(set(source_layers)) != len(source_layers) or any(
        layer < 0 or layer >= 32 for layer in source_layers
    ):
        raise ValueError("source layers must be unique values from zero through 31")

    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    manifest = load_json(args.manifest)
    contexts = [
        context
        for context in manifest["contexts"]
        if str(context["analysis_split"]) == args.analysis_split
    ]
    if args.context_id:
        selected = set(args.context_id)
        known = {str(context["context_id"]) for context in contexts}
        if selected - known:
            raise ValueError(
                f"requested contexts are not in {args.analysis_split}: {sorted(selected-known)}"
            )
        contexts = [
            context for context in contexts if str(context["context_id"]) in selected
        ]
    contexts.sort(key=lambda value: str(value["context_id"]))

    project_root = Path(__file__).resolve().parents[1]
    validate_pinned_runtime(
        args.checkpoint,
        args.openvla_root,
        args.libero_root,
        project_root=project_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    identity = _run_identity(args, contexts)
    if run_path.exists() and not args.resume:
        raise FileExistsError(f"analysis output already exists: {run_path}")
    if run_path.exists():
        existing = load_json(run_path)
        for key in (
            "analysis",
            "analysis_split",
            "context_ids",
            "source_layers",
            "method",
            "code",
            "manifest",
            "frozen_design",
            "checkpoint_revision",
            "runtime_packages",
            "archive_validation",
        ):
            if existing.get(key) != identity.get(key):
                raise ValueError(f"resume run identity changed at {key}")
    else:
        write_json_atomic(run_path, identity)

    runtime = load_runtime(args.openvla_root, args.libero_root)
    runtime.set_seed_everywhere(int(manifest["seed"]))
    model = runtime.get_model(model_config(args.checkpoint, "libero_10")).eval()
    model.requires_grad_(False)
    counts = Counter()
    started = time.perf_counter()

    for context in contexts:
        context_id = str(context["context_id"])
        destination = args.output_dir / "contexts" / context_id
        destination.mkdir(parents=True, exist_ok=True)
        try:
            _source_dir, local, archive_path = _context_source(
                args.campaign_root, context
            )
            arrays = load_context_arrays(runtime.np, archive_path)
            token = int(context["action_token_position"])
            tensor_context = context_tensors(
                runtime.np,
                runtime.torch,
                arrays,
                token,
                next(model.parameters()).device,
            )
            if token == 0:
                layers = model.language_model.model.layers
                tensor_context["full_prompt_states"] = clean_full_prompt_states(
                    runtime.torch, layers, tensor_context
                )
            indices = {
                name: sparse_rows(runtime.np, arrays, name)
                for name in (
                    "residuals",
                    "post_attention_residuals",
                    "attention_cache_keys",
                    "attention_cache_values",
                )
            }
            for source_layer in source_layers:
                output = destination / f"layer-{source_layer:02d}.json"
                if output.is_file() and args.resume:
                    counts[load_json(output).get("status", "unknown")] += 1
                    continue
                layer_started = time.perf_counter()
                try:
                    result = analyze_intervention(
                        runtime.np,
                        runtime.torch,
                        model,
                        arrays,
                        tensor_context,
                        indices,
                        source_layer,
                        token,
                        scales=tuple(args.scale) or None,
                    )
                    result.update(
                        {
                            "context": context,
                            "source_archive": local["interface_archive"]["artifact"],
                            "elapsed_seconds": time.perf_counter() - layer_started,
                        }
                    )
                except Exception as error:
                    result = _error(
                        error,
                        context=context,
                        source_layer=source_layer,
                        elapsed_seconds=time.perf_counter() - layer_started,
                    )
                    runtime.torch.cuda.empty_cache()
                write_json_atomic(output, result)
                counts[result["status"]] += 1
                print(
                    f"{context_id} layer {source_layer}: {result['status']}", flush=True
                )
            del tensor_context, arrays
            runtime.torch.cuda.empty_cache()
        except Exception as error:
            write_json_atomic(destination / "context.error.json", _error(error, context=context))
            counts["context_error"] += len(source_layers)
        write_json_atomic(
            args.output_dir / "status.json",
            {
                "schema_version": 1,
                "state": "running",
                "counts": dict(sorted(counts.items())),
                "last_context": context_id,
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at": _now(),
            },
        )

    final = {
        "schema_version": 1,
        "state": "complete",
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": _now(),
    }
    write_json_atomic(args.output_dir / "status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
