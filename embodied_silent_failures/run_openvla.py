import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from embodied_silent_failures.artifacts import (
    completion_path,
    exclusion_path,
    prepare_trial,
    safe_stem,
    write_json_atomic,
)
from embodied_silent_failures.faults import (
    FAULT_SITES,
    FaultSpec,
    TransientActivationFault,
)
from embodied_silent_failures.evidence_graph.rollout import (
    RolloutEvidence,
    prepare_evidence_output,
    summarize_saturation,
)
from embodied_silent_failures.fault_manifest import load_fault_manifest
from embodied_silent_failures.plan import (
    Trial,
    build_trial_plan,
    load_trial_manifest,
    parse_task_ids,
    seed_for_trial,
)
from embodied_silent_failures.provenance import (
    file_sha256,
    git_dirty,
    git_revision,
    git_state,
)
from embodied_silent_failures.replay import (
    CleanTrace,
    load_clean_trace,
    observation_error,
    replay_action,
)
from embodied_silent_failures.stale_image_manifest import (
    StaleImageSpec,
    load_stale_image_manifest,
)


OPENVLA_REVISION = "300dce26d44f407c725695d16cd445755c92cbd1"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_REVISION = "80970322773f81baa2e22fe495d0487b93a05cfa"
CONTAINER_IMAGE = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
REPLAY_OBSERVATION_TOLERANCE = 1e-6


class CounterfactualReplayInvalid(RuntimeError):
    reason: str


class CounterfactualReplayDivergence(CounterfactualReplayInvalid):
    reason = "counterfactual_replay_diverged_before_intervention"

    def __init__(self, policy_step: int, error: float):
        self.policy_step = policy_step
        self.error = error
        super().__init__(
            f"counterfactual replay diverged at step {policy_step}: "
            f"maximum numeric observation error {error:.3g} exceeds "
            f"{REPLAY_OBSERVATION_TOLERANCE:.3g}"
        )


class CounterfactualReplayTerminated(CounterfactualReplayInvalid):
    reason = "counterfactual_replay_terminated_before_intervention"

    def __init__(self, policy_step: int, intervention_step: int):
        self.policy_step = policy_step
        self.intervention_step = intervention_step
        super().__init__(
            f"counterfactual replay terminated after step {policy_step}, "
            f"before intervention step {intervention_step}"
        )


EXPECTED_PACKAGES = {
    "flash-attn": "2.5.5",
    "mujoco": "3.3.2",
    "numpy": "1.26.3",
    "protobuf": "3.20.3",
    "robosuite": "1.4.0",
    "tensorflow": "2.15.0",
    "tensorflow-datasets": "4.9.3",
    "tensorflow-metadata": "1.15.0",
    "torch": "2.2.0",
    "transformers": "4.40.1",
    "wandb": "0.16.6",
}

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass(frozen=True)
class Arguments:
    checkpoint: Path
    openvla_root: Path
    libero_root: Path
    output_dir: Path
    task_suite: str
    task_ids: str
    trial_manifest: Path | None
    episode_start: int
    episode_stop: int
    episode_stride: int
    seed: int
    wait_steps: int
    save_video: bool
    resume: bool
    fault_site: str | None
    fault_manifest: Path | None
    stale_image_manifest: Path | None
    image_input_mode: str
    fault_layer: int | None
    fault_policy_step: int | None
    fault_generation_step: int
    fault_bit_index: int | None
    fault_feature_index: int | None
    fault_seed: int
    replay_clean_prefix: bool
    paired_clean_dirs: list[Path]
    evidence_dir: Path | None = None
    evidence_trace_steps: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Runtime:
    np: Any
    torch: Any
    benchmark: Any
    get_action: Any
    get_image_resize_size: Any
    get_libero_dummy_action: Any
    get_libero_env: Any
    get_libero_image: Any
    get_model: Any
    get_processor: Any
    invert_gripper_action: Any
    normalize_gripper_action: Any
    quat2axisangle: Any
    save_video: Any
    set_seed_everywhere: Any
    compute_token_uncertainty_metrics: Any


def _parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        description="Collect clean OpenVLA rollouts and SAFE-compatible evidence on LIBERO."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-suite", choices=sorted(MAX_STEPS), default="libero_10")
    parser.add_argument("--task-ids", default="0-9")
    parser.add_argument("--trial-manifest", type=Path)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-stop", type=int, default=50)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fault-site", choices=FAULT_SITES)
    parser.add_argument("--fault-manifest", type=Path)
    parser.add_argument("--stale-image-manifest", type=Path)
    parser.add_argument(
        "--image-input-mode",
        choices=("stale", "current_control"),
        default="stale",
    )
    parser.add_argument("--fault-layer", type=int)
    parser.add_argument("--fault-policy-step", type=int)
    parser.add_argument("--fault-generation-step", type=int, default=0)
    parser.add_argument("--fault-bit-index", type=int)
    parser.add_argument("--fault-feature-index", type=int)
    parser.add_argument("--fault-seed", type=int, default=0)
    parser.add_argument("--replay-clean-prefix", action="store_true")
    parser.add_argument(
        "--paired-clean-dir",
        dest="paired_clean_dirs",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--evidence-trace-step",
        dest="evidence_trace_steps",
        action="append",
        type=int,
        default=[],
    )
    namespace = parser.parse_args()
    return Arguments(**vars(namespace))


def _fault_spec(args: Arguments) -> FaultSpec | None:
    if args.image_input_mode != "stale" and args.stale_image_manifest is None:
        raise ValueError(
            "--image-input-mode current_control requires --stale-image-manifest"
        )
    if args.stale_image_manifest is not None:
        static_values = (
            args.fault_site,
            args.fault_layer,
            args.fault_policy_step,
            args.fault_bit_index,
            args.fault_feature_index,
        )
        if args.fault_manifest is not None or any(
            value is not None for value in static_values
        ):
            raise ValueError(
                "--stale-image-manifest cannot be combined with activation-fault options"
            )
        return None
    if args.fault_manifest is not None:
        static_values = (
            args.fault_site,
            args.fault_layer,
            args.fault_policy_step,
            args.fault_bit_index,
            args.fault_feature_index,
        )
        if any(value is not None for value in static_values):
            raise ValueError("--fault-manifest cannot be combined with static fault options")
        return None
    if args.fault_site is None:
        if any(
            value is not None
            for value in (
                args.fault_layer,
                args.fault_policy_step,
                args.fault_bit_index,
                args.fault_feature_index,
            )
        ):
            raise ValueError("fault options require --fault-site")
        if args.paired_clean_dirs and args.stale_image_manifest is None:
            raise ValueError("--paired-clean-dir requires --fault-site")
        return None
    if args.fault_policy_step is None:
        raise ValueError("--fault-site requires --fault-policy-step")
    if not args.paired_clean_dirs:
        raise ValueError("--fault-site requires at least one --paired-clean-dir")
    return FaultSpec(
        site=args.fault_site,
        layer=args.fault_layer,
        policy_step=args.fault_policy_step,
        generation_step=args.fault_generation_step,
        bit_index=args.fault_bit_index,
        seed=args.fault_seed,
        feature_index=args.fault_feature_index,
    )


def _image_intervention_record(
    spec: StaleImageSpec, mode: str, trial_seed: int
) -> dict[str, Any]:
    if mode == "stale":
        return {**spec.to_dict(), "trial_seed": trial_seed}
    if mode == "current_control":
        return {
            "kind": "current_image_control",
            "policy_step": spec.policy_step,
            "input_policy_step": spec.policy_step,
            "matched_stale_image_lag": spec.image_lag,
            "matched_stale_source_policy_step": spec.source_policy_step,
            "trial_seed": trial_seed,
        }
    raise ValueError(f"unsupported image input mode: {mode}")


def _image_fault_applied(spec: StaleImageSpec, mode: str, policy_step: int) -> bool:
    if mode not in {"stale", "current_control"}:
        raise ValueError(f"unsupported image input mode: {mode}")
    return mode == "stale" and policy_step == spec.policy_step


def _validate_inputs(args: Arguments) -> None:
    fault_spec = _fault_spec(args)
    has_fault = fault_spec is not None or args.fault_manifest is not None
    has_stale_image = args.stale_image_manifest is not None
    has_intervention = has_fault or has_stale_image
    if args.replay_clean_prefix and not has_intervention:
        raise ValueError("--replay-clean-prefix requires an intervention")
    if has_intervention and not args.paired_clean_dirs:
        raise ValueError("intervention experiments require at least one --paired-clean-dir")
    if args.fault_manifest is not None and not args.fault_manifest.is_file():
        raise FileNotFoundError(f"fault manifest is not a file: {args.fault_manifest}")
    if args.stale_image_manifest is not None and not args.stale_image_manifest.is_file():
        raise FileNotFoundError(
            f"stale-image manifest is not a file: {args.stale_image_manifest}"
        )
    for name, path in {
        "checkpoint": args.checkpoint,
        "OpenVLA root": args.openvla_root,
        "LIBERO root": args.libero_root,
    }.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name} is not a directory: {path}")
    for path in args.paired_clean_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"paired clean directory is not a directory: {path}")
    if any(step < 0 for step in args.evidence_trace_steps):
        raise ValueError("evidence trace steps must be non-negative")
    if any(step >= MAX_STEPS[args.task_suite] for step in args.evidence_trace_steps):
        raise ValueError(
            "evidence trace steps must be below the suite's maximum policy steps"
        )
    if args.evidence_trace_steps and args.evidence_dir is None:
        raise ValueError("--evidence-trace-step requires --evidence-dir")

    if CHECKPOINT_REVISION not in args.checkpoint.resolve().parts:
        raise RuntimeError(
            "checkpoint must be the pinned Hugging Face snapshot at revision "
            f"{CHECKPOINT_REVISION}: {args.checkpoint.resolve()}"
        )

    revisions = {
        "OpenVLA": (git_revision(args.openvla_root), OPENVLA_REVISION),
        "LIBERO": (git_revision(args.libero_root), LIBERO_REVISION),
    }
    for name, (actual, expected) in revisions.items():
        if actual != expected:
            raise RuntimeError(f"{name} revision is {actual}, expected {expected}")

    repositories = {
        "experiment code": Path(__file__).resolve().parents[1],
        "OpenVLA": args.openvla_root,
        "LIBERO": args.libero_root,
    }
    for name, path in repositories.items():
        if git_dirty(path):
            raise RuntimeError(f"{name} has uncommitted changes: {path}")

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            f"Python {platform.python_version()} is unsupported; expected Python 3.10"
        )

    for package, expected in EXPECTED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        if package == "torch":
            matches = actual == expected or actual.startswith(f"{expected}+")
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(f"{package} version is {actual}, expected {expected}")


def _load_runtime(openvla_root: Path, libero_root: Path) -> Runtime:
    sys.path.insert(0, str(openvla_root))
    sys.path.insert(0, str(libero_root))

    import numpy as np
    import torch
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
        save_rollout_video_given_path,
    )
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (
        get_action,
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from experiments.robot.unc_utils import compute_token_uncertainty_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run OpenVLA rollouts")
    if not Path(get_action.__code__.co_filename).resolve().is_relative_to(
        openvla_root.resolve()
    ):
        raise RuntimeError("imported OpenVLA code does not come from --openvla-root")
    if not Path(benchmark.__file__).resolve().is_relative_to(libero_root.resolve()):
        raise RuntimeError("imported LIBERO code does not come from --libero-root")

    return Runtime(
        np=np,
        torch=torch,
        benchmark=benchmark,
        get_action=get_action,
        get_image_resize_size=get_image_resize_size,
        get_libero_dummy_action=get_libero_dummy_action,
        get_libero_env=get_libero_env,
        get_libero_image=get_libero_image,
        get_model=get_model,
        get_processor=get_processor,
        invert_gripper_action=invert_gripper_action,
        normalize_gripper_action=normalize_gripper_action,
        quat2axisangle=quat2axisangle,
        save_video=save_rollout_video_given_path,
        set_seed_everywhere=set_seed_everywhere,
        compute_token_uncertainty_metrics=compute_token_uncertainty_metrics,
    )


def _model_config(args: Arguments) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(args.checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation="flash_attention_2",
        center_crop=True,
        n_samples=1,
        output_logits=True,
        output_attentions=False,
        output_hidden_states=True,
        task_suite_name=args.task_suite,
        unnorm_key=args.task_suite,
    )


def _optional_file_sha256(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def _array_sha256(runtime: Runtime, value: Any) -> str:
    array = runtime.np.ascontiguousarray(runtime.np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("cannot hash an initial state containing Python objects")

    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _nvidia_smi() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _evidence_code_hashes(project_root: Path) -> dict[str, str | None]:
    directory = project_root / "embodied_silent_failures" / "evidence_graph"
    paths = [
        project_root / "embodied_silent_failures" / name
        for name in (
            "faults.py",
            "provenance.py",
            "qwen_artifacts.py",
            "run_openvla.py",
            "score_qwen.py",
            "score_safe.py",
        )
    ]
    paths.extend(sorted(directory.glob("*.py")))
    return {
        str(path.relative_to(project_root)): _optional_file_sha256(path)
        for path in paths
    }


def _checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        resolved = path.resolve()
        entries.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "size": path.stat().st_size,
                "resolved_name": resolved.name,
            }
        )
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(entries),
        "basis": "relative path, byte size, and resolved Hugging Face blob name",
    }


def _write_evidence_saturation(evidence_dir: Path) -> None:
    composition_paths = sorted(evidence_dir.rglob("composition.json"))
    compositions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in composition_paths
    ]
    saturation = summarize_saturation(compositions)
    saturation["composition_files"] = [
        str(path.relative_to(evidence_dir)) for path in composition_paths
    ]
    write_json_atomic(evidence_dir / "saturation.json", saturation)


def _run_metadata(
    args: Arguments,
    runtime: Runtime,
    plan: list[Trial],
    condition: str,
    fault_model: dict[str, Any] | None,
) -> dict[str, Any]:
    package_versions = {
        package: importlib.metadata.version(package) for package in EXPECTED_PACKAGES
    }
    for package in ("mujoco", "numpy", "pandas", "libero"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    config = asdict(args)
    config["checkpoint"] = str(args.checkpoint.resolve())
    config["openvla_root"] = str(args.openvla_root.resolve())
    config["libero_root"] = str(args.libero_root.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["trial_manifest"] = (
        str(args.trial_manifest.resolve()) if args.trial_manifest else None
    )
    config["fault_manifest"] = (
        str(args.fault_manifest.resolve()) if args.fault_manifest else None
    )
    config["stale_image_manifest"] = (
        str(args.stale_image_manifest.resolve()) if args.stale_image_manifest else None
    )
    config["paired_clean_dirs"] = [
        str(path.resolve()) for path in args.paired_clean_dirs
    ]
    config["evidence_dir"] = (
        str(args.evidence_dir.resolve()) if args.evidence_dir else None
    )
    config.pop("resume")

    project_root = Path(__file__).resolve().parents[1]
    repository_states = {
        "experiment_code": git_state(project_root),
        "openvla": git_state(args.openvla_root),
        "libero": git_state(args.libero_root),
    }
    return {
        "schema_version": 1,
        "condition": condition,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "trial_count": len(plan),
        "trial_plan": [
            {**trial.to_dict(), "seed": seed_for_trial(args.seed, trial)}
            for trial in plan
        ],
        "fault_model": fault_model,
        "upstream_revisions": {
            "experiment_code": repository_states["experiment_code"]["revision"],
            "experiment_code_dirty": repository_states["experiment_code"]["dirty"],
            "openvla": OPENVLA_REVISION,
            "libero": LIBERO_REVISION,
            "checkpoint": CHECKPOINT_REVISION,
        },
        "repository_states": repository_states,
        "evidence_graph_code_sha256": _evidence_code_hashes(project_root),
        "checkpoint_manifest": _checkpoint_manifest(args.checkpoint),
        "checkpoint_files": {
            "config.json": _optional_file_sha256(args.checkpoint / "config.json"),
            "dataset_statistics.json": _optional_file_sha256(
                args.checkpoint / "dataset_statistics.json"
            ),
        },
        "trial_manifest_sha256": (
            file_sha256(args.trial_manifest) if args.trial_manifest else None
        ),
        "fault_manifest_sha256": (
            file_sha256(args.fault_manifest) if args.fault_manifest else None
        ),
        "stale_image_manifest_sha256": (
            file_sha256(args.stale_image_manifest)
            if args.stale_image_manifest
            else None
        ),
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "nvidia_gpus": _nvidia_smi(),
            "cuda": runtime.torch.version.cuda,
            "cudnn": runtime.torch.backends.cudnn.version(),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "container_image": CONTAINER_IMAGE,
        },
        "packages": package_versions,
    }


def _prepare_run(args: Arguments, metadata: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "run.json"
    if not path.exists():
        write_json_atomic(path, metadata)
        return
    if not args.resume:
        raise FileExistsError(f"output directory already contains {path.name}: {path}")

    with path.open("r", encoding="utf-8") as file:
        existing = json.load(file)
    if existing.get("configuration") != metadata["configuration"]:
        raise ValueError("resume configuration does not match the existing run")
    if existing.get("trial_plan") != metadata["trial_plan"]:
        raise ValueError("resume trial plan does not match the existing run")

    current_revision = metadata["upstream_revisions"]["experiment_code"]
    initial_revision = existing["upstream_revisions"]["experiment_code"]
    resume_revisions = existing.setdefault("resume_code_revisions", [])
    recorded_revisions = {
        initial_revision,
        *(record["experiment_code"] for record in resume_revisions),
    }
    if current_revision not in recorded_revisions:
        resume_revisions.append(
            {
                "resumed_at": metadata["created_at"],
                "experiment_code": current_revision,
                "experiment_code_dirty": metadata["upstream_revisions"][
                    "experiment_code_dirty"
                ],
                "existing_completion_count": len(
                    list(args.output_dir.glob("*.complete.json"))
                ),
                "existing_exclusion_count": len(
                    list(args.output_dir.glob("*.excluded.json"))
                ),
            }
        )
        write_json_atomic(path, existing)


def _execution_record(metadata: dict[str, Any]) -> dict[str, Any]:
    experiment_code = metadata["repository_states"]["experiment_code"]
    runner_path = "embodied_silent_failures/run_openvla.py"
    return {
        "run_started_at": metadata["created_at"],
        "experiment_code": experiment_code,
        "run_openvla_sha256": metadata["evidence_graph_code_sha256"][runner_path],
    }


def _paired_clean_results(
    directories: list[Path], plan: list[Trial]
) -> tuple[list[Trial], dict[Trial, dict[str, Any]]]:
    indexed: dict[Trial, dict[str, Any]] = {}
    requested = set(plan)
    for directory in directories:
        for path in sorted(directory.glob("*.complete.json")):
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            if result.get("status") != "complete" or result.get("condition") != "clean":
                raise ValueError(f"paired reference is not a completed clean rollout: {path}")
            trial = Trial(
                task_id=int(result["task_id"]),
                episode_index=int(result["episode_index"]),
            )
            if trial not in requested:
                continue
            if trial in indexed:
                fields = (
                    "initial_state_sha256",
                    "trial_seed",
                    "success",
                    "policy_steps",
                )
                if any(indexed[trial].get(key) != result.get(key) for key in fields):
                    raise ValueError(f"paired clean results conflict for {trial}")
                continue
            indexed[trial] = {**result, "_source_dir": str(directory.resolve())}

    missing = [trial for trial in plan if trial not in indexed]
    if missing:
        preview = ", ".join(
            f"{trial.task_id}/{trial.episode_index}" for trial in missing[:5]
        )
        raise FileNotFoundError(f"missing paired clean results for {preview}")

    eligible = [trial for trial in plan if indexed[trial].get("success") is True]
    if not eligible:
        raise ValueError("none of the paired clean rollouts succeeded")
    for trial in eligible:
        clean_steps = int(indexed[trial]["policy_steps"])
        if clean_steps <= 0:
            raise ValueError(f"paired clean rollout has invalid length for {trial}")
    return eligible, indexed


def _numeric_observation(runtime: Runtime, observation: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in observation.items():
        if "image" in key.lower():
            continue
        array = runtime.np.asarray(value)
        if array.dtype.kind not in "biuf" or array.size > 4096:
            continue
        values[key] = array.copy()
    return values


def _stack_observations(runtime: Runtime, history: list[dict[str, Any]]) -> dict[str, Any]:
    common_keys = set(history[0]) if history else set()
    for observation in history[1:]:
        common_keys.intersection_update(observation)
    return {
        key: runtime.np.stack([observation[key] for observation in history])
        for key in sorted(common_keys)
    }


def _python_value(runtime: Runtime, value: Any) -> Any:
    if isinstance(value, runtime.np.generic):
        return value.item()
    return value


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty rollout log")
    temporary_path = _temporary_path(path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_pickle_atomic(path: Path, value: Any) -> None:
    temporary_path = _temporary_path(path)
    try:
        with temporary_path.open("wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _extract_hidden_states(runtime: Runtime, generated: Any) -> Any:
    hidden_states = generated["hidden_states"]
    final_layer = [token_states[-1][0, -1, :] for token_states in hidden_states]
    result = runtime.torch.stack(final_layer, dim=0).detach().cpu()
    if result.ndim != 2 or result.shape[0] != 7:
        raise ValueError(f"unexpected OpenVLA hidden-state shape: {tuple(result.shape)}")
    return result


def _run_trial(
    args: Arguments,
    runtime: Runtime,
    model_config: SimpleNamespace,
    model: Any,
    processor: Any,
    trial: Trial,
    env: Any,
    task_description: str,
    initial_state: Any,
    trial_seed: int,
    initial_state_sha256: str,
    run_condition: str,
    fault_injector: TransientActivationFault | None,
    clean_trace: CleanTrace | None,
    stale_image_spec: StaleImageSpec | None,
    execution: dict[str, Any],
    evidence: RolloutEvidence | None = None,
) -> dict[str, Any]:
    runtime.torch.cuda.reset_peak_memory_stats()
    rollout_started = time.perf_counter()
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(args.wait_steps):
        observation, _, _, _ = env.step(runtime.get_libero_dummy_action("openvla"))

    resize_size = runtime.get_image_resize_size(model_config)
    hidden_history = []
    observation_history: list[dict[str, Any]] = []
    policy_images = []
    replay_images = []
    rows: list[dict[str, Any]] = []
    inference_seconds: list[float] = []
    simulator_seconds: list[float] = []
    success = False
    replay_maximum_error = 0.0
    replayed_steps = 0
    if fault_injector is not None:
        fault_injector.begin_trial(trial_seed)
    intervention_step = None
    if fault_injector is not None:
        intervention_step = fault_injector.spec.policy_step
    elif stale_image_spec is not None:
        intervention_step = stale_image_spec.policy_step
    image_intervention_record = None

    for policy_step in range(MAX_STEPS[args.task_suite]):
        if clean_trace is not None and intervention_step is not None and policy_step <= intervention_step:
            error = observation_error(
                runtime.np, clean_trace, observation, policy_step
            )
            replay_maximum_error = max(replay_maximum_error, error)
            if error > REPLAY_OBSERVATION_TOLERANCE:
                raise CounterfactualReplayDivergence(policy_step, error)

        if clean_trace is not None and intervention_step is not None and policy_step < intervention_step:
            image = runtime.get_libero_image(observation, resize_size)
            policy_images.append(image.copy())
            if args.save_video:
                replay_images.append(image)
            hidden_history.append(
                runtime.torch.as_tensor(clean_trace.hidden_states[policy_step])
                .detach()
                .cpu()
            )
            observation_history.append(_numeric_observation(runtime, observation))
            row = dict(clean_trace.rows[policy_step])
            row["timing/inference_seconds"] = 0.0
            row["fault/injected"] = False
            action = replay_action(runtime.np, clean_trace, policy_step)

            if evidence is not None:
                evidence.begin_step(
                    policy_step,
                    observation,
                    image,
                    image,
                    policy_step,
                    None,
                    policy_replayed=True,
                )
                action = evidence.replayed_evidence(
                    policy_step,
                    action,
                    runtime.torch.as_tensor(clean_trace.hidden_states[policy_step, -1, :]),
                )

            simulator_started = time.perf_counter()
            if evidence is not None:
                observation, reward, done, _ = evidence.environment_step(
                    env, action, policy_step, policy_replayed=True
                )
            else:
                observation, reward, done, _ = env.step(action.tolist())
            simulator_seconds.append(time.perf_counter() - simulator_started)
            row["timing/simulator_seconds"] = simulator_seconds[-1]
            row["environment/reward"] = reward
            row["environment/done"] = bool(done)
            rows.append(row)
            if evidence is not None:
                evidence.finish_step(
                    policy_step,
                    fault_applied=False,
                    reward=reward,
                    done=bool(done),
                )
            replayed_steps += 1
            if done:
                raise CounterfactualReplayTerminated(
                    policy_step, intervention_step
                )
            continue

        image = runtime.get_libero_image(observation, resize_size)
        policy_image = image
        if stale_image_spec is not None and policy_step == stale_image_spec.policy_step:
            if stale_image_spec.source_policy_step != (
                stale_image_spec.policy_step - stale_image_spec.image_lag
            ):
                raise RuntimeError("stale-image source policy step does not match the lag")
            if stale_image_spec.source_policy_step >= len(policy_images):
                raise RuntimeError(
                    f"stale-image source step {stale_image_spec.source_policy_step} "
                    f"is unavailable at policy step {policy_step}"
                )
            if args.image_input_mode == "stale":
                policy_image = policy_images[
                    stale_image_spec.source_policy_step
                ].copy()
            image_intervention_record = _image_intervention_record(
                stale_image_spec, args.image_input_mode, trial_seed
            )
        state = runtime.np.concatenate(
            (
                observation["robot0_eef_pos"],
                runtime.quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        )
        policy_observation = {"full_image": policy_image, "state": state}

        intervention = None
        source_step = policy_step
        if stale_image_spec is not None and policy_step == stale_image_spec.policy_step:
            intervention = image_intervention_record
            if args.image_input_mode == "stale":
                source_step = stale_image_spec.source_policy_step
        if evidence is not None:
            evidence.begin_step(
                policy_step,
                observation,
                image,
                policy_image,
                source_step,
                intervention,
            )

        runtime.torch.cuda.synchronize()
        inference_started = time.perf_counter()
        fault_context = (
            fault_injector.inference(policy_step)
            if fault_injector is not None
            else nullcontext()
        )
        evidence_model = (
            evidence.policy_model(model, policy_step)
            if evidence is not None
            else model
        )
        evidence_processor = (
            evidence.processor(
                processor, policy_image, task_description, policy_step
            )
            if evidence is not None
            else processor
        )
        with runtime.torch.inference_mode(), fault_context:
            raw_actions, generated = runtime.get_action(
                model_config,
                evidence_model,
                policy_observation,
                task_description,
                processor=evidence_processor,
                n_samples=1,
            )
        runtime.torch.cuda.synchronize()
        inference_seconds.append(time.perf_counter() - inference_started)

        raw_action = runtime.np.asarray(raw_actions).copy()
        if raw_action.shape != (7,):
            raise ValueError(f"unexpected OpenVLA action shape: {raw_action.shape}")
        if evidence is not None:
            evidence.policy_outputs(
                model,
                generated,
                raw_actions,
                model_config.unnorm_key,
                policy_step,
                runtime.torch,
            )
            action = evidence.command(runtime, raw_action, policy_step)
        else:
            action = runtime.normalize_gripper_action(raw_action.copy(), binarize=True)
            action = runtime.invert_gripper_action(action)

        hidden_history.append(_extract_hidden_states(runtime, generated))
        observation_history.append(_numeric_observation(runtime, observation))
        policy_images.append(image.copy())
        if args.save_video:
            replay_images.append(image)

        metrics = runtime.compute_token_uncertainty_metrics(generated, model)
        row: dict[str, Any] = {
            "action/timestep": policy_step,
            "action/dx": action[0],
            "action/dy": action[1],
            "action/dz": action[2],
            "action/droll": action[3],
            "action/dpitch": action[4],
            "action/dyaw": action[5],
            "action/dgripper": action[6],
            "raw_action/dx": raw_action[0],
            "raw_action/dy": raw_action[1],
            "raw_action/dz": raw_action[2],
            "raw_action/droll": raw_action[3],
            "raw_action/dpitch": raw_action[4],
            "raw_action/dyaw": raw_action[5],
            "raw_action/dgripper": raw_action[6],
            "robot/eef_x": observation["robot0_eef_pos"][0],
            "robot/eef_y": observation["robot0_eef_pos"][1],
            "robot/eef_z": observation["robot0_eef_pos"][2],
            "timing/inference_seconds": inference_seconds[-1],
        }
        row.update({f"action/{key}": value for key, value in metrics.items()})
        if fault_injector is not None:
            record = fault_injector.record
            row["fault/injected"] = bool(
                record is not None and record["policy_step"] == policy_step
            )
        elif stale_image_spec is not None:
            row["fault/injected"] = _image_fault_applied(
                stale_image_spec, args.image_input_mode, policy_step
            )

        simulator_started = time.perf_counter()
        if evidence is not None:
            observation, reward, done, _ = evidence.environment_step(
                env, action, policy_step
            )
        else:
            observation, reward, done, _ = env.step(action.tolist())
        simulator_seconds.append(time.perf_counter() - simulator_started)
        row["timing/simulator_seconds"] = simulator_seconds[-1]
        row["environment/reward"] = reward
        row["environment/done"] = bool(done)
        rows.append({key: _python_value(runtime, value) for key, value in row.items()})
        if evidence is not None:
            evidence.finish_step(
                policy_step,
                fault_applied=bool(row.get("fault/injected", False)),
                reward=reward,
                done=bool(done),
            )

        if done:
            success = True
            break

    rollout_seconds = time.perf_counter() - rollout_started
    fault_record = None
    if fault_injector is not None:
        fault_record = fault_injector.require_injected()
    elif stale_image_spec is not None:
        if image_intervention_record is None:
            raise RuntimeError("image intervention was never applied")
        fault_record = image_intervention_record
    condition = run_condition if fault_record else "clean"
    evidence_result = None

    stem = safe_stem(trial, success)
    csv_path = args.output_dir / f"{stem}.csv"
    pickle_path = args.output_dir / f"{stem}.pkl"
    video_path = args.output_dir / f"{stem}.mp4"

    artifact_started = time.perf_counter()
    _write_csv_atomic(csv_path, rows)
    if args.save_video:
        temporary_video = _temporary_path(video_path)
        try:
            runtime.save_video(replay_images, temporary_video)
            temporary_video.replace(video_path)
        finally:
            temporary_video.unlink(missing_ok=True)

    hidden_states = runtime.torch.stack(hidden_history, dim=0)
    _write_pickle_atomic(
        pickle_path,
        {
            "hidden_states": hidden_states,
            "observations": _stack_observations(runtime, observation_history),
            "condition": condition,
            "fault": fault_record,
            "task_suite_name": args.task_suite,
            "task_id": trial.task_id,
            "task_description": task_description,
            "episode_idx": trial.episode_index,
            "episode_success": success,
            "trial_seed": trial_seed,
            "initial_state_sha256": initial_state_sha256,
            "mp4_path": str(video_path) if args.save_video else None,
        },
    )
    artifact_seconds = time.perf_counter() - artifact_started
    if evidence is not None:
        evidence_result = evidence.close(
            success=success,
            policy_steps=len(rows),
            fault=fault_record,
        )
        evidence_result["directory_relative_to_run"] = os.path.relpath(
            evidence.output_dir, args.output_dir
        )

    result = {
        "schema_version": 1,
        "status": "complete",
        "condition": condition,
        "task_suite_name": args.task_suite,
        "task_id": trial.task_id,
        "task_description": task_description,
        "episode_index": trial.episode_index,
        "trial_seed": trial_seed,
        "initial_state_sha256": initial_state_sha256,
        "success": success,
        "policy_steps": len(rows),
        "maximum_policy_steps": MAX_STEPS[args.task_suite],
        "rollout_seconds": rollout_seconds,
        "inference_seconds": sum(inference_seconds),
        "mean_inference_seconds": sum(inference_seconds) / len(inference_seconds),
        "simulator_seconds": sum(simulator_seconds),
        "artifact_seconds": artifact_seconds,
        "peak_cuda_memory_bytes": runtime.torch.cuda.max_memory_allocated(),
        "execution": execution,
        "fault": fault_record,
        "counterfactual_replay": (
            {
                "enabled": True,
                "replayed_policy_steps": replayed_steps,
                "policy_inferences": len(inference_seconds),
                "maximum_numeric_observation_error": replay_maximum_error,
                "observation_tolerance": REPLAY_OBSERVATION_TOLERANCE,
                "clean_source_directory": str(clean_trace.source_dir),
            }
            if clean_trace is not None
            else {"enabled": False}
        ),
        "evidence_graph": evidence_result,
        "files": {
            "csv": csv_path.name,
            "pickle": pickle_path.name,
            "video": video_path.name if args.save_video else None,
        },
    }
    write_json_atomic(completion_path(args.output_dir, trial), result)
    return result


def main() -> None:
    args = _parse_arguments()
    static_fault_spec = _fault_spec(args)
    _validate_inputs(args)

    manifest = None
    stale_manifest = None
    if args.fault_manifest is not None:
        if args.trial_manifest is not None:
            raise ValueError("--fault-manifest already defines the trial plan")
        manifest = load_fault_manifest(args.fault_manifest)
        fault_specs = manifest.specs
        plan = sorted(fault_specs)
        stale_specs = {}
    elif args.stale_image_manifest is not None:
        if args.trial_manifest is not None:
            raise ValueError("--stale-image-manifest already defines the trial plan")
        stale_manifest = load_stale_image_manifest(args.stale_image_manifest)
        stale_specs = stale_manifest.specs
        fault_specs = {}
        plan = sorted(stale_specs)
    elif args.trial_manifest is None:
        task_ids = parse_task_ids(args.task_ids)
        plan = build_trial_plan(
            task_ids,
            args.episode_start,
            args.episode_stop,
            args.episode_stride,
        )
    else:
        plan = load_trial_manifest(args.trial_manifest)
    task_ids = sorted({trial.task_id for trial in plan})
    if static_fault_spec is not None:
        fault_specs = {trial: static_fault_spec for trial in plan}
        stale_specs = {}
    elif manifest is None and stale_manifest is None:
        fault_specs = {}
        stale_specs = {}

    paired_clean: dict[Trial, dict[str, Any]] = {}
    if fault_specs or stale_specs:
        requested_count = len(plan)
        plan, paired_clean = _paired_clean_results(args.paired_clean_dirs, plan)
        too_short = []
        for trial in plan:
            clean_steps = int(paired_clean[trial]["policy_steps"])
            if trial in fault_specs and clean_steps <= fault_specs[trial].policy_step:
                too_short.append(trial)
            if trial in stale_specs and clean_steps <= stale_specs[trial].policy_step:
                too_short.append(trial)
        if too_short:
            preview = ", ".join(
                f"{trial.task_id}/{trial.episode_index}" for trial in too_short[:5]
            )
            raise ValueError(
                f"fault policy step is not before paired clean completion for {preview}"
            )
        print(
            f"fault eligibility: {len(plan)} clean successes selected from "
            f"{requested_count} paired rollouts"
        )
    if manifest is not None:
        fault_model = {
            "kind": "per_trial_fault_manifest",
            "selection_basis": manifest.selection_basis,
            "trial_count": len(plan),
        }
        run_condition = "activation_fault"
    elif stale_manifest is not None:
        input_mode = args.image_input_mode
        fault_model = {
            "kind": (
                "per_trial_stale_image_manifest"
                if input_mode == "stale"
                else "per_trial_current_image_control_manifest"
            ),
            "selection_basis": stale_manifest.selection_basis,
            "trial_count": len(plan),
        }
        run_condition = (
            "stale_image" if input_mode == "stale" else "current_image_control"
        )
    elif static_fault_spec is not None:
        fault_model = static_fault_spec.to_dict()
        run_condition = "activation_fault"
    else:
        fault_model = None
        run_condition = "clean"

    runtime = _load_runtime(args.openvla_root, args.libero_root)
    metadata = _run_metadata(args, runtime, plan, run_condition, fault_model)
    _prepare_run(args, metadata)
    execution = _execution_record(metadata)

    benchmark_class = runtime.benchmark.get_benchmark_dict()[args.task_suite]
    task_suite = benchmark_class()
    if any(task_id >= task_suite.n_tasks for task_id in task_ids):
        raise ValueError(
            f"{args.task_suite} has {task_suite.n_tasks} tasks, requested {task_ids}"
        )

    initial_states_by_task = {
        task_id: task_suite.get_task_init_states(task_id) for task_id in task_ids
    }
    for trial in plan:
        state_count = len(initial_states_by_task[trial.task_id])
        if trial.episode_index >= state_count:
            raise IndexError(
                f"task {trial.task_id} has {state_count} initial states, "
                f"but episode {trial.episode_index} was requested"
            )

    indexed_plan = list(enumerate(plan, start=1))
    pending: list[tuple[int, Trial]] = []
    for index, trial in indexed_plan:
        terminal_state = prepare_trial(args.output_dir, trial, args.resume)
        if terminal_state is not None:
            print(
                f"[{index}/{len(plan)}] skipping {terminal_state} task "
                f"{trial.task_id}, episode {trial.episode_index}"
            )
        else:
            pending.append((index, trial))

    if not pending:
        if args.evidence_dir is not None:
            _write_evidence_saturation(args.evidence_dir)
        print("run complete: no new trials")
        return

    model_config = _model_config(args)
    runtime.set_seed_everywhere(args.seed)
    model = runtime.get_model(model_config)
    model.eval()
    processor = runtime.get_processor(model_config)
    if args.task_suite not in model.norm_stats:
        alternate_key = f"{args.task_suite}_no_noops"
        if alternate_key not in model.norm_stats:
            raise KeyError(
                f"checkpoint has no normalization statistics for {args.task_suite}"
            )
        model_config.unnorm_key = alternate_key

    completed = 0
    excluded = 0
    successes = 0
    for task_id in task_ids:
        task_trials = [item for item in pending if item[1].task_id == task_id]
        if not task_trials:
            continue

        task = task_suite.get_task(task_id)
        env, task_description = runtime.get_libero_env(
            task, "openvla", resolution=256
        )
        try:
            for index, trial in task_trials:
                trial_seed = seed_for_trial(args.seed, trial)
                runtime.set_seed_everywhere(trial_seed)
                initial_state = initial_states_by_task[task_id][trial.episode_index]
                initial_state_sha256 = _array_sha256(runtime, initial_state)
                clean_trace = None
                trial_fault_spec = fault_specs.get(trial)
                trial_stale_image_spec = stale_specs.get(trial)
                if trial_fault_spec is not None or trial_stale_image_spec is not None:
                    clean_result = paired_clean[trial]
                    if clean_result.get("initial_state_sha256") != initial_state_sha256:
                        raise ValueError(
                            f"paired clean initial state does not match task "
                            f"{trial.task_id}, episode {trial.episode_index}"
                        )
                    if clean_result.get("trial_seed") != trial_seed:
                        raise ValueError(
                            f"paired clean seed does not match task {trial.task_id}, "
                            f"episode {trial.episode_index}"
                        )
                    if args.replay_clean_prefix or trial_stale_image_spec is not None:
                        clean_trace = load_clean_trace(clean_result)
                print(
                    f"[{index}/{len(plan)}] running task {trial.task_id}, "
                    f"episode {trial.episode_index}, seed {trial_seed}"
                )
                fault_injector = None
                if trial_fault_spec is not None:
                    fault_injector = TransientActivationFault(
                        runtime.torch, trial_fault_spec
                    )
                    fault_injector.install(model)
                try:
                    try:
                        evidence = None
                        if args.evidence_dir is not None:
                            traced_steps = set(args.evidence_trace_steps)
                            if not traced_steps:
                                intervention = (
                                    trial_fault_spec.policy_step
                                    if trial_fault_spec is not None
                                    else (
                                        trial_stale_image_spec.policy_step
                                        if trial_stale_image_spec is not None
                                        else 0
                                    )
                                )
                                traced_steps.add(intervention)
                            evidence_output = (
                                args.evidence_dir
                                / f"task{trial.task_id}--ep{trial.episode_index}"
                            )
                            prepare_evidence_output(evidence_output)
                            evidence = RolloutEvidence(
                                evidence_output,
                                {
                                    "schema_version": 2,
                                    "scope": "actual_openvla_rollout_with_selected_operator_traces",
                                    "condition": run_condition,
                                    "task_suite": args.task_suite,
                                    "task_id": trial.task_id,
                                    "episode_index": trial.episode_index,
                                    "trial_seed": trial_seed,
                                    "traced_steps": sorted(traced_steps),
                                    "upstream_revisions": metadata["upstream_revisions"],
                                    "repository_states": metadata["repository_states"],
                                    "evidence_graph_code_sha256": metadata[
                                        "evidence_graph_code_sha256"
                                    ],
                                    "checkpoint_manifest": metadata[
                                        "checkpoint_manifest"
                                    ],
                                    "checkpoint_files": metadata["checkpoint_files"],
                                },
                                traced_steps,
                            )
                            if fault_injector is not None:
                                fault_injector.set_observer(
                                    evidence.activation_fault_observer
                                )
                        result = _run_trial(
                            args,
                            runtime,
                            model_config,
                            model,
                            processor,
                            trial,
                            env,
                            task_description,
                            initial_state,
                            trial_seed,
                            initial_state_sha256,
                            run_condition,
                            fault_injector,
                            clean_trace,
                            trial_stale_image_spec,
                            execution,
                            evidence,
                        )
                    except CounterfactualReplayInvalid as error:
                        if evidence is not None:
                            evidence.abort(error.reason)
                        intervention_step = (
                            trial_fault_spec.policy_step
                            if trial_fault_spec is not None
                            else trial_stale_image_spec.policy_step
                        )
                        write_json_atomic(
                            exclusion_path(args.output_dir, trial),
                            {
                                "schema_version": 1,
                                "status": "excluded",
                                "reason": error.reason,
                                "condition": run_condition,
                                "task_suite_name": args.task_suite,
                                "task_id": trial.task_id,
                                "episode_index": trial.episode_index,
                                "trial_seed": trial_seed,
                                "initial_state_sha256": initial_state_sha256,
                                "intervention_policy_step": intervention_step,
                                "last_replay_policy_step": error.policy_step,
                                "maximum_numeric_observation_error": getattr(
                                    error, "error", None
                                ),
                                "observation_tolerance": REPLAY_OBSERVATION_TOLERANCE,
                                "clean_source_directory": str(clean_trace.source_dir),
                                "execution": execution,
                            },
                        )
                        excluded += 1
                        print(
                            f"excluded task {trial.task_id}, episode "
                            f"{trial.episode_index}: {error}"
                        )
                        continue
                    except BaseException:
                        if evidence is not None:
                            evidence.abort("rollout_execution_error")
                        raise
                finally:
                    if fault_injector is not None:
                        fault_injector.close()
                completed += 1
                successes += int(result["success"])
                print(
                    f"completed in {result['rollout_seconds']:.1f}s: "
                    f"success={result['success']}, steps={result['policy_steps']}, "
                    f"mean inference={result['mean_inference_seconds']:.3f}s"
                )
        finally:
            env.close()

    if completed or excluded:
        success_summary = (
            f", {successes} successes ({successes / completed:.1%})"
            if completed
            else ""
        )
        print(
            f"run complete: {completed} new trials{success_summary}, "
            f"{excluded} excluded"
        )
    if args.evidence_dir is not None:
        _write_evidence_saturation(args.evidence_dir)


if __name__ == "__main__":
    main()
