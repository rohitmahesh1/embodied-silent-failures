import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from embodied_silent_failures.provenance import git_dirty, git_revision


OPENVLA_REVISION = "300dce26d44f407c725695d16cd445755c92cbd1"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_REVISION = "80970322773f81baa2e22fe495d0487b93a05cfa"


def validate_pinned_runtime(
    checkpoint: Path,
    openvla_root: Path,
    libero_root: Path,
    *,
    project_root: Path | None = None,
) -> None:
    for name, path in (
        ("checkpoint", checkpoint),
        ("OpenVLA root", openvla_root),
        ("LIBERO root", libero_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} is not a directory: {path}")
    if CHECKPOINT_REVISION not in checkpoint.resolve().parts:
        raise RuntimeError(
            "checkpoint is not the pinned OpenVLA snapshot at revision "
            f"{CHECKPOINT_REVISION}: {checkpoint.resolve()}"
        )
    for name, root, expected in (
        ("OpenVLA", openvla_root, OPENVLA_REVISION),
        ("LIBERO", libero_root, LIBERO_REVISION),
    ):
        actual = git_revision(root)
        if actual != expected:
            raise RuntimeError(f"{name} revision is {actual}, expected {expected}")
        if git_dirty(root):
            raise RuntimeError(f"{name} has uncommitted changes: {root}")
    if project_root is not None and git_dirty(project_root):
        raise RuntimeError(f"experiment code has uncommitted changes: {project_root}")


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


def load_runtime(openvla_root: Path, libero_root: Path) -> Runtime:
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


def model_config(checkpoint: Path, task_suite: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        attn_implementation="flash_attention_2",
        center_crop=True,
        n_samples=1,
        output_logits=True,
        output_attentions=False,
        output_hidden_states=True,
        task_suite_name=task_suite,
        unnorm_key=task_suite,
    )


def array_sha256(runtime: Runtime, value: Any) -> str:
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
