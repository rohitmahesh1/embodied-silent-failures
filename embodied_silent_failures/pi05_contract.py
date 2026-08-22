from __future__ import annotations

import hashlib

from embodied_silent_failures.plan import Trial


OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
SAFE_OPENPI_REVISION = "9c99ed53f6a0c9be93a1c63cee5792620777d96b"
SAFE_REVISION = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
POLICY_CONFIG = "pi05_libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
PROTOCOL_VERSION = 1
ACTION_HORIZON = 10
ACTION_DIMENSION = 32
LIBERO_ACTION_DIMENSION = 7
DIFFUSION_STEPS = 10
DEFAULT_REPLAN_STEPS = 5
DEFAULT_WAIT_STEPS = 10
IMAGE_SIZE = 224
ENVIRONMENT_RESOLUTION = 256

# Physical Intelligence openpi commit 15a9616, examples/libero/main.py::eval_libero,
# assigns these suite limits from the longest training demonstration and evaluates
# for that many policy-controlled environment steps after the stabilization wait.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def validate_replan_steps(replan_steps: int) -> int:
    if not 1 <= replan_steps <= ACTION_HORIZON:
        raise ValueError(
            f"replan steps must be between 1 and {ACTION_HORIZON}, got {replan_steps}"
        )
    return replan_steps


def decision_noise_seed(base_seed: int, trial: Trial, decision_index: int) -> int:
    if base_seed < 0 or decision_index < 0:
        raise ValueError("base seed and decision index must be non-negative")
    identity = (
        f"pi05-noise-v1:{base_seed}:{trial.task_id}:"
        f"{trial.episode_index}:{decision_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def decision_for_step(environment_step: int, replan_steps: int) -> tuple[int, int]:
    if environment_step < 0:
        raise ValueError("environment step must be non-negative")
    validate_replan_steps(replan_steps)
    return divmod(environment_step, replan_steps)
