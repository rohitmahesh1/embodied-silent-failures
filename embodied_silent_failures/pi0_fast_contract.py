from __future__ import annotations


SAFE_OPENPI_REVISION = "9c99ed53f6a0c9be93a1c63cee5792620777d96b"
SAFE_OPENPI_PARENT_REVISION = "29068dd2741d5a45db41479e0b5d45ebac774fa6"
SAFE_REVISION = "b6036abe07b2b2bb9996afb2c07f13d6a9f507c0"
LIBERO_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
FAST_TOKENIZER_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
POLICY_CONFIG = "pi0_fast_libero"
CHECKPOINT = "s3://openpi-assets/checkpoints/pi0_fast_libero"
PROTOCOL_VERSION = 1
ACTION_HORIZON = 10
ACTION_DIMENSION = 7
FEATURE_DIMENSION = 2048
ACTION_TOKEN_START = 254_976
ACTION_TOKEN_STOP = 257_024
MAX_DECODING_STEPS = 256
PALIGEMMA_EOS_TOKEN = 1
PARITY_ACTION_ATOL = 1e-6
FEATURE_SOURCE_DTYPE = "bfloat16"
FEATURE_TRANSPORT_ENCODING = "bfloat16_as_uint16_bits"
DEFAULT_REPLAN_STEPS = 5
DEFAULT_WAIT_STEPS = 10
IMAGE_SIZE = 224
ENVIRONMENT_RESOLUTION = 256

# SAFE openpi commit 9c99ed5, examples/libero/main.py::eval_libero, assigns
# these suite limits from the longest training demonstrations and runs this many
# policy-controlled steps after the ten-step stabilization period.
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
