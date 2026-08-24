from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi0_fast_contract import (
    CHECKPOINT,
    FAST_TOKENIZER_REVISION,
    POLICY_CONFIG,
    SAFE_OPENPI_REVISION,
    SAFE_REVISION,
)
from embodied_silent_failures.pi0_fast_policy import create_evidence_policy
from embodied_silent_failures.provenance import file_sha256, git_dirty, git_revision


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the pinned pi0-FAST LIBERO policy with SAFE evidence."
    )
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--config", default=POLICY_CONFIG)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return args


def _file_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError(f"artifact contains no files: {root}")
    return {"root": str(root), "files": files}


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_record() -> list[str]:
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


def main() -> None:
    args = _arguments()
    if not args.openpi_root.is_dir():
        raise FileNotFoundError(f"OpenPI root is not a directory: {args.openpi_root}")
    if git_revision(args.openpi_root) != SAFE_OPENPI_REVISION:
        raise RuntimeError(
            f"SAFE OpenPI must be at {SAFE_OPENPI_REVISION}: {args.openpi_root}"
        )
    if git_dirty(args.openpi_root):
        raise RuntimeError(f"SAFE OpenPI has uncommitted changes: {args.openpi_root}")

    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root / "packages" / "openpi-client" / "src"))

    import jax
    import jax.numpy as jnp
    from huggingface_hub import snapshot_download
    from openpi import transforms
    from openpi.models import model as model_module
    from openpi.models import tokenizer as tokenizer_module
    from openpi.serving import websocket_policy_server
    from openpi.shared import download
    from openpi.training import checkpoints
    from openpi.training import config as training_config

    if not Path(training_config.__file__).resolve().is_relative_to(
        args.openpi_root.resolve()
    ):
        raise RuntimeError("imported OpenPI does not come from --openpi-root")

    checkpoint_path = Path(download.maybe_download(args.checkpoint)).resolve()
    paligemma_tokenizer = Path(
        download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
    ).resolve()
    tokenizer_snapshot = Path(
        snapshot_download(
            "physical-intelligence/fast",
            revision=FAST_TOKENIZER_REVISION,
        )
    ).resolve()
    train_config = training_config.get_config(args.config)
    model = train_config.model.load(
        model_module.restore_params(checkpoint_path / "params", dtype=jnp.bfloat16)
    )

    original_from_pretrained = tokenizer_module.AutoProcessor.from_pretrained

    def pinned_processor(path: str, *values: Any, **kwargs: Any) -> Any:
        if path != "physical-intelligence/fast":
            raise ValueError(f"unexpected FAST tokenizer source: {path}")
        kwargs.pop("revision", None)
        return original_from_pretrained(
            str(tokenizer_snapshot), *values, **kwargs
        )

    # SAFE openpi commit 9c99ed5, src/openpi/models/tokenizer.py::FASTTokenizer,
    # requests mutable Hugging Face remote code without a revision. The repository
    # has not changed since this January 2025 revision; binding the constructor to
    # its immutable snapshot removes the remaining network-time dependency.
    with mock.patch.object(
        tokenizer_module.AutoProcessor,
        "from_pretrained",
        side_effect=pinned_processor,
    ):
        data_config = train_config.data.create(
            train_config.assets_dirs, train_config.model
        )

    if data_config.asset_id is None:
        raise ValueError("pi0-FAST LIBERO config has no normalization asset ID")
    norm_stats = checkpoints.load_norm_stats(
        checkpoint_path / "assets", data_config.asset_id
    )
    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.data_transforms.outputs,
    ]
    policy = create_evidence_policy(
        model,
        input_transforms,
        output_transforms,
        train_config.policy_metadata,
    )

    project_root = Path(__file__).resolve().parents[1]
    metadata = {
        **policy.metadata,
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_manifest": _file_manifest(checkpoint_path),
            "fast_tokenizer_revision": FAST_TOKENIZER_REVISION,
            "fast_tokenizer_manifest": _file_manifest(tokenizer_snapshot),
            "paligemma_tokenizer": {
                "path": str(paligemma_tokenizer),
                "bytes": paligemma_tokenizer.stat().st_size,
                "sha256": file_sha256(paligemma_tokenizer),
            },
        },
        "provenance": {
            "safe_openpi_revision": SAFE_OPENPI_REVISION,
            "safe_revision": SAFE_REVISION,
            "experiment_revision": git_revision(project_root),
            "experiment_dirty": git_dirty(project_root),
            "files": {
                "pi0_fast_policy.py": file_sha256(
                    project_root / "embodied_silent_failures" / "pi0_fast_policy.py"
                ),
                "serve_pi0_fast.py": file_sha256(Path(__file__)),
                "openpi_pi0_fast.py": file_sha256(
                    args.openpi_root / "src" / "openpi" / "models" / "pi0_fast.py"
                ),
                "openpi_gemma_fast.py": file_sha256(
                    args.openpi_root / "src" / "openpi" / "models" / "gemma_fast.py"
                ),
                "openpi_policy_config.py": file_sha256(
                    args.openpi_root
                    / "src"
                    / "openpi"
                    / "policies"
                    / "policy_config.py"
                ),
                "openpi_tokenizer.py": file_sha256(
                    args.openpi_root / "src" / "openpi" / "models" / "tokenizer.py"
                ),
            },
        },
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "nvidia_gpus": _gpu_record(),
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        },
        "packages": _package_versions(
            (
                "jax",
                "jaxlib",
                "flax",
                "numpy",
                "openpi",
                "openpi-client",
                "transformers",
                "huggingface-hub",
            )
        ),
    }
    write_json_atomic(args.metadata_output, metadata)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    print(json.dumps({"state": "ready", "port": args.port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
