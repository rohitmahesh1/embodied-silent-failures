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

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.pi05_contract import (
    CHECKPOINT,
    OPENPI_REVISION,
    POLICY_CONFIG,
    SAFE_OPENPI_REVISION,
    SAFE_REVISION,
)
from embodied_silent_failures.pi05_policy import create_evidence_policy
from embodied_silent_failures.provenance import file_sha256, git_dirty, git_revision


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the pinned pi0.5 LIBERO policy with SAFE evidence."
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


def _checkpoint_manifest(root: Path) -> dict[str, Any]:
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
        raise ValueError(f"checkpoint contains no files: {root}")
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
    if git_revision(args.openpi_root) != OPENPI_REVISION:
        raise RuntimeError(f"OpenPI must be at {OPENPI_REVISION}: {args.openpi_root}")
    if git_dirty(args.openpi_root):
        raise RuntimeError(f"OpenPI has uncommitted changes: {args.openpi_root}")

    sys.path.insert(0, str(args.openpi_root / "src"))
    sys.path.insert(0, str(args.openpi_root / "packages" / "openpi-client" / "src"))

    import jax
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import download
    from openpi.training import config as training_config

    if (
        not Path(policy_config.__file__)
        .resolve()
        .is_relative_to(args.openpi_root.resolve())
    ):
        raise RuntimeError("imported OpenPI does not come from --openpi-root")

    checkpoint_path = Path(download.maybe_download(args.checkpoint)).resolve()
    trained = policy_config.create_trained_policy(
        training_config.get_config(args.config), checkpoint_path
    )
    policy = create_evidence_policy(trained)
    project_root = Path(__file__).resolve().parents[1]
    metadata = {
        **policy.metadata,
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_manifest": _checkpoint_manifest(checkpoint_path),
        },
        "provenance": {
            "openpi_revision": OPENPI_REVISION,
            "safe_openpi_revision": SAFE_OPENPI_REVISION,
            "safe_revision": SAFE_REVISION,
            "experiment_revision": git_revision(project_root),
            "experiment_dirty": git_dirty(project_root),
            "files": {
                "pi05_policy.py": file_sha256(
                    project_root / "embodied_silent_failures" / "pi05_policy.py"
                ),
                "serve_pi05.py": file_sha256(Path(__file__)),
                "openpi_pi0.py": file_sha256(
                    args.openpi_root / "src" / "openpi" / "models" / "pi0.py"
                ),
                "openpi_policy.py": file_sha256(
                    args.openpi_root / "src" / "openpi" / "policies" / "policy.py"
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
            ("jax", "jaxlib", "flax", "numpy", "openpi", "openpi-client")
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
