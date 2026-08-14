import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def source_file_record(value: Any) -> dict[str, Any]:
    path_value = inspect.getsourcefile(type(value))
    if path_value is None:
        raise RuntimeError(f"cannot locate source for {type(value).__qualname__}")
    path = Path(path_value).resolve()
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "path": str(path),
        "sha256": file_sha256(path),
    }


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def git_state(path: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    digest.update(status.encode("utf-8"))
    digest.update(diff)
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        untracked = path / relative
        if untracked.is_file():
            digest.update(relative.encode("utf-8"))
            with untracked.open("rb") as file:
                for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "revision": git_revision(path),
        "dirty": bool(status.strip()),
        "worktree_sha256": digest.hexdigest(),
    }
