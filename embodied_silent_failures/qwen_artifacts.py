import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic


SOURCE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class TrialSource:
    source: str
    run_dir: Path
    run_path: Path
    run_sha256: str
    task_id: int
    episode_index: int
    completion_path: Path
    video_path: Path
    completion: dict[str, Any]
    completion_sha256: str
    video_sha256: str

    @property
    def key(self) -> str:
        return f"{self.source}--task{self.task_id}--ep{self.episode_index}"


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


def load_trial_manifest(path: Path) -> tuple[dict[str, Any], list[TrialSource]]:
    manifest = load_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Qwen trial manifest schema_version must be 1")
    basis = manifest.get("selection_basis")
    if not isinstance(basis, str) or not basis.startswith(
        ("paper:", "protocol:", "observed:")
    ):
        raise ValueError("Qwen trial manifest must state a provenance selection_basis")
    entries = manifest.get("trials")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Qwen trial manifest must contain at least one trial")

    trials = []
    keys = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each Qwen trial must be a JSON object")
        source = entry.get("source")
        if not isinstance(source, str) or SOURCE_NAME_PATTERN.fullmatch(source) is None:
            raise ValueError("trial source must be a lowercase file-safe name")
        run_dir_value = entry.get("run_dir")
        if not isinstance(run_dir_value, str):
            raise ValueError("trial run_dir must be a path string")
        task_id = int(entry["task_id"])
        episode_index = int(entry["episode_index"])
        key = (source, task_id, episode_index)
        if key in keys:
            raise ValueError(f"duplicate Qwen trial: {key}")
        keys.add(key)

        run_dir = _resolve_path(run_dir_value, path.parent)
        run_path = run_dir / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"rollout run record is missing: {run_path}")
        load_json(run_path)
        completion_path = run_dir / f"task{task_id}--ep{episode_index}.complete.json"
        if not completion_path.is_file():
            raise FileNotFoundError(f"completion record is missing: {completion_path}")
        completion = load_json(completion_path)
        if completion.get("status") != "complete":
            raise ValueError(f"rollout is not complete: {completion_path}")
        if int(completion.get("task_id", -1)) != task_id:
            raise ValueError(f"task ID disagrees with manifest: {completion_path}")
        if int(completion.get("episode_index", -1)) != episode_index:
            raise ValueError(f"episode index disagrees with manifest: {completion_path}")
        files = completion.get("files")
        video_name = files.get("video") if isinstance(files, dict) else None
        if not isinstance(video_name, str) or Path(video_name).name != video_name:
            raise ValueError(
                f"completion record has no local rollout video: {completion_path}"
            )
        video_path = run_dir / video_name
        if not video_path.is_file():
            raise FileNotFoundError(f"rollout video is missing: {video_path}")
        trials.append(
            TrialSource(
                source=source,
                run_dir=run_dir,
                run_path=run_path,
                run_sha256=file_sha256(run_path),
                task_id=task_id,
                episode_index=episode_index,
                completion_path=completion_path,
                video_path=video_path,
                completion=completion,
                completion_sha256=file_sha256(completion_path),
                video_sha256=file_sha256(video_path),
            )
        )
    return manifest, trials


def frame_sha256(frame: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(size) for size in frame.shape)).encode("ascii"))
    digest.update(str(frame.dtype).encode("ascii"))
    digest.update(frame.tobytes(order="C"))
    return digest.hexdigest()


def decode_selected_frames(
    video_path: Path,
    selected_steps: set[int],
    expected_steps: int,
    *,
    cv2: Any,
    np: Any,
) -> tuple[dict[int, Any], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open rollout video: {video_path}")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    frames = {}
    decoded_frames = 0
    try:
        while True:
            readable, bgr = capture.read()
            if not readable:
                break
            if decoded_frames in selected_steps:
                frames[decoded_frames] = np.ascontiguousarray(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )
            decoded_frames += 1
    finally:
        capture.release()
    if decoded_frames != expected_steps:
        raise ValueError(
            f"video has {decoded_frames} decoded frames but rollout records "
            f"{expected_steps} policy steps: {video_path}"
        )
    missing = sorted(selected_steps - set(frames))
    if missing:
        raise ValueError(f"video is missing selected policy steps {missing}: {video_path}")
    return frames, {
        "declared_frame_count": declared_frames,
        "decoded_frame_count": decoded_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "codec_fourcc": "".join(
            chr((fourcc >> (8 * byte)) & 0xFF) for byte in range(4)
        ).rstrip("\0"),
    }


def prepare_output(
    output_dir: Path, run_record: dict[str, Any], resume: bool
) -> dict[str, Any]:
    path = output_dir / "run.json"
    if path.is_file():
        existing = load_json(path)
        if not resume:
            raise FileExistsError(f"Qwen output already exists; pass --resume: {path}")
        if existing.get("configuration_sha256") != run_record["configuration_sha256"]:
            raise ValueError("Qwen resume configuration does not match the existing run")
        if existing.get("repository_state") != run_record["repository_state"]:
            raise ValueError("Qwen experiment code changed since the existing run began")
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Qwen output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, run_record)
    return run_record


def trial_checkpoint(
    path: Path,
    source: TrialSource,
    configuration_sha256: str,
    expected_queries: tuple[int, ...],
    resume: bool,
) -> dict[str, Any]:
    if path.is_file():
        if not resume:
            raise FileExistsError(f"Qwen trial already has output: {path}")
        value = load_json(path)
        expected = {
            "configuration_sha256": configuration_sha256,
            "run_sha256": source.run_sha256,
            "completion_sha256": source.completion_sha256,
            "video_sha256": source.video_sha256,
            "expected_query_steps": list(expected_queries),
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError(f"Qwen trial checkpoint does not match its inputs: {path}")
        timeline = value.get("timeline")
        if not isinstance(timeline, list):
            raise ValueError(f"Qwen trial checkpoint has no timeline: {path}")
        completed_steps = [int(item["policy_step"]) for item in timeline]
        if completed_steps != list(expected_queries[: len(completed_steps)]):
            raise ValueError(f"Qwen trial checkpoint is not a query prefix: {path}")
        return value
    return {
        "schema_version": 1,
        "status": "running",
        "configuration_sha256": configuration_sha256,
        "source": source.source,
        "task_id": source.task_id,
        "episode_index": source.episode_index,
        "condition": source.completion.get("condition"),
        "task_description": source.completion.get("task_description"),
        "success": bool(source.completion["success"]),
        "fault": source.completion.get("fault"),
        "policy_steps": int(source.completion["policy_steps"]),
        "run_path": str(source.run_path),
        "run_sha256": source.run_sha256,
        "completion_path": str(source.completion_path),
        "completion_sha256": source.completion_sha256,
        "video_path": str(source.video_path),
        "video_sha256": source.video_sha256,
        "expected_query_steps": list(expected_queries),
        "timeline": [],
    }


def _resolve_path(value: str, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()
