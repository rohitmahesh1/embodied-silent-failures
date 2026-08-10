import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class Trial:
    task_id: int
    episode_index: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def seed_for_trial(base_seed: int, trial: Trial) -> int:
    if base_seed < 0:
        raise ValueError("base seed must be non-negative")

    identity = f"{base_seed}:{trial.task_id}:{trial.episode_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def parse_task_ids(value: str) -> list[int]:
    """Parse a comma-separated list containing integers and inclusive ranges."""
    if not value.strip():
        raise ValueError("task IDs cannot be empty")

    task_ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"invalid task ID list: {value!r}")

        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise ValueError(f"invalid task range: {part!r}")
            start, end = (int(piece) for piece in pieces)
            if end < start:
                raise ValueError(f"task range ends before it starts: {part!r}")
            task_ids.extend(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"invalid task ID: {part!r}")
            task_ids.append(int(part))

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must not contain duplicates")

    return task_ids


def build_trial_plan(
    task_ids: list[int],
    episode_start: int,
    episode_stop: int,
    episode_stride: int,
) -> list[Trial]:
    if not task_ids:
        raise ValueError("at least one task ID is required")
    if any(task_id < 0 for task_id in task_ids):
        raise ValueError("task IDs must be non-negative")
    if episode_start < 0:
        raise ValueError("episode start must be non-negative")
    if episode_stop <= episode_start:
        raise ValueError("episode stop must be greater than episode start")
    if episode_stride <= 0:
        raise ValueError("episode stride must be positive")

    return [
        Trial(task_id=task_id, episode_index=episode_index)
        for task_id in task_ids
        for episode_index in range(episode_start, episode_stop, episode_stride)
    ]
