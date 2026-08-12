import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_silent_failures.plan import Trial


@dataclass(frozen=True)
class StaleImageSpec:
    policy_step: int
    image_lag: int
    source_policy_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "stale_image",
            "policy_step": self.policy_step,
            "image_lag": self.image_lag,
            "source_policy_step": self.source_policy_step,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StaleImageSpec":
        policy_step = value.get("policy_step")
        image_lag = value.get("image_lag")
        source_policy_step = value.get("source_policy_step")
        if type(policy_step) is not int or policy_step < 0:
            raise ValueError("stale-image policy_step must be a non-negative integer")
        if type(image_lag) is not int or image_lag <= 0:
            raise ValueError("stale-image image_lag must be a positive integer")
        if source_policy_step is None:
            source_policy_step = policy_step - image_lag
        if type(source_policy_step) is not int or source_policy_step < 0:
            raise ValueError(
                "stale-image source_policy_step must be a non-negative integer"
            )
        if source_policy_step != policy_step - image_lag:
            raise ValueError(
                "stale-image source_policy_step must equal policy_step - image_lag"
            )
        return cls(
            policy_step=policy_step,
            image_lag=image_lag,
            source_policy_step=source_policy_step,
        )


@dataclass(frozen=True)
class StaleImageManifest:
    selection_basis: str
    specs: dict[Trial, StaleImageSpec]


def _trial(entry: dict[str, Any], index: int, context: str) -> Trial:
    task_id = entry.get("task_id")
    episode_index = entry.get("episode_index")
    if type(task_id) is not int or task_id < 0:
        raise ValueError(f"{context} entry {index} has an invalid task_id")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError(f"{context} entry {index} has an invalid episode_index")
    return Trial(task_id=task_id, episode_index=episode_index)


def _selection_basis(value: dict[str, Any]) -> str:
    selection_basis = value.get("selection_basis")
    if not isinstance(selection_basis, str) or not selection_basis.strip():
        raise ValueError("stale-image manifest must name its selection basis")
    return selection_basis


def _load_explicit_manifest(value: dict[str, Any]) -> StaleImageManifest:
    entries = value.get("trials")
    if not isinstance(entries, list) or not entries:
        raise ValueError("stale-image manifest must contain at least one trial")

    specs: dict[Trial, StaleImageSpec] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("stale_image"), dict):
            raise ValueError("each stale-image trial must contain a stale_image object")
        trial = _trial(entry, index, "stale-image manifest")
        if trial in specs:
            raise ValueError(f"duplicate stale-image trial: {trial}")
        specs[trial] = StaleImageSpec.from_dict(entry["stale_image"])
    return StaleImageManifest(selection_basis=_selection_basis(value), specs=specs)


def _load_probe_selection(value: dict[str, Any]) -> StaleImageManifest:
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("stale-image probe must contain at least one record")

    specs: dict[Trial, StaleImageSpec] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"stale-image probe record {index} is not an object")
        if record.get("clean_reproduces_trace") is not True:
            continue
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(
                f"stale-image probe record {index} has no candidate list"
            )
        eligible: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"stale-image probe record {index} has a non-object candidate"
                )
            change = candidate.get("action_change")
            if not isinstance(change, dict):
                raise ValueError(
                    f"stale-image probe record {index} has a candidate with no action_change"
                )
            if change.get("gripper_changed") is True:
                eligible.append(candidate)
        if not eligible:
            continue

        trial = _trial(record, index, "stale-image probe")
        if trial in specs:
            raise ValueError(f"duplicate stale-image probe record: {trial}")
        chosen = min(eligible, key=lambda candidate: candidate["image_lag"])
        specs[trial] = StaleImageSpec.from_dict(
            {
                "policy_step": record.get("policy_step"),
                "image_lag": chosen.get("image_lag"),
                "source_policy_step": chosen.get("source_policy_step"),
            }
        )

    if not specs:
        raise ValueError(
            "stale-image probe does not contain any clean-reproducing gripper-changing trials"
        )
    return StaleImageManifest(
        selection_basis=(
            "smallest_gripper_changing_lag_from_clean_reproducing_probe_records"
        ),
        specs=specs,
    )


def load_stale_image_manifest(path: Path) -> StaleImageManifest:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if value.get("schema_version") != 1:
        raise ValueError("stale-image manifest must use schema version 1")
    if "records" in value:
        return _load_probe_selection(value)
    return _load_explicit_manifest(value)
