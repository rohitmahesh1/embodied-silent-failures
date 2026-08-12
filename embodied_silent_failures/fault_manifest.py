import json
from dataclasses import dataclass
from pathlib import Path

from embodied_silent_failures.faults import FaultSpec
from embodied_silent_failures.plan import Trial


@dataclass(frozen=True)
class FaultManifest:
    selection_basis: str
    specs: dict[Trial, FaultSpec]


def load_fault_manifest(path: Path) -> FaultManifest:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if value.get("schema_version") != 1:
        raise ValueError("fault manifest must use schema version 1")
    selection_basis = value.get("selection_basis")
    if not isinstance(selection_basis, str) or not selection_basis.strip():
        raise ValueError("fault manifest must name its selection basis")
    entries = value.get("trials")
    if not isinstance(entries, list) or not entries:
        raise ValueError("fault manifest must contain at least one trial")

    specs: dict[Trial, FaultSpec] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("fault"), dict):
            raise ValueError("each fault-manifest trial must contain a fault")
        trial = Trial(int(entry["task_id"]), int(entry["episode_index"]))
        if trial in specs:
            raise ValueError(f"duplicate fault-manifest trial: {trial}")
        specs[trial] = FaultSpec.from_dict(entry["fault"])
    return FaultManifest(selection_basis=selection_basis, specs=specs)
