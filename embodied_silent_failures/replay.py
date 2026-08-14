import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_silent_failures.plan import Trial


ACTION_COLUMNS = (
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
)


@dataclass(frozen=True)
class CleanTrace:
    result: dict[str, Any]
    rows: list[dict[str, str]]
    hidden_states: Any
    observations: dict[str, Any]
    source_dir: Path


def paired_clean_results(
    directories: list[Path], plan: list[Trial]
) -> tuple[list[Trial], dict[Trial, dict[str, Any]]]:
    indexed: dict[Trial, dict[str, Any]] = {}
    requested = set(plan)
    for directory in directories:
        for path in sorted(directory.glob("*.complete.json")):
            with path.open("r", encoding="utf-8") as file:
                result = json.load(file)
            if result.get("status") != "complete" or result.get("condition") != "clean":
                raise ValueError(
                    f"paired reference is not a completed clean rollout: {path}"
                )
            trial = Trial(
                task_id=int(result["task_id"]),
                episode_index=int(result["episode_index"]),
            )
            if trial not in requested:
                continue
            if trial in indexed:
                fields = (
                    "initial_state_sha256",
                    "trial_seed",
                    "success",
                    "policy_steps",
                )
                if any(indexed[trial].get(key) != result.get(key) for key in fields):
                    raise ValueError(f"paired clean results conflict for {trial}")
                continue
            indexed[trial] = {**result, "_source_dir": str(directory.resolve())}

    missing = [trial for trial in plan if trial not in indexed]
    if missing:
        preview = ", ".join(
            f"{trial.task_id}/{trial.episode_index}" for trial in missing[:5]
        )
        raise FileNotFoundError(f"missing paired clean results for {preview}")

    eligible = [trial for trial in plan if indexed[trial].get("success") is True]
    if not eligible:
        raise ValueError("none of the paired clean rollouts succeeded")
    for trial in eligible:
        clean_steps = int(indexed[trial]["policy_steps"])
        if clean_steps <= 0:
            raise ValueError(f"paired clean rollout has invalid length for {trial}")
    return eligible, indexed


def load_clean_trace(result: dict[str, Any]) -> CleanTrace:
    if result.get("condition") != "clean" or result.get("success") is not True:
        raise ValueError("counterfactual replay requires a successful clean result")
    source = result.get("_source_dir")
    files = result.get("files")
    if not isinstance(source, str) or not isinstance(files, dict):
        raise ValueError("clean result does not identify its source artifacts")

    source_dir = Path(source)
    csv_path = source_dir / str(files.get("csv"))
    pickle_path = source_dir / str(files.get("pickle"))
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with pickle_path.open("rb") as file:
        payload = pickle.load(file)

    expected = int(result["policy_steps"])
    hidden_states = payload.get("hidden_states")
    observations = payload.get("observations")
    if len(rows) != expected:
        raise ValueError(f"clean CSV has {len(rows)} steps, expected {expected}")
    if hidden_states is None or len(hidden_states) != expected:
        raise ValueError("clean hidden-state history has the wrong length")
    if not isinstance(observations, dict) or not observations:
        raise ValueError("clean trace has no numeric observation history")
    if any(len(values) != expected for values in observations.values()):
        raise ValueError("clean observation history has inconsistent lengths")
    if any(column not in rows[0] for column in ACTION_COLUMNS):
        raise ValueError("clean CSV does not contain executed actions")

    return CleanTrace(
        result=result,
        rows=rows,
        hidden_states=hidden_states,
        observations=observations,
        source_dir=source_dir,
    )


def replay_action(np: Any, trace: CleanTrace, step: int) -> Any:
    return np.asarray(
        [float(trace.rows[step][column]) for column in ACTION_COLUMNS],
        dtype=np.float64,
    )


def observation_error(
    np: Any, trace: CleanTrace, observation: dict[str, Any], step: int
) -> float:
    maximum = 0.0
    for key, expected_history in trace.observations.items():
        if key not in observation:
            raise KeyError(f"replayed observation has no {key}")
        expected = np.asarray(expected_history[step])
        actual = np.asarray(observation[key])
        if actual.shape != expected.shape:
            raise ValueError(
                f"replayed observation {key} has shape {actual.shape}, "
                f"expected {expected.shape}"
            )
        if actual.size:
            error = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
            maximum = max(maximum, error)
    return maximum
