from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired stale-frame freshness-gate rollouts."
    )
    parser.add_argument("--stale-dir", required=True, type=Path)
    parser.add_argument("--current-control-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _trial(result: dict[str, Any]) -> tuple[int, int]:
    return int(result["task_id"]), int(result["episode_index"])


def _load_results(
    directory: Path, condition: str
) -> dict[tuple[int, int], dict[str, Any]]:
    results: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(directory.glob("*.complete.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("condition") != condition:
            raise ValueError(f"unexpected completion marker in {path}")
        trial = _trial(value)
        if trial in results:
            raise ValueError(f"duplicate completion marker for trial {trial}")
        results[trial] = {**value, "_directory": directory}
    if not results:
        raise ValueError(f"no {condition} completion markers in {directory}")
    return results


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected CSV boolean, got {value!r}")


def _rows(result: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(result["_directory"]) / result["files"]["csv"]
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _exact_binomial_two_sided(left: int, right: int) -> float | None:
    discordant = left + right
    if discordant == 0:
        return None
    smaller = min(left, right)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def analyze(stale_dir: Path, current_dir: Path) -> dict[str, Any]:
    stale = _load_results(stale_dir, "stale_image")
    current = _load_results(current_dir, "current_image_control")
    paired = sorted(set(stale) & set(current))
    if not paired:
        raise ValueError("stale and current-control directories have no paired trials")

    both_success = stale_only_failure = current_only_failure = both_failure = 0
    stale_intervention_alarms = {
        "source_metadata": 0,
        "relabelled_metadata": 0,
        "exact_duplicate": 0,
        "selected_gate": 0,
        "response_applied": 0,
    }
    clean_steps = clean_duplicate_alarms = clean_selected_alarms = 0
    clean_episodes_with_duplicate = clean_episodes_with_selected = 0

    for trial in paired:
        stale_result = stale[trial]
        current_result = current[trial]
        stale_success = bool(stale_result["success"])
        current_success = bool(current_result["success"])
        if stale_success and current_success:
            both_success += 1
        elif not stale_success and current_success:
            stale_only_failure += 1
        elif stale_success and not current_success:
            current_only_failure += 1
        else:
            both_failure += 1

        intervention_step = int(stale_result["fault"]["policy_step"])
        stale_rows = _rows(stale_result)
        intervention = next(
            row
            for row in stale_rows
            if int(row["action/timestep"]) == intervention_step
        )
        for name, column in (
            ("source_metadata", "freshness/source_metadata_alarm"),
            ("relabelled_metadata", "freshness/relabelled_metadata_alarm"),
            ("exact_duplicate", "freshness/exact_duplicate_alarm"),
            ("selected_gate", "freshness/selected_gate_alarm"),
            ("response_applied", "freshness/response_applied"),
        ):
            stale_intervention_alarms[name] += int(_bool(intervention[column]))

        current_rows = _rows(current_result)
        duplicate_in_episode = False
        selected_in_episode = False
        for row in current_rows:
            duplicate = _bool(row["freshness/exact_duplicate_alarm"])
            selected = _bool(row["freshness/selected_gate_alarm"])
            clean_steps += 1
            clean_duplicate_alarms += int(duplicate)
            clean_selected_alarms += int(selected)
            duplicate_in_episode = duplicate_in_episode or duplicate
            selected_in_episode = selected_in_episode or selected
        clean_episodes_with_duplicate += int(duplicate_in_episode)
        clean_episodes_with_selected += int(selected_in_episode)

    return {
        "schema_version": 1,
        "paired_trials": len(paired),
        "unpaired_stale_trials": len(set(stale) - set(current)),
        "unpaired_current_control_trials": len(set(current) - set(stale)),
        "paired_task_outcomes": {
            "both_success": both_success,
            "stale_only_failure": stale_only_failure,
            "current_control_only_failure": current_only_failure,
            "both_failure": both_failure,
            "exact_mcnemar_two_sided_p": _exact_binomial_two_sided(
                stale_only_failure, current_only_failure
            ),
        },
        "stale_intervention_detection": {
            **stale_intervention_alarms,
            "trials": len(paired),
        },
        "current_control_false_alarms": {
            "policy_steps": clean_steps,
            "exact_duplicate_alarms": clean_duplicate_alarms,
            "exact_duplicate_rate": clean_duplicate_alarms / clean_steps,
            "exact_duplicate_rate_wilson_95": _wilson(
                clean_duplicate_alarms, clean_steps
            ),
            "episodes": len(paired),
            "episodes_with_exact_duplicate_alarm": clean_episodes_with_duplicate,
            "episode_exact_duplicate_rate": clean_episodes_with_duplicate / len(paired),
            "selected_gate_alarms": clean_selected_alarms,
            "episodes_with_selected_gate_alarm": clean_episodes_with_selected,
        },
        "interpretation_boundary": {
            "source_metadata": (
                "Simulator source-step proxy; valid only when frame metadata remains "
                "bound to the replayed pixels."
            ),
            "relabelled_metadata": (
                "Old pixels assigned current metadata by a downstream consumer; "
                "timestamp and frame-ID freshness checks cannot detect this condition."
            ),
            "response": (
                "The hold response is enabled only at the manifest-declared matched "
                "intervention step; other alarms are shadow observations."
            ),
        },
    }


def main() -> None:
    args = _parse_arguments()
    result = analyze(args.stale_dir, args.current_control_dir)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
