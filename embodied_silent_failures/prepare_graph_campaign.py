import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic


TASK_IDS = tuple(range(10))
CENSUS_PER_OUTCOME = 5
TEMPORAL_MINIMUM_STEPS = 401
CANARY_TRIALS = ((3, 7), (3, 8))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze manifests for the OpenVLA evidence-graph campaign."
    )
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--stale-dir", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--stale-source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sampling-seed", type=int, default=20260813)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(directory: Path, condition: str) -> dict[tuple[int, int], dict[str, Any]]:
    records = {}
    for path in sorted(directory.glob("*.complete.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("condition") != condition:
            raise ValueError(f"unexpected completion record: {path}")
        key = (int(value["task_id"]), int(value["episode_index"]))
        if key in records:
            raise ValueError(f"duplicate completion record for {key}")
        records[key] = {**value, "_path": str(path.resolve()), "_sha256": _sha256(path)}
    return records


def _rank(seed: int, label: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> str:
        identity = f"{seed}:{label}:{record['task_id']}:{record['episode_index']}"
        return hashlib.sha256(identity.encode("ascii")).hexdigest()

    return sorted(records, key=key)


def _trial_record(record: dict[str, Any], stratum: str) -> dict[str, Any]:
    return {
        "task_id": int(record["task_id"]),
        "episode_index": int(record["episode_index"]),
        "sampling_stratum": stratum,
        "source_success": bool(record["success"]),
        "source_policy_steps": int(record["policy_steps"]),
        "source_completion": record["_path"],
        "source_completion_sha256": record["_sha256"],
    }


def _write_trial_manifest(
    path: Path, selection_basis: str, trials: list[dict[str, Any]]
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "selection_basis": selection_basis,
            "trials": trials,
        },
    )


def _clean_stages(
    clean: dict[tuple[int, int], dict[str, Any]], output_dir: Path, seed: int
) -> list[dict[str, Any]]:
    if len(clean) != 450:
        raise ValueError(f"expected 450 clean baseline records, found {len(clean)}")

    temporal = []
    census_by_wave: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for task_id in TASK_IDS:
        for success in (False, True):
            stratum = [
                value
                for value in clean.values()
                if value["task_id"] == task_id and bool(value["success"]) is success
            ]
            if len(stratum) < CENSUS_PER_OUTCOME + 1:
                raise ValueError(f"too few clean records for task {task_id}, success={success}")
            long = [
                value for value in stratum if int(value["policy_steps"]) >= TEMPORAL_MINIMUM_STEPS
            ]
            if not long:
                raise ValueError(f"no step-400 record for task {task_id}, success={success}")
            chosen_temporal = _rank(seed, f"temporal:{task_id}:{success}", long)[0]
            temporal.append(_trial_record(chosen_temporal, f"task={task_id},success={success}"))

            remaining = [value for value in stratum if value is not chosen_temporal]
            chosen_census = _rank(seed, f"census:{task_id}:{success}", remaining)[
                :CENSUS_PER_OUTCOME
            ]
            for wave, record in enumerate(chosen_census):
                census_by_wave[wave].append(
                    _trial_record(record, f"task={task_id},success={success}")
                )

    stages = []
    for wave, trials in enumerate(census_by_wave):
        name = f"census-wave-{wave}"
        manifest = output_dir / "manifests" / f"{name}.json"
        _write_trial_manifest(
            manifest,
            "fixed-seed hash-ranked sample with one success and one failure per task",
            trials,
        )
        stages.append(
            {
                "name": name,
                "kind": "clean",
                "manifest": str(manifest.resolve()),
                "trace_steps": [100],
                "expected_trials": 20,
                "allow_exclusions": False,
            }
        )

    for wave, task_ids in enumerate((range(0, 5), range(5, 10))):
        name = f"temporal-wave-{wave}"
        trials = [item for item in temporal if int(item["task_id"]) in task_ids]
        manifest = output_dir / "manifests" / f"{name}.json"
        _write_trial_manifest(
            manifest,
            "fixed-seed hash-ranked length-eligible sample with one success and one failure per task",
            trials,
        )
        stages.append(
            {
                "name": name,
                "kind": "clean",
                "manifest": str(manifest.resolve()),
                "trace_steps": [0, 100, 200, 300, 400],
                "expected_trials": 10,
                "allow_exclusions": False,
            }
        )
    return stages


def _stale_specs(path: Path) -> dict[tuple[int, int], dict[str, int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = {}
    for item in value.get("trials", []):
        key = (int(item["task_id"]), int(item["episode_index"]))
        specs[key] = dict(item["stale_image"])
    return specs


def _paired_stages(
    stale_dir: Path,
    control_dir: Path,
    source_manifest: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stale = _records(stale_dir, "stale_image")
    control = _records(control_dir, "current_image_control")
    specs = _stale_specs(source_manifest)
    pairs = []
    for key in sorted(stale.keys() & control.keys()):
        left, right = stale[key], control[key]
        for field in ("trial_seed", "initial_state_sha256"):
            if left.get(field) != right.get(field):
                raise ValueError(f"matched pair {key} disagrees on {field}")
        if left.get("success") is False and right.get("success") is True:
            spec = specs[key]
            if int(left["fault"]["policy_step"]) != int(spec["policy_step"]):
                raise ValueError(f"stale manifest disagrees with result for {key}")
            pairs.append((key, spec, left, right))
    if len(pairs) != 26:
        raise ValueError(f"expected 26 stale-only failures, found {len(pairs)}")

    stages = []
    for task_id in sorted({key[0] for key, *_ in pairs}):
        task_pairs = [item for item in pairs if item[0][0] == task_id]
        manifest = output_dir / "manifests" / f"intervention-task-{task_id}.json"
        trials = []
        for (pair_task, episode_index), spec, left, right in task_pairs:
            trials.append(
                {
                    "task_id": pair_task,
                    "episode_index": episode_index,
                    "stale_image": spec,
                    "source_stale_completion": left["_path"],
                    "source_stale_completion_sha256": left["_sha256"],
                    "source_control_completion": right["_path"],
                    "source_control_completion_sha256": right["_sha256"],
                }
            )
        write_json_atomic(
            manifest,
            {
                "schema_version": 1,
                "selection_basis": (
                    "all stale-only failures from the prior unbiased lag-1 campaign; "
                    "outcome-selected for mechanism analysis"
                ),
                "trials": trials,
            },
        )
        for mode in ("stale", "current_control"):
            stages.append(
                {
                    "name": f"intervention-task-{task_id}-{mode.replace('_', '-')}",
                    "kind": "stale_image",
                    "image_input_mode": mode,
                    "manifest": str(manifest.resolve()),
                    "trace_steps": [],
                    "expected_trials": len(trials),
                    "allow_exclusions": True,
                }
            )

    return stages, {
        "pair_count": len(pairs),
        "task_ids": sorted({key[0] for key, *_ in pairs}),
    }


def _canary(source_manifest: Path, stale_dir: Path, output_dir: Path) -> dict[str, Any]:
    specs = _stale_specs(source_manifest)
    first, second = CANARY_TRIALS
    if not (stale_dir / f"task{first[0]}--ep{first[1]}.excluded.json").is_file():
        raise ValueError("the first canary trial is not a known replay exclusion")
    if not (stale_dir / f"task{second[0]}--ep{second[1]}.complete.json").is_file():
        raise ValueError("the second canary trial is not a known valid replay")
    manifest = output_dir / "manifests" / "canary.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": 1,
            "selection_basis": "known replay exclusion followed by a known valid replay",
            "trials": [
                {"task_id": task, "episode_index": episode, "stale_image": specs[(task, episode)]}
                for task, episode in CANARY_TRIALS
            ],
        },
    )
    return {
        "name": "canary",
        "kind": "stale_image",
        "image_input_mode": "stale",
        "manifest": str(manifest.resolve()),
        "trace_steps": [],
        "expected_trials": 2,
        "allow_exclusions": True,
        "expected_complete": [[second[0], second[1]]],
        "expected_excluded": [[first[0], first[1]]],
    }


def main() -> None:
    args = _parse_arguments()
    if args.sampling_seed < 0:
        raise ValueError("sampling seed must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"campaign output is not empty: {args.output_dir}")
    (args.output_dir / "manifests").mkdir(parents=True, exist_ok=True)

    clean = _records(args.clean_dir, "clean")
    stages = _clean_stages(clean, args.output_dir, args.sampling_seed)
    paired, paired_summary = _paired_stages(
        args.stale_dir,
        args.control_dir,
        args.stale_source_manifest,
        args.output_dir,
    )
    stages.extend(paired)
    campaign = {
        "schema_version": 1,
        "preparation_code_sha256": _sha256(Path(__file__)),
        "sampling_seed": args.sampling_seed,
        "design": {
            "census": "five successes and five failures per task, traced at step 100",
            "temporal": "one length-eligible success and failure per task, traced at five fixed steps",
            "intervention": paired_summary,
        },
        "sources": {
            "clean_dir": str(args.clean_dir.resolve()),
            "stale_dir": str(args.stale_dir.resolve()),
            "control_dir": str(args.control_dir.resolve()),
            "stale_source_manifest": str(args.stale_source_manifest.resolve()),
            "stale_source_manifest_sha256": _sha256(args.stale_source_manifest),
        },
        "canary": _canary(args.stale_source_manifest, args.stale_dir, args.output_dir),
        "stages": stages,
    }
    write_json_atomic(args.output_dir / "campaign.json", campaign)
    print(json.dumps({"stages": len(stages), "design": campaign["design"]}, indent=2))


if __name__ == "__main__":
    main()
