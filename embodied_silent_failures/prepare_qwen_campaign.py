import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import file_sha256


TASK_IDS = tuple(range(10))
NATIVE_PER_OUTCOME = 2
TEMPORAL_MINIMUM_STEPS = 401


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze video-backed trials for the initial Qwen campaign."
    )
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--stale-dir", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sampling-seed", type=int, default=20260813)
    return parser.parse_args()


def _records(directory: Path, condition: str) -> dict[tuple[int, int], dict[str, Any]]:
    if not (directory / "run.json").is_file():
        raise FileNotFoundError(f"run record is missing: {directory / 'run.json'}")
    records = {}
    for path in sorted(directory.glob("*.complete.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete" or value.get("condition") != condition:
            raise ValueError(f"unexpected completion record: {path}")
        files = value.get("files")
        video_name = files.get("video") if isinstance(files, dict) else None
        if not isinstance(video_name, str) or not (directory / video_name).is_file():
            raise FileNotFoundError(f"video-backed completion is required: {path}")
        key = (int(value["task_id"]), int(value["episode_index"]))
        if key in records:
            raise ValueError(f"duplicate completion record for {key}: {directory}")
        records[key] = value
    return records


def _rank(
    seed: int, label: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def key(record: dict[str, Any]) -> str:
        identity = f"{seed}:{label}:{record['task_id']}:{record['episode_index']}"
        return hashlib.sha256(identity.encode("ascii")).hexdigest()

    return sorted(records, key=key)


def select_native_trials(
    clean: dict[tuple[int, int], dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    if len(clean) != 450:
        raise ValueError(f"expected 450 main clean records, found {len(clean)}")
    selected = []
    for task_id in TASK_IDS:
        for success in (False, True):
            stratum = [
                value
                for value in clean.values()
                if int(value["task_id"]) == task_id
                and bool(value["success"]) is success
            ]
            long = [
                value
                for value in stratum
                if int(value["policy_steps"]) >= TEMPORAL_MINIMUM_STEPS
            ]
            if not long:
                raise ValueError(
                    f"no temporal-extension candidate for task {task_id}, success={success}"
                )

            # prepare_graph_campaign.py::_clean_stages at 505b350 first reserves
            # this hash-ranked long rollout, then ranks the remaining stratum for
            # its census waves. Repeating that declared rule makes these the first
            # two video-backed census waves rather than a new outcome search.
            temporal = _rank(seed, f"temporal:{task_id}:{success}", long)[0]
            remaining = [value for value in stratum if value is not temporal]
            selected.extend(
                _rank(seed, f"census:{task_id}:{success}", remaining)[
                    :NATIVE_PER_OUTCOME
                ]
            )
    if len(selected) != 40:
        raise ValueError(f"expected 40 native trials, selected {len(selected)}")
    return selected


def select_causal_pairs(
    stale: dict[tuple[int, int], dict[str, Any]],
    control: dict[tuple[int, int], dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = []
    for key in sorted(stale.keys() & control.keys()):
        left, right = stale[key], control[key]
        for field in ("trial_seed", "initial_state_sha256"):
            if left.get(field) != right.get(field):
                raise ValueError(f"matched pair {key} disagrees on {field}")
        if left.get("success") is False and right.get("success") is True:
            if left.get("fault", {}).get("policy_step") != right.get("fault", {}).get(
                "policy_step"
            ):
                raise ValueError(f"matched pair {key} disagrees on intervention step")
            pairs.append((left, right))
    if len(pairs) != 26:
        raise ValueError(f"expected 26 stale-only failure pairs, found {len(pairs)}")
    return pairs


def _entry(source: str, run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "run_dir": str(run_dir.resolve()),
        "task_id": int(record["task_id"]),
        "episode_index": int(record["episode_index"]),
    }


def _write_manifest(
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


def main() -> None:
    args = _parse_arguments()
    if args.sampling_seed < 0:
        raise ValueError("sampling seed must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"campaign directory is not empty: {args.output_dir}")
    manifest_dir = args.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    clean = _records(args.clean_dir, "clean")
    stale = _records(args.stale_dir, "stale_image")
    control = _records(args.control_dir, "current_image_control")
    native = select_native_trials(clean, args.sampling_seed)
    pairs = select_causal_pairs(stale, control)

    native_entries = [_entry("native", args.clean_dir, record) for record in native]
    canary_records = [
        next(record for record in native if bool(record["success"]) is False),
        next(record for record in native if bool(record["success"]) is True),
    ]
    causal_entries = []
    for left, right in pairs:
        causal_entries.append(_entry("stale", args.stale_dir, left))
        causal_entries.append(_entry("control", args.control_dir, right))

    paths = {
        "canary": manifest_dir / "canary.json",
        "native": manifest_dir / "native.json",
        "causal": manifest_dir / "causal.json",
    }
    _write_manifest(
        paths["canary"],
        "protocol:qwen-initial-campaign-v1:first-hash-ranked-native-failure-and-success",
        [_entry("native", args.clean_dir, record) for record in canary_records],
    )
    _write_manifest(
        paths["native"],
        "protocol:qwen-initial-campaign-v1:first-two-prior-census-waves-per-task-outcome",
        native_entries,
    )
    _write_manifest(
        paths["causal"],
        "protocol:qwen-initial-campaign-v1:all-unbiased-lag1-stale-only-failures-and-controls",
        causal_entries,
    )
    campaign = {
        "schema_version": 1,
        "preparation_code_sha256": file_sha256(Path(__file__)),
        "sampling_seed": args.sampling_seed,
        "protocol": {
            "history_frames": 8,
            "history_stride": 5,
            "query_stride": 5,
            "canary_query_stride": 25,
            "max_new_tokens": 64,
            "torch_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "generation": "greedy",
        },
        "design": {
            "canary_trials": len(canary_records),
            "native_trials": len(native_entries),
            "native_trials_per_task_outcome": NATIVE_PER_OUTCOME,
            "causal_pairs": len(pairs),
            "causal_trials": len(causal_entries),
        },
        "sources": {
            "clean_dir": str(args.clean_dir.resolve()),
            "stale_dir": str(args.stale_dir.resolve()),
            "control_dir": str(args.control_dir.resolve()),
            "clean_run_sha256": file_sha256(args.clean_dir / "run.json"),
            "stale_run_sha256": file_sha256(args.stale_dir / "run.json"),
            "control_run_sha256": file_sha256(args.control_dir / "run.json"),
        },
        "manifests": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    write_json_atomic(args.output_dir / "campaign.json", campaign)
    print(json.dumps(campaign["design"], sort_keys=True))


if __name__ == "__main__":
    main()
