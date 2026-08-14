import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.provenance import file_sha256, load_json
from embodied_silent_failures.qwen_saturation import (
    SATURATION_BASIS,
    coverage_novelty,
    coverage_record,
    empty_coverage_union,
    select_saturation_queries,
    update_coverage_union,
    zero_discovery_upper_bound,
)
from embodied_silent_failures.trace_qwen_query import (
    load_trace_runtime,
    trace_query,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace the predeclared Qwen structural-saturation campaign."
    )
    parser.add_argument("--native-run", required=True, type=Path)
    parser.add_argument("--native-trials", required=True, type=Path)
    parser.add_argument("--causal-run", required=True, type=Path)
    parser.add_argument("--causal-trials", required=True, type=Path)
    parser.add_argument("--seed-trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--holdouts-per-stratum", type=int, default=5)
    return parser.parse_args()


def _source(
    label: str, run_path: Path, trials_dir: Path
) -> dict[str, Any]:
    run_path = run_path.resolve()
    return {
        "label": label,
        "run": load_json(run_path),
        "run_path": run_path,
        "run_sha256": file_sha256(run_path),
        "trials_dir": trials_dir.resolve(),
    }


def _event_stream(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"trace line {line_number} is not a JSON object: {path}")
            yield value


def _coverage_from_artifact(path: Path) -> dict[str, Any]:
    existing = path / "coverage.json"
    if existing.is_file():
        return load_json(existing)
    composition = load_json(path / "composition.json")
    return coverage_record(
        _event_stream(path / "raw.jsonl"),
        load_json(path / "graph.json"),
        composition["processed_inputs"],
    )


def _artifact_hashes(path: Path) -> dict[str, str]:
    names = ("audit.json", "composition.json", "graph.json", "raw.jsonl")
    return {name: file_sha256(path / name) for name in names}


def _validate_seed_trace(
    path: Path, first_selection: dict[str, Any]
) -> dict[str, Any]:
    audit = load_json(path / "audit.json")
    composition = load_json(path / "composition.json")
    if audit.get("passed") is not True:
        raise ValueError("seed Qwen trace does not have a passing audit")
    if composition.get("equivalent_to_frozen_query") is not True:
        raise ValueError("seed Qwen trace did not reproduce its frozen query")
    expected = first_selection
    actual = composition["selection"]
    if actual.get("trial") != expected["trial_path"].name:
        raise ValueError("seed Qwen trace trial differs from saturation discovery")
    if int(actual.get("policy_step", -1)) != expected["policy_step"]:
        raise ValueError("seed Qwen trace step differs from saturation discovery")
    if composition.get("trial_sha256") != expected["trial_sha256"]:
        raise ValueError("seed Qwen trace trial hash differs from saturation discovery")
    return {
        "path": str(path.resolve()),
        "hashes": _artifact_hashes(path),
        "trace_revision": composition["trace_revision"],
    }


def _query_directory(root: Path, selected: dict[str, Any]) -> Path:
    alarm = int(selected["alarm"])
    return root / "queries" / (
        f"{selected['index']:03d}--{selected['phase']}--"
        f"{selected['condition_label']}--alarm-{alarm}"
    )


def _completed_attempt(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    for attempt in sorted(path.glob("attempt-*"), reverse=True):
        audit_path = attempt / "audit.json"
        coverage_path = attempt / "coverage.json"
        if (
            audit_path.is_file()
            and coverage_path.is_file()
            and load_json(audit_path).get("passed") is True
        ):
            return attempt
    return None


def _validate_completed_attempt(path: Path, selected: dict[str, Any]) -> None:
    composition = load_json(path / "composition.json")
    selection = composition.get("selection", {})
    if selection.get("trial") != selected["trial_path"].name:
        raise ValueError(f"completed trace has the wrong trial: {path}")
    if int(selection.get("policy_step", -1)) != selected["policy_step"]:
        raise ValueError(f"completed trace has the wrong policy step: {path}")
    if composition.get("trial_sha256") != selected["trial_sha256"]:
        raise ValueError(f"completed trace has the wrong trial hash: {path}")
    if composition.get("equivalent_to_frozen_query") is not True:
        raise ValueError(f"completed trace did not reproduce its frozen query: {path}")


def _next_attempt(path: Path) -> Path:
    existing = sorted(path.glob("attempt-*")) if path.is_dir() else []
    return path / f"attempt-{len(existing) + 1:03d}"


def _checkpoint(
    output_dir: Path,
    manifest: dict[str, Any],
    outcomes: list[dict[str, Any]],
    *,
    status: str,
) -> None:
    completed = [item for item in outcomes if item["status"] == "complete"]
    failures = [item for item in outcomes if item["status"] == "failed"]
    holdouts = [item for item in completed if item["phase"] == "holdout"]
    novel_holdouts = [
        item
        for item in holdouts
        if item.get("novelty_vs_discovery")
        and item["novelty_vs_discovery"]["novel"]
    ]
    measured_holdouts = [
        item for item in holdouts if item.get("novelty_vs_discovery") is not None
    ]
    summary = {
        "schema_version": 1,
        "status": status,
        "basis": SATURATION_BASIS,
        "selection_sha256": manifest["selection_sha256"],
        "total_queries": manifest["total_queries"],
        "completed_queries": len(completed),
        "failed_attempts": len(failures),
        "completed_holdouts": len(holdouts),
        "novel_holdouts": len(novel_holdouts),
        "saturated": (
            len(holdouts) == manifest["holdout_queries"]
            and len(measured_holdouts) == manifest["holdout_queries"]
            and not novel_holdouts
            and len(completed) == manifest["total_queries"]
        ),
        "zero_discovery_95_upper_bound": (
            zero_discovery_upper_bound(len(measured_holdouts))
            if measured_holdouts and not novel_holdouts
            else None
        ),
        "outcomes": outcomes,
    }
    write_json_atomic(output_dir / "campaign.json", summary)


def _load_prior_outcomes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return load_json(path).get("outcomes", [])


def _recompute_novelty(
    outcomes: list[dict[str, Any]],
    seed_coverage: dict[str, Any],
    discovery_queries: int,
) -> None:
    completed = {
        item["index"]: item for item in outcomes if item["status"] == "complete"
    }
    coverages = {
        index: (
            seed_coverage
            if index == 0
            else load_json(Path(item["artifact"]) / "coverage.json")
        )
        for index, item in completed.items()
    }
    discovery = empty_coverage_union()
    discovery_indexes = [
        index for index, item in completed.items() if item["phase"] == "discovery"
    ]
    for index in sorted(discovery_indexes):
        update_coverage_union(discovery, coverages[index])

    prior = empty_coverage_union()
    discovery_complete = len(discovery_indexes) == discovery_queries
    for index in sorted(completed):
        item = completed[index]
        coverage = coverages[index]
        item["novelty_vs_prior"] = coverage_novelty(coverage, prior)
        item["novelty_vs_discovery"] = (
            coverage_novelty(coverage, discovery)
            if item["phase"] == "holdout" and discovery_complete
            else None
        )
        update_coverage_union(prior, coverage)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    native = _source("ordinary", args.native_run, args.native_trials)
    causal_run = load_json(args.causal_run.resolve())
    causal_sha256 = file_sha256(args.causal_run.resolve())
    sources = {
        "ordinary": native,
        "control": {
            "label": "control",
            "run": causal_run,
            "run_path": args.causal_run.resolve(),
            "run_sha256": causal_sha256,
            "trials_dir": args.causal_trials.resolve(),
        },
        "stale": {
            "label": "stale",
            "run": causal_run,
            "run_path": args.causal_run.resolve(),
            "run_sha256": causal_sha256,
            "trials_dir": args.causal_trials.resolve(),
        },
    }
    selected = select_saturation_queries(
        sources,
        seed=args.seed,
        holdouts_per_stratum=args.holdouts_per_stratum,
    )
    for item in selected["selections"]:
        item["selection_basis"] = SATURATION_BASIS
        item["alarm_used_for_selection"] = True
    seed_trace = _validate_seed_trace(args.seed_trace.resolve(), selected["selections"][0])
    manifest = {
        key: value
        for key, value in selected.items()
        if key not in {"selections"}
    }
    manifest["source_runs"] = {
        label: source["run_sha256"] for label, source in sources.items()
    }
    manifest["seed_trace"] = seed_trace

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        if load_json(manifest_path) != manifest:
            raise ValueError("Qwen saturation manifest changed while resuming")
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError("Qwen saturation output is nonempty without a manifest")
        write_json_atomic(manifest_path, manifest)

    seed_coverage_path = output_dir / "seed-coverage.json"
    seed_coverage = (
        load_json(seed_coverage_path)
        if seed_coverage_path.is_file()
        else _coverage_from_artifact(args.seed_trace.resolve())
    )
    write_json_atomic(seed_coverage_path, seed_coverage)
    outcomes = _load_prior_outcomes(output_dir / "campaign.json")
    outcome_by_index = {
        item["index"]: item for item in outcomes if item["status"] == "complete"
    }
    seed_outcome = {
        "index": 0,
        "phase": "discovery",
        "stratum": selected["selections"][0]["stratum"],
        "status": "complete",
        "artifact": seed_trace["path"],
        "seed_artifact": True,
        "novelty_vs_prior": coverage_novelty(
            seed_coverage, empty_coverage_union()
        ),
        "novelty_vs_discovery": None,
    }
    if 0 not in outcome_by_index:
        outcomes.append(seed_outcome)
        outcome_by_index[0] = seed_outcome

    expected_indexes = {item["index"] for item in selected["selections"]}
    if set(outcome_by_index) == expected_indexes:
        _recompute_novelty(outcomes, seed_coverage, manifest["discovery_queries"])
        _checkpoint(output_dir, manifest, outcomes, status="complete")
        return load_json(output_dir / "campaign.json")

    runtime = load_trace_runtime(args.native_run.resolve(), args.cache_dir)
    for item in selected["selections"][1:]:
        if item["index"] in outcome_by_index:
            continue
        directory = _query_directory(output_dir, item)
        completed_attempt = _completed_attempt(directory)
        if completed_attempt is None:
            attempt = _next_attempt(directory)
            attempt.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = trace_query(runtime, item, attempt)
            except Exception as error:
                failure = {
                    "index": item["index"],
                    "phase": item["phase"],
                    "stratum": item["stratum"],
                    "status": "failed",
                    "attempt": str(attempt),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                }
                outcomes.append(failure)
                _recompute_novelty(
                    outcomes, seed_coverage, manifest["discovery_queries"]
                )
                _checkpoint(output_dir, manifest, outcomes, status="running")
                print(json.dumps(failure, sort_keys=True), flush=True)
                gc.collect()
                runtime.torch.cuda.empty_cache()
                continue
            completed_attempt = attempt
        else:
            result = None

        _validate_completed_attempt(completed_attempt, item)
        outcome = {
            "index": item["index"],
            "phase": item["phase"],
            "stratum": item["stratum"],
            "status": "complete",
            "artifact": str(completed_attempt.resolve()),
            "seed_artifact": False,
            "result": result,
            "novelty_vs_prior": None,
            "novelty_vs_discovery": None,
        }
        outcome_by_index[item["index"]] = outcome
        outcomes.append(outcome)
        _recompute_novelty(outcomes, seed_coverage, manifest["discovery_queries"])
        _checkpoint(output_dir, manifest, outcomes, status="running")
        print(
            json.dumps(
                {
                    "index": item["index"],
                    "phase": item["phase"],
                    "stratum": item["stratum"],
                    "result": result,
                    "novelty_vs_discovery": outcome["novelty_vs_discovery"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
        runtime.torch.cuda.empty_cache()

    completed_indexes = set(outcome_by_index)
    status = "complete" if completed_indexes == expected_indexes else "complete_with_failures"
    _recompute_novelty(outcomes, seed_coverage, manifest["discovery_queries"])
    _checkpoint(output_dir, manifest, outcomes, status=status)
    return load_json(output_dir / "campaign.json")


def main() -> None:
    args = _parse_arguments()
    result = run_campaign(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
