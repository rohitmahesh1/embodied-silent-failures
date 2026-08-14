import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_silent_failures.artifacts import write_json_atomic


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen evidence-graph campaign as resumable stages."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--openvla-root", required=True, type=Path)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument(
        "--paired-clean-dir", action="append", default=[], type=Path
    )
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--require-canary", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stall-minutes", type=int, default=30)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--workspace-quota-gb", type=float, default=100.0)
    parser.add_argument("--minimum-free-gb", type=float, default=15.0)
    args = parser.parse_args()
    if args.max_attempts <= 0:
        raise ValueError("max attempts must be positive")
    if args.poll_seconds <= 0 or args.stall_minutes <= 0:
        raise ValueError("poll and stall intervals must be positive")
    if args.minimum_free_gb <= 0 or args.workspace_quota_gb <= args.minimum_free_gb:
        raise ValueError("workspace quota must exceed the minimum free space")
    return args


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _terminal_counts(output_dir: Path) -> tuple[int, int]:
    return (
        len(list(output_dir.glob("*.complete.json"))),
        len(list(output_dir.glob("*.excluded.json"))),
    )


def _workspace_used_bytes(workspace: Path) -> int:
    result = subprocess.run(
        ["du", "-s", "-B1", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split()[0])


def _check_storage(args: argparse.Namespace) -> dict[str, float]:
    used_gb = _workspace_used_bytes(args.workspace) / 1_000_000_000
    free_gb = args.workspace_quota_gb - used_gb
    if free_gb < args.minimum_free_gb:
        raise RuntimeError(
            f"workspace has about {free_gb:.1f} GB free under the configured quota; "
            f"at least {args.minimum_free_gb:.1f} GB is required"
        )
    return {"used_gb": used_gb, "estimated_free_gb": free_gb}


def _stage_command(
    args: argparse.Namespace, stage: dict[str, Any]
) -> tuple[list[str], Path, Path]:
    output_dir = args.output_root / stage["name"]
    evidence_dir = args.output_root / "evidence" / stage["name"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "embodied_silent_failures.run_openvla",
        "--checkpoint",
        str(args.checkpoint),
        "--openvla-root",
        str(args.openvla_root),
        "--libero-root",
        str(args.libero_root),
        "--task-suite",
        "libero_10",
        "--seed",
        "7",
        "--wait-steps",
        "10",
        "--save-video" if stage.get("save_video", False) else "--no-save-video",
        "--resume",
        "--output-dir",
        str(output_dir),
        "--evidence-dir",
        str(evidence_dir),
    ]
    if stage["kind"] == "clean":
        command.extend(("--trial-manifest", stage["manifest"]))
    elif stage["kind"] == "stale_image":
        command.extend(
            (
                "--stale-image-manifest",
                stage["manifest"],
                "--image-input-mode",
                stage["image_input_mode"],
            )
        )
        for directory in args.paired_clean_dir:
            command.extend(("--paired-clean-dir", str(directory)))
    else:
        raise ValueError(f"unsupported campaign stage kind: {stage['kind']}")
    for policy_step in stage["trace_steps"]:
        command.extend(("--evidence-trace-step", str(policy_step)))
    return command, output_dir, evidence_dir


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _validate_stage(
    stage: dict[str, Any], output_dir: Path, evidence_dir: Path
) -> dict[str, Any]:
    complete_paths = sorted(output_dir.glob("*.complete.json"))
    excluded_paths = sorted(output_dir.glob("*.excluded.json"))
    if len(complete_paths) + len(excluded_paths) != int(stage["expected_trials"]):
        raise ValueError(
            f"stage has {len(complete_paths)} complete and {len(excluded_paths)} "
            f"excluded trials; expected {stage['expected_trials']} terminal records"
        )
    if excluded_paths and not stage["allow_exclusions"]:
        raise ValueError("stage unexpectedly excluded one or more trials")

    completed = []
    for path in complete_paths:
        result = _load_json(path)
        evidence = result.get("evidence_graph")
        if not isinstance(evidence, dict) or evidence.get("audit_passed") is not True:
            raise ValueError(f"completion has no passing evidence audit: {path}")
        directory = evidence_dir / f"task{result['task_id']}--ep{result['episode_index']}"
        audit = _load_json(directory / "audit.json")
        composition = _load_json(directory / "composition.json")
        if audit.get("passed") is not True:
            raise ValueError(f"evidence audit did not pass: {directory}")
        requested = sorted(int(value) for value in stage["trace_steps"])
        if not requested:
            requested = [int(result["fault"]["policy_step"])]
        if composition.get("traced_steps") != requested:
            raise ValueError(f"evidence trace steps disagree for {directory}")
        completed.append([int(result["task_id"]), int(result["episode_index"])])

    excluded = []
    for path in excluded_paths:
        result = _load_json(path)
        excluded.append([int(result["task_id"]), int(result["episode_index"])])
    if "expected_complete" in stage and sorted(completed) != sorted(stage["expected_complete"]):
        raise ValueError("canary completion set does not match its frozen expectation")
    if "expected_excluded" in stage and sorted(excluded) != sorted(stage["expected_excluded"]):
        raise ValueError("canary exclusion set does not match its frozen expectation")
    return {"complete": completed, "excluded": excluded}


def _write_status(campaign_dir: Path, value: dict[str, Any]) -> None:
    write_json_atomic(campaign_dir / "status.json", value)


def _run_stage(
    args: argparse.Namespace,
    stage: dict[str, Any],
    campaign_started: str,
) -> dict[str, Any]:
    command, output_dir, evidence_dir = _stage_command(args, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.campaign_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    attempts = []

    try:
        validation = _validate_stage(stage, output_dir, evidence_dir)
        return {"name": stage["name"], "state": "complete", "resumed": True, **validation}
    except (FileNotFoundError, ValueError):
        pass

    for attempt in range(1, args.max_attempts + 1):
        storage = _check_storage(args)
        log_path = log_dir / f"{stage['name']}--attempt-{attempt}.log"
        started = _now()
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            previous_counts = _terminal_counts(output_dir)
            last_progress = time.monotonic()
            stalled = False
            while process.poll() is None:
                time.sleep(args.poll_seconds)
                counts = _terminal_counts(output_dir)
                if counts != previous_counts:
                    previous_counts = counts
                    last_progress = time.monotonic()
                if time.monotonic() - last_progress > args.stall_minutes * 60:
                    stalled = True
                    _stop_process(process)
                    break
                _write_status(
                    args.campaign_dir,
                    {
                        "schema_version": 1,
                        "state": "running",
                        "campaign_started_at": campaign_started,
                        "updated_at": _now(),
                        "stage": stage["name"],
                        "attempt": attempt,
                        "complete_trials": counts[0],
                        "excluded_trials": counts[1],
                        "storage": storage,
                    },
                )
            return_code = process.wait()
        attempts.append(
            {
                "attempt": attempt,
                "started_at": started,
                "finished_at": _now(),
                "return_code": return_code,
                "stalled": stalled,
                "log": str(log_path.resolve()),
            }
        )
        if return_code == 0:
            try:
                validation = _validate_stage(stage, output_dir, evidence_dir)
                return {
                    "name": stage["name"],
                    "state": "complete",
                    "attempts": attempts,
                    **validation,
                }
            except (FileNotFoundError, ValueError) as error:
                attempts[-1]["validation_error"] = str(error)
    return {"name": stage["name"], "state": "failed", "attempts": attempts}


def main() -> None:
    args = _parse_arguments()
    campaign = _load_json(args.campaign_dir / "campaign.json")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.campaign_dir / "campaign.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another campaign process holds the campaign lock") from error

    if args.require_canary:
        canary_status = _load_json(args.campaign_dir / "canary-status.json")
        if canary_status.get("state") != "complete":
            raise RuntimeError("the campaign requires a passing canary")

    stages = [campaign["canary"]] if args.canary_only else campaign["stages"]
    campaign_started = _now()
    results = []
    for index, stage in enumerate(stages, start=1):
        result = _run_stage(args, stage, campaign_started)
        results.append(result)
        _write_status(
            args.campaign_dir,
            {
                "schema_version": 1,
                "state": "running",
                "campaign_started_at": campaign_started,
                "updated_at": _now(),
                "completed_stages": index,
                "total_stages": len(stages),
                "last_stage": result,
            },
        )

    summary = {
        "schema_version": 1,
        "state": "complete" if all(item["state"] == "complete" for item in results) else "partial",
        "campaign_started_at": campaign_started,
        "finished_at": _now(),
        "stages": results,
    }
    path = args.campaign_dir / ("canary-status.json" if args.canary_only else "status.json")
    write_json_atomic(path, summary)
    print(json.dumps(summary, indent=2))
    if summary["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
