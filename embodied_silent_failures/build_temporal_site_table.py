from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.evidence_graph.census import (
    DEFAULT_ACTION_INTERFACE,
    DEFAULT_MONITOR_INTERFACE,
)
from embodied_silent_failures.evidence_graph.record import read_events
from embodied_silent_failures.evidence_graph.temporal_sites import (
    build_temporal_site_table,
    csv_rows,
)
from embodied_silent_failures.provenance import file_sha256, load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an auditable table of temporal-replacement fault sites."
    )
    parser.add_argument("--source-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--action-interface", default=DEFAULT_ACTION_INTERFACE)
    parser.add_argument("--monitor-interface", default=DEFAULT_MONITOR_INTERFACE)
    return parser.parse_args()


def _source_id(path: Path, metadata: dict[str, Any]) -> str:
    task_id = metadata.get("task_id", "unknown")
    episode = metadata.get("episode_index", "unknown")
    condition = metadata.get("condition", "unknown")
    return f"{path.parent.name}/{path.name}:task{task_id}:ep{episode}:{condition}"


def _sources(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        raw_path = path / "raw.jsonl"
        graph_path = path / "graph.json"
        audit_path = path / "audit.json"
        audit = load_json(audit_path)
        if audit.get("passed") is not True:
            raise ValueError(
                f"source evidence graph does not have a passing audit: {path}"
            )
        events = read_events(raw_path)
        metadata = events[0].get("details", {}) if events else {}
        yield {
            "source_id": _source_id(path, metadata),
            "graph": load_json(graph_path),
            "events": events,
            "artifacts": {
                "directory": str(path.resolve()),
                "raw_trace": {
                    "path": str(raw_path.resolve()),
                    "sha256": file_sha256(raw_path),
                },
                "graph": {
                    "path": str(graph_path.resolve()),
                    "sha256": file_sha256(graph_path),
                },
                "audit": {
                    "path": str(audit_path.resolve()),
                    "sha256": file_sha256(audit_path),
                    "passed": True,
                },
            },
        }


def main() -> None:
    args = _arguments()
    table = build_temporal_site_table(
        _sources(args.source_dir),
        action_interface=args.action_interface,
        monitor_interface=args.monitor_interface,
    )
    write_json_atomic(args.output, table)
    write_csv_atomic(args.csv, csv_rows(table))
    print(json.dumps(table["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
