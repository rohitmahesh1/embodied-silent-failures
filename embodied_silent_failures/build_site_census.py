from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.evidence_graph.census import (
    DEFAULT_ACTION_INTERFACE,
    DEFAULT_MONITOR_INTERFACE,
    build_site_census,
    source_artifacts,
)
from embodied_silent_failures.evidence_graph.record import read_events
from embodied_silent_failures.provenance import load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an auditable sampling census from an evidence graph."
    )
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--raw-trace", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-interface", default=DEFAULT_ACTION_INTERFACE)
    parser.add_argument("--monitor-interface", default=DEFAULT_MONITOR_INTERFACE)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    audit = load_json(args.audit)
    if audit.get("passed") is not True:
        raise ValueError("source evidence graph does not have a passing audit")
    result = build_site_census(
        load_json(args.graph),
        read_events(args.raw_trace),
        action_interface=args.action_interface,
        monitor_interface=args.monitor_interface,
    )
    result["sources"] = source_artifacts(args.graph, args.raw_trace, args.audit)
    write_json_atomic(args.output, result)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
