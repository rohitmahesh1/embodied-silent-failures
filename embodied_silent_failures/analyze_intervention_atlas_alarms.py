from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.intervention_atlas_alarm_specificity import (
    analyze_alarm_specificity,
)
from embodied_silent_failures.provenance import file_sha256, git_state


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAFE alarms on atlas failures with matched controls."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        required=True,
        metavar=("SITE_ANALYSIS", "PHYSICAL_SCORES"),
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", default="0.1")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = analyze_alarm_specificity(
        [(site, physical) for site, physical in args.pair], alpha=args.alpha
    )
    result["analysis_code"] = {
        **git_state(Path(__file__).resolve().parents[1]),
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "methods_sha256": file_sha256(
            Path(__file__).with_name("intervention_atlas_alarm_specificity.py")
        ),
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
