from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_csv_atomic, write_json_atomic
from embodied_silent_failures.language_interface_prediction_evaluation import (
    analyze_prediction,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit local OpenVLA interface transformations and compose them to "
            "the policy-monitor fork and terminal residual-risk outcome."
        )
    )
    parser.add_argument(
        "--atlas-dir", action="append", dest="atlas_dirs", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--sketch-width", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if len(args.atlas_dirs) != 2:
        raise ValueError("interface prediction requires exactly two atlas shards")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap sample count cannot be negative")
    import numpy as np

    output, records = analyze_prediction(
        np,
        args.atlas_dirs,
        sketch_width=args.sketch_width,
        ridge_alpha=args.ridge_alpha,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json_atomic(args.output, output)
    write_csv_atomic(args.records_csv, records)
    print(
        json.dumps(
            {
                "population": output["population"],
                "one_step": {
                    name: value["overall"]
                    for name, value in output["one_step"].items()
                },
                "recursive_final_state": {
                    name: value["overall"]
                    for name, value in output["recursive_final_state"].items()
                },
                "residual_risk": output["residual_risk"]["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
