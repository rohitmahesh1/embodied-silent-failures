from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.atlas_results import consolidate_intervention_atlas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate graph-derived intervention-atlas worker results."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = consolidate_intervention_atlas(args.manifest, args.campaign_dir)
    write_json_atomic(args.output, result)
    print(json.dumps(result["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
