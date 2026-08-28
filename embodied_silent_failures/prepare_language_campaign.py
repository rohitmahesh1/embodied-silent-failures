from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.language_campaign import (
    build_language_campaign_manifest,
    validate_language_campaign_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the OpenVLA language-block residual-risk campaign."
    )
    parser.add_argument("--site-table", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    manifest = build_language_campaign_manifest(
        args.site_table,
        args.clean_root,
        seed=args.seed,
    )
    validate_language_campaign_manifest(manifest)
    write_json_atomic(args.output, manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
