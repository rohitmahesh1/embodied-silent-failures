from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_silent_failures.language_product_state import extract_campaign


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join existing OpenVLA trajectories, SAFE traces, and terminal outcomes "
            "into a compact product-state dataset."
        )
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--language-scores", type=Path)
    parser.add_argument("--physical-scores", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Recompute every trajectory SHA-256 in addition to checking byte counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    import numpy as np

    scoring_dir = args.campaign_dir / "scoring"
    result = extract_campaign(
        np=np,
        campaign_dir=args.campaign_dir,
        language_scores_path=args.language_scores
        or scoring_dir / "language-safe.json",
        physical_scores_path=args.physical_scores
        or scoring_dir / "physical-safe.json",
        output_dir=args.output_dir,
        verify_source_hashes=args.verify_source_hashes,
    )
    print(json.dumps(result["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
