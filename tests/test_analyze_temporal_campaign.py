import tempfile
import unittest
from pathlib import Path

import numpy as np

from embodied_silent_failures.analyze_temporal_campaign import _safe_index


class AnalyzeTemporalCampaignTests(unittest.TestCase):
    def test_primary_alpha_matches_float32_archive_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "scores.npz"
            np.savez_compressed(
                archive,
                runs=np.asarray(["a0000"]),
                lengths=np.asarray([2]),
                scores=np.asarray([[0.1, 0.2]]),
                alphas=np.asarray([0.05, 0.1, 0.2], dtype=np.float32),
                bands=np.asarray(
                    [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]], dtype=np.float32
                ),
            )

            indexed = _safe_index(
                {"monitor": {"primary_alpha": 0.1}}, archive
            )

        np.testing.assert_array_equal(
            indexed["a0000"]["band"], np.asarray([0.6, 0.6], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
