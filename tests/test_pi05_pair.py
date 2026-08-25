import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from embodied_silent_failures.pi05_pair import (
    _image_difference,
    branch_order,
    pair_directory,
    pair_terminal_state,
    prepare_pair,
)
from embodied_silent_failures.plan import Trial


class Pi05PairTests(unittest.TestCase):
    @unittest.skipUnless(np is not None, "NumPy is installed in the pi0.5 runtime")
    def test_image_difference_reports_rounding_scale_pixel_change(self) -> None:
        expected = np.zeros((2, 3, 3), dtype=np.uint8)
        actual = expected.copy()
        actual[1, 2, 0] = 1

        self.assertEqual(
            _image_difference(np, expected, actual),
            {
                "maximum_absolute_channel_error": 1,
                "mean_absolute_channel_error": 1 / 18,
                "changed_pixels": 1,
                "total_pixels": 6,
            },
        )

    def test_branch_order_balances_without_changing_branch_identity(self) -> None:
        self.assertEqual(
            branch_order("stale_main_camera", 0), ("current", "stale")
        )
        self.assertEqual(
            branch_order("stale_main_camera", 1), ("stale", "current")
        )
        self.assertEqual(
            set(branch_order("current_current_null", 1)),
            {"current_a", "current_b"},
        )

    def test_prepare_pair_removes_an_incomplete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = Trial(2, 4)
            pair = pair_directory(root, trial)
            pair.mkdir(parents=True)
            (pair / "partial.pkl").write_bytes(b"partial")

            state = prepare_pair(root, trial, resume=True)

            self.assertIsNone(state)
            self.assertFalse(pair.exists())

    def test_prepare_pair_rejects_a_completion_without_both_branches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = Trial(2, 4)
            pair = pair_directory(root, trial)
            pair.mkdir(parents=True)
            (pair / "pair.complete.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "task_id": 2,
                        "episode_index": 4,
                        "pair_condition": "stale_main_camera",
                        "branches": {},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(pair_terminal_state(root, trial), "complete")
            with self.assertRaises(ValueError):
                prepare_pair(root, trial, resume=True)


if __name__ == "__main__":
    unittest.main()
