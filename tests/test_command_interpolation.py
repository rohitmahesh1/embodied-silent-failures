from __future__ import annotations

import unittest

from embodied_silent_failures.command_interpolation import interpolate_command


class CommandInterpolationTests(unittest.TestCase):
    def test_interpolation_preserves_categorical_gripper(self) -> None:
        clean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        failed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]
        self.assertEqual(
            interpolate_command(clean, failed, 0.25),
            [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.0],
        )

    def test_interpolation_rejects_gripper_blending(self) -> None:
        with self.assertRaisesRegex(ValueError, "categorical"):
            interpolate_command([0.0] * 6 + [-1.0], [0.0] * 6 + [1.0], 0.5)


if __name__ == "__main__":
    unittest.main()
