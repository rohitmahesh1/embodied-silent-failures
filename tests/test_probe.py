import unittest
from pathlib import Path

from embodied_silent_failures.probe_openvla import _probe_step
from embodied_silent_failures.probe_stale_image import _action_metrics, _parse_lags
from embodied_silent_failures.replay import CleanTrace


class ProbeTests(unittest.TestCase):
    def trace(self, gripper):
        return CleanTrace(
            result={},
            rows=[{"action/dgripper": str(value)} for value in gripper],
            hidden_states=[],
            observations={},
            source_dir=Path("/clean"),
        )

    def test_selects_first_gripper_transition_after_minimum(self) -> None:
        trace = self.trace([-1, -1, 1, 1, -1])
        self.assertEqual(
            _probe_step(trace, "first_gripper_transition", fixed_step=0, minimum=3),
            4,
        )

    def test_rejects_trace_without_eligible_transition(self) -> None:
        with self.assertRaises(ValueError):
            _probe_step(
                self.trace([-1, -1, -1]),
                "first_gripper_transition",
                fixed_step=0,
                minimum=1,
            )

    def test_stale_image_lags_are_positive_unique_and_sorted(self) -> None:
        self.assertEqual(_parse_lags("8, 1,4"), [1, 4, 8])
        for value in ("", "0,1", "1,1", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_lags(value)

    def test_stale_action_metrics_separate_motion_and_gripper(self) -> None:
        class Arrays:
            @staticmethod
            def asarray(value, dtype=None):
                return value if isinstance(value, Vector) else Vector(value)

            class linalg:
                @staticmethod
                def norm(value):
                    return sum(item * item for item in value.values) ** 0.5

            @staticmethod
            def max(value):
                return max(value.values)

            @staticmethod
            def abs(value):
                return Vector(abs(item) for item in value.values)

            @staticmethod
            def any(value):
                return any(value.values)

        class Vector:
            def __init__(self, values):
                self.values = list(values)

            def __sub__(self, other):
                return Vector(a - b for a, b in zip(self.values, other.values))

            def __iter__(self):
                return iter(self.values)

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return Vector(self.values[key])
                return self.values[key]

            def __ne__(self, other):
                return Vector(a != b for a, b in zip(self.values, other.values))

        metrics = _action_metrics(
            Arrays,
            Vector([0, 0, 0, 0, 0, 0, -1]),
            Vector([3, 4, 0, 0, 0, 0, 1]),
        )
        self.assertEqual(metrics["translation_l2"], 5.0)
        self.assertTrue(metrics["gripper_changed"])


if __name__ == "__main__":
    unittest.main()
