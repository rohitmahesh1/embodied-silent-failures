import unittest
from pathlib import Path

from embodied_silent_failures.probe_openvla import _probe_step
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


if __name__ == "__main__":
    unittest.main()
