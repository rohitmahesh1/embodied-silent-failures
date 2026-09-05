from __future__ import annotations

import unittest

import numpy as np

from embodied_silent_failures.atlas_path_alignment import (
    align_at_horizon,
    align_state_stream,
    is_reference_control,
)
from embodied_silent_failures.atlas_path_analysis import (
    exact_rejoin_diagnosis,
    paired_signal_difference,
    path_measurements,
)


def _trajectory(steps, values):
    return {
        "policy_steps": np.asarray(steps),
        "snapshot_stages": np.zeros(len(steps), dtype=np.uint8),
        "streams": {
            name: np.asarray(values, dtype=float)
            for name in (
                "simulator_state",
                "object-state",
                "robot0_proprio-state",
            )
        },
    }


class AtlasPathAlignmentTests(unittest.TestCase):
    def test_reference_control_is_not_a_faulted_continuation(self) -> None:
        self.assertTrue(
            is_reference_control(
                {"run": "c0001-control", "control_run": "c0001-control"}
            )
        )
        self.assertFalse(
            is_reference_control(
                {
                    "run": "c0001-command-abcd",
                    "control_run": "c0001-control",
                }
            )
        )

    def test_alignment_recovers_a_branch_that_is_one_step_behind(self) -> None:
        control = _trajectory([10, 11, 12, 13], [[0.0], [1.0], [2.0], [3.0]])
        faulted = _trajectory([10, 11, 12, 13], [[0.0], [0.0], [1.0], [2.0]])

        result = align_at_horizon(
            np, faulted, control, fault_step=10, horizon=2, window=2
        )

        stream = result["streams"]["simulator_state"]
        self.assertGreater(
            stream["same_clock"]["symmetric_normalized_difference_l2"], 0
        )
        self.assertEqual(
            stream["nearest_path"]["symmetric_normalized_difference_l2"], 0
        )
        self.assertEqual(stream["nearest_path"]["signed_step_offset"], -1)

    def test_alignment_never_uses_a_pre_intervention_state(self) -> None:
        control = _trajectory([9, 10, 11], [[7.0], [0.0], [1.0]])
        faulted = _trajectory([10, 11], [[0.0], [7.0]])

        result = align_at_horizon(
            np, faulted, control, fault_step=10, horizon=1, window=5
        )

        nearest = result["streams"]["simulator_state"]["nearest_path"]
        self.assertNotEqual(nearest["nearest_control_policy_step"], 9)
        self.assertGreater(nearest["symmetric_normalized_difference_l2"], 0)

    def test_terminal_post_action_snapshot_is_not_an_early_observation(self) -> None:
        control = _trajectory([10, 11], [[0.0], [1.0]])
        faulted = _trajectory([10, 11], [[0.0], [1.0]])
        faulted["snapshot_stages"][-1] = 1

        result = align_at_horizon(
            np, faulted, control, fault_step=10, horizon=1, window=5
        )

        self.assertIsNone(result)

    def test_equal_matches_prefer_the_closest_clock_step(self) -> None:
        result = align_state_stream(
            np,
            [1.0],
            [[1.0], [1.0], [1.0]],
            [9, 11, 13],
            target_step=12,
        )

        self.assertEqual(result["nearest_control_policy_step"], 11)
        self.assertEqual(result["signed_step_offset"], -1)

    def test_path_measurements_preserve_signed_phase_information(self) -> None:
        row = {
            "alignments": {
                "25": {
                    "25": {
                        "streams": {
                            "simulator_state": {
                                "same_clock": {
                                    "symmetric_normalized_difference_l2": 2.0
                                },
                                "nearest_path": {
                                    "symmetric_normalized_difference_l2": 0.5,
                                    "signed_step_offset": -3,
                                },
                                "unexplained_fraction": 0.25,
                            }
                        }
                    }
                }
            }
        }

        result = path_measurements(
            row, horizon=25, window=25, stream="simulator_state"
        )

        self.assertEqual(result["absolute_phase_offset"], 3)
        self.assertEqual(result["behind_steps"], 3)
        self.assertEqual(result["ahead_steps"], 0)
        self.assertEqual(result["unexplained_fraction"], 0.25)

    def test_paired_difference_uses_failure_direction(self) -> None:
        rows = [
            {
                "context_id": "a",
                "task_id": 0,
                "episode_index": 0,
                "faulted_success": False,
            },
            {
                "context_id": "a",
                "task_id": 0,
                "episode_index": 0,
                "faulted_success": True,
            },
            {
                "context_id": "b",
                "task_id": 1,
                "episode_index": 0,
                "faulted_success": False,
            },
            {
                "context_id": "b",
                "task_id": 1,
                "episode_index": 0,
                "faulted_success": True,
            },
        ]

        result = paired_signal_difference(
            rows,
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            bootstrap_samples=20,
            seed=4,
        )

        self.assertEqual(
            result["failure_roc_auc_candidate_minus_baseline"]["estimate"], 0.5
        )
        self.assertEqual(
            result[
                "within_context_concordance_candidate_minus_baseline"
            ]["estimate"],
            1.0,
        )

    def test_exact_rejoin_diagnosis_separates_trivial_recovery(self) -> None:
        def row(context, success, current, nearest):
            return {
                "context_id": context,
                "task_id": 0,
                "episode_index": 0,
                "faulted_success": success,
                "alignments": {
                    "25": {
                        "25": {
                            "streams": {
                                "simulator_state": {
                                    "same_clock": {
                                        "symmetric_normalized_difference_l2": current
                                    },
                                    "nearest_path": {
                                        "symmetric_normalized_difference_l2": nearest,
                                        "signed_step_offset": 0,
                                    },
                                    "unexplained_fraction": (
                                        nearest / current if current else 0.0
                                    ),
                                }
                            }
                        }
                    }
                },
            }

        rows = [
            row("a", True, 0.0, 0.0),
            row("a", False, 2.0, 1.0),
            row("b", True, 2.0, 1.0),
            row("b", False, 4.0, 3.0),
        ]

        result = exact_rejoin_diagnosis(rows, bootstrap_samples=10, seed=2)

        self.assertEqual(result["exact_same_clock_rejoins"]["successful"], 1)
        self.assertEqual(result["exact_same_clock_rejoins"]["failed"], 0)
        self.assertEqual(
            result["unexplained_fraction_after_excluding_exact_rejoins"]["rows"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
