import unittest

import numpy as np

from embodied_silent_failures.language_evidence_factorial import (
    factorial_cells,
    paired_detection_summary,
    score_shift_summary,
)


class LanguageEvidenceFactorialTests(unittest.TestCase):
    def test_factorial_swaps_only_the_intervention_evidence(self) -> None:
        band = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        control = np.asarray([0.1, 0.2, 0.1, 0.1], dtype=np.float32)
        natural = np.asarray([0.1, 0.8, 0.1, 0.1], dtype=np.float32)

        result = factorial_cells(np, natural, control, band, 1)
        cells = result["cells"]

        self.assertTrue(
            cells["faulted_action_faulted_evidence"]["alarms"]
            ["at_intervention"]["triggered"]
        )
        self.assertFalse(
            cells["faulted_action_clean_evidence"]["alarms"]
            ["at_intervention"]["triggered"]
        )
        self.assertAlmostEqual(
            result["evidence_contribution"]["faulted_minus_clean"], 0.6
        )
        self.assertTrue(
            cells["clean_action_faulted_evidence"]["alarms"]
            ["at_intervention"]["triggered"]
        )
        self.assertFalse(
            cells["clean_action_clean_evidence"]["alarms"]
            ["at_intervention"]["triggered"]
        )

    def test_restoration_can_recover_suppressed_detection(self) -> None:
        band = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        control = np.asarray([0.1, 0.8, 0.1], dtype=np.float32)
        natural = np.asarray([0.1, 0.2, 0.1], dtype=np.float32)

        cells = factorial_cells(np, natural, control, band, 1)["cells"]

        self.assertFalse(
            cells["faulted_action_faulted_evidence"]["alarms"]
            ["at_intervention"]["triggered"]
        )
        self.assertTrue(
            cells["faulted_action_clean_evidence"]["alarms"]
            ["at_intervention"]["triggered"]
        )

    def test_cumulative_contribution_changes_later_scores(self) -> None:
        band = np.asarray([1.0, 0.5, 0.5], dtype=np.float32)
        control = np.asarray([0.1, 0.2, 0.4], dtype=np.float32)
        natural = np.asarray([0.1, 0.3, 0.45], dtype=np.float32)

        cells = factorial_cells(np, natural, control, band, 1)["cells"]

        self.assertFalse(
            cells["clean_action_clean_evidence"]["alarms"]
            ["post_fault_any"]["triggered"]
        )
        self.assertTrue(
            cells["clean_action_faulted_evidence"]["alarms"]
            ["post_fault_any"]["triggered"]
        )

    def test_paired_summary_counts_both_discordant_directions(self) -> None:
        rows = [
            {
                "task_id": 0,
                "episode_index": 1,
                "context_id": "c0",
                "command_id": "a",
                "physical_run": "run-a",
                "shared_at_intervention": False,
                "restored_at_intervention": True,
                "fault_evidence_clean_action_at_intervention": False,
                "control_at_intervention": False,
            },
            {
                "task_id": 0,
                "episode_index": 2,
                "context_id": "c1",
                "command_id": "b",
                "physical_run": "run-b",
                "shared_at_intervention": True,
                "restored_at_intervention": False,
                "fault_evidence_clean_action_at_intervention": True,
                "control_at_intervention": False,
            },
        ]

        result = paired_detection_summary(
            rows, "at_intervention", samples=0, seed=5
        )

        self.assertEqual(result["restoration_recovers_detection"], 1)
        self.assertEqual(result["faulted_evidence_adds_detection"], 1)
        self.assertEqual(result["faulted_evidence_adds_clean_action_alarm"], 1)
        self.assertEqual(
            result["paired_detection_rate_difference"]["estimate"], 0.0
        )

    def test_zero_discordance_reports_bootstrap_limitation(self) -> None:
        row = {
            "task_id": 0,
            "episode_index": 1,
            "context_id": "c0",
            "command_id": "a",
            "physical_run": "run-a",
            "shared_at_intervention": False,
            "restored_at_intervention": False,
            "fault_evidence_clean_action_at_intervention": False,
            "control_at_intervention": False,
        }

        result = paired_detection_summary(
            [row], "at_intervention", samples=10, seed=5
        )

        self.assertEqual(result["trajectories_with_any_paired_difference"], 0)
        self.assertIsNotNone(result["zero_difference_bootstrap_limitation"])
        self.assertAlmostEqual(
            result["zero_difference_trajectory_probability_one_sided_95_upper"],
            0.95,
        )

    def test_score_shift_reports_trajectory_clustered_uncertainty(self) -> None:
        rows = [
            {
                "task_id": 0,
                "episode_index": 0,
                "faulted_minus_clean_score": 2.0,
                "clean_score_at_intervention": 1.0,
                "threshold_at_intervention": 5.0,
            },
            {
                "task_id": 0,
                "episode_index": 0,
                "faulted_minus_clean_score": 2.0,
                "clean_score_at_intervention": 1.0,
                "threshold_at_intervention": 5.0,
            },
            {
                "task_id": 0,
                "episode_index": 1,
                "faulted_minus_clean_score": -1.0,
                "clean_score_at_intervention": 1.0,
                "threshold_at_intervention": 5.0,
            },
        ]

        result = score_shift_summary(rows, samples=100, seed=7)

        self.assertEqual(result["interventions"], 3)
        self.assertEqual(result["trajectory_clusters"], 2)
        self.assertEqual(result["faulted_evidence_raises_score"], 2)
        self.assertEqual(result["faulted_evidence_lowers_score"], 1)


if __name__ == "__main__":
    unittest.main()
