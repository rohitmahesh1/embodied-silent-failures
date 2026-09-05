from __future__ import annotations

import unittest

import numpy as np

from embodied_silent_failures.recovery_evidence import (
    direct_measurements,
    feature_dict,
    monitor_window_features,
)
from embodied_silent_failures.recovery_observability import (
    analyze_recovery_observability,
    trajectory_fold_assignments,
    within_context_concordance,
)


def _comparison(value: float) -> dict:
    return {
        "executed_command": {"symmetric_normalized_difference_l2": value},
        "changed_action_token_fraction": value,
        "mean_absolute_action_entropy_difference": value,
        "object-state": {"symmetric_normalized_difference_l2": value},
        "robot0_proprio-state": {"symmetric_normalized_difference_l2": value},
        "simulator_state": {"symmetric_normalized_difference_l2": value},
    }


class RecoveryObservabilityTests(unittest.TestCase):
    def test_monitor_features_separate_net_response_from_cancellation(self) -> None:
        arrays = {
            "monitor_increment_delta": np.asarray([[2.0, -2.0, 1.0]]),
            "absolute_monitor_increment_delta": np.asarray([[2.0, 2.0, 1.0]]),
            "selected_feature_normalized_l2": np.asarray([[3.0, 4.0, 0.0]]),
        }

        result = monitor_window_features(arrays, 0, 2)

        self.assertEqual(result["safe:response_signed_sum"], 0.0)
        self.assertAlmostEqual(
            result["safe:response_absolute_sum"], np.log1p(4.0)
        )
        self.assertEqual(result["safe:response_cancellation_fraction"], 1.0)

    def test_path_uses_only_checkpoints_observed_by_horizon(self) -> None:
        row = {
            "task_id": 0,
            "phase_fraction": 0.25,
            "comparisons": {
                str(value): _comparison(float(value))
                for value in (0, 1, 5, 10, 25)
            },
            "monitor_features": {"5": {"safe:value": 1.0}},
        }

        result = feature_dict(row, horizon=5, family="policy_physical_path")

        self.assertIn("physical_path:h1:object-state", result)
        self.assertIn("physical_path:h5:object-state", result)
        self.assertNotIn("physical_path:h10:object-state", result)
        self.assertIn("policy:h0:command_distance", result)
        self.assertIn("policy:h5:command_distance", result)
        self.assertNotIn("policy:h10:command_distance", result)

    def test_missing_terminal_policy_decision_is_retained(self) -> None:
        terminal = _comparison(1.0)
        del terminal["executed_command"]
        row = {
            "task_id": 0,
            "phase_fraction": 0.25,
            "comparisons": {"0": _comparison(1.0), "1": terminal},
            "monitor_features": {"1": {"safe:value": 1.0}},
        }

        result = feature_dict(row, horizon=1, family="policy")

        self.assertEqual(result["policy:h1:command_distance:missing"], 1.0)
        self.assertIn("policy:h1:changed_token_fraction", result)

    def test_direct_growth_uses_adjacent_declared_checkpoints(self) -> None:
        row = {
            "comparisons": {"1": _comparison(1.0), "5": _comparison(3.0)},
            "monitor_features": {
                "5": {
                    "safe:feature_displacement_rms": 1.0,
                    "safe:response_signed_sum": 2.0,
                    "safe:response_absolute_sum": 3.0,
                    "safe:response_cancellation_fraction": 0.25,
                }
            },
        }

        result = direct_measurements(row, 5)

        self.assertAlmostEqual(
            result["physical_object_log_growth"],
            np.log1p(3.0) - np.log1p(1.0),
        )

    def test_within_context_concordance_uses_only_same_context_pairs(self) -> None:
        rows = [
            {
                "context_id": "a",
                "task_id": 0,
                "episode_index": 0,
                "policy_failure": True,
            },
            {
                "context_id": "a",
                "task_id": 0,
                "episode_index": 0,
                "policy_failure": False,
            },
            {
                "context_id": "b",
                "task_id": 1,
                "episode_index": 0,
                "policy_failure": True,
            },
            {
                "context_id": "b",
                "task_id": 1,
                "episode_index": 0,
                "policy_failure": False,
            },
        ]

        result = within_context_concordance(
            rows,
            [2.0, 1.0, 10.0, 20.0],
            bootstrap_samples=20,
            seed=7,
        )

        self.assertEqual(result["comparable_contexts"], 2)
        self.assertEqual(result["failure_success_pairs"], 2)
        self.assertEqual(result["pair_weighted_concordance"], 0.5)

    def test_analysis_excludes_outcomes_reached_at_a_horizon(self) -> None:
        rows = [
            {
                "analysis_split": "development",
                "task_id": 0,
                "episode_index": 0,
                "policy_failure": True,
                "fault_step": 10,
                "faulted_length": 35,
                "control_length": 100,
            },
            {
                "analysis_split": "development",
                "task_id": 1,
                "episode_index": 1,
                "policy_failure": False,
                "fault_step": 10,
                "faulted_length": 100,
                "control_length": 100,
            },
            {
                "analysis_split": "holdout",
                "task_id": 2,
                "episode_index": 2,
                "policy_failure": False,
                "fault_step": 10,
                "faulted_length": 100,
                "control_length": 100,
            },
        ]
        seen = {}

        def capture(development, holdout, *, horizon, **_kwargs):
            seen[horizon] = (len(development), len(holdout))
            return {}

        import embodied_silent_failures.recovery_observability as module

        original = module.analyze_horizon
        module.analyze_horizon = capture
        try:
            analyze_recovery_observability(
                rows, folds=2, bootstrap_samples=1, seed=1
            )
        finally:
            module.analyze_horizon = original

        self.assertEqual(seen[10], (2, 1))
        self.assertEqual(seen[25], (1, 1))

    def test_fold_assignment_is_reusable_after_removing_a_row(self) -> None:
        rows = [
            {"task_id": task, "episode_index": episode}
            for task in range(2)
            for episode in range(3)
            for _duplicate in range(2)
        ]

        assignments = trajectory_fold_assignments(rows, 3)

        self.assertEqual(len(assignments), 6)
        for row in rows[:-1]:
            self.assertIn(
                f"task{row['task_id']}:episode{row['episode_index']}", assignments
            )


if __name__ == "__main__":
    unittest.main()
