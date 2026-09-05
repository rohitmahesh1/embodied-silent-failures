from __future__ import annotations

import unittest

from embodied_silent_failures.safe_trajectory_analysis import (
    attach_temporal_geometry,
    binary_metric_summary,
    derived_geometry,
    quiet_failure_summary,
)


def _record(value: float, *, failure: bool) -> dict:
    return {
        "task_id": 0,
        "episode_index": 0,
        "policy_failure": failure,
        "outcome_group": "silent_failure" if failure else "successful_continuation",
        "feature_displacement_l2_energy": value,
        "normalized_feature_displacement_l2_energy": value,
        "safe_response_signed_sum": value,
        "safe_response_absolute_sum": value * 2,
        "safe_response_cancellation_fraction": 0.5,
        "gradient_projection_absolute_sum": value,
        "gradient_alignment_fraction": 0.1,
        "mean_absolute_gradient_cosine": 0.1,
        "linearization_error_absolute_sum": value,
        "linearization_error_fraction": 0.2,
        "mean_relu_gate_flip_fraction": 0.3,
    }


class SafeTrajectoryAnalysisTests(unittest.TestCase):
    def test_derived_response_rate_uses_total_feature_movement(self) -> None:
        record = derived_geometry(_record(2.0, failure=True))
        self.assertEqual(record["safe_response_per_feature_displacement"], 2.0)
        self.assertEqual(record["safe_response_net_fraction"], 0.5)

    def test_binary_summary_preserves_metric_direction(self) -> None:
        rows = [_record(4.0, failure=True), _record(1.0, failure=False)]
        result = binary_metric_summary(
            rows,
            metric="feature_displacement_l2_energy",
            positive=lambda row: row["policy_failure"],
            negative=lambda row: not row["policy_failure"],
        )
        self.assertEqual(result["roc_auc_for_larger_value"], 1.0)
        self.assertEqual(result["median_difference_positive_minus_negative"], 3.0)

    def test_temporal_geometry_separates_fault_step_from_later_response(self) -> None:
        import numpy as np

        arrays = {
            "selected_feature_l2": np.asarray([[2.0, 3.0, 4.0]]),
            "monitor_increment_delta": np.asarray([[0.1, -0.2, 0.3]]),
            "clean_gradient_dot_delta": np.asarray([[0.2, -0.4, 0.5]]),
        }
        result = attach_temporal_geometry({}, arrays, 0)
        self.assertAlmostEqual(result["safe_response_at_fault"], 0.1)
        self.assertAlmostEqual(result["later_safe_response_signed_sum"], 0.1)
        self.assertAlmostEqual(result["later_safe_response_absolute_sum"], 0.5)

    def test_quiet_failure_group_uses_absolute_net_response(self) -> None:
        rows = [_record(float(value), failure=True) for value in range(1, 9)]
        rows[0]["safe_response_signed_sum"] = -1.0
        result = quiet_failure_summary(rows)
        self.assertEqual(result["failures"], 8)
        self.assertEqual(result["quiet_failures"], 2)


if __name__ == "__main__":
    unittest.main()
