from __future__ import annotations

import unittest

import numpy as np

from embodied_silent_failures.action_monitor_analysis import (
    attach_safe_arrays,
    rank_cdf,
    rank_mismatch_diagnostic,
)


class ActionMonitorAnalysisTests(unittest.TestCase):
    def test_safe_arrays_keep_fault_step_distinct_from_later_response(self) -> None:
        arrays = {
            "monitor_increment_delta": np.asarray([[0.2, -0.5, 0.1]]),
            "selected_feature_l2": np.asarray([[3.0, 4.0, 5.0]]),
        }

        row = attach_safe_arrays({}, arrays, 0)

        self.assertAlmostEqual(row["absolute_safe_response_at_fault"], 0.2)
        self.assertAlmostEqual(row["later_safe_response_absolute_sum"], 0.6)
        self.assertAlmostEqual(row["feature_displacement_at_fault"], 3.0)

    def test_rank_cdf_uses_only_reference_distribution(self) -> None:
        self.assertEqual(rank_cdf([1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0]), [0.0, 0.5, 1.0])

    def test_rank_mismatch_cutoff_is_fit_on_development(self) -> None:
        development = [
            {
                "same_feature_action_js_at_fault": float(index),
                "absolute_safe_response_at_fault": float(10 - index),
                "policy_failure": index > 7,
            }
            for index in range(10)
        ]
        holdout = [
            {
                "same_feature_action_js_at_fault": 9.0,
                "absolute_safe_response_at_fault": 1.0,
                "policy_failure": True,
            },
            {
                "same_feature_action_js_at_fault": 1.0,
                "absolute_safe_response_at_fault": 9.0,
                "policy_failure": False,
            },
        ]

        result = rank_mismatch_diagnostic(development, holdout)

        self.assertEqual(result["holdout_roc_auc"], 1.0)
        self.assertEqual(result["holdout_top_fifth"]["failures"], 1)


if __name__ == "__main__":
    unittest.main()
