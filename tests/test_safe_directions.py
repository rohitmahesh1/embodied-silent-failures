from __future__ import annotations

import unittest

from embodied_silent_failures.safe_directions import direction_group_summary


class SafeDirectionTests(unittest.TestCase):
    def test_group_summary_uses_medians_and_keeps_missing_cosines(self) -> None:
        records = [
            {
                "selected_feature_l2": 1.0,
                "selected_feature_normalized_l2": 0.5,
                "absolute_monitor_increment_delta": 0.1,
                "monitor_secant_sensitivity": 0.1,
                "absolute_clean_gradient_cosine": None,
                "relu_gate_flip_fraction": 0.0,
                "threshold_margin_after_fault": 2.0,
                "safe_alarm_at_fault": False,
            },
            {
                "selected_feature_l2": 3.0,
                "selected_feature_normalized_l2": 1.5,
                "absolute_monitor_increment_delta": 0.3,
                "monitor_secant_sensitivity": 0.1,
                "absolute_clean_gradient_cosine": 0.2,
                "relu_gate_flip_fraction": 0.5,
                "threshold_margin_after_fault": 4.0,
                "safe_alarm_at_fault": True,
            },
        ]
        summary = direction_group_summary(records)
        self.assertEqual(summary["interventions"], 2)
        self.assertEqual(summary["median_selected_feature_l2"], 2.0)
        self.assertEqual(summary["median_absolute_clean_gradient_cosine"], 0.2)
        self.assertEqual(summary["alarm_at_fault_interventions"], 1)


if __name__ == "__main__":
    unittest.main()
