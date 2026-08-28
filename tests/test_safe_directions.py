from __future__ import annotations

import unittest

from embodied_silent_failures.safe_directions import (
    collapse_physical_failures,
    direction_group_summary,
)


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

    def test_physical_failure_collapse_weights_each_branch_once(self) -> None:
        records = []
        for layer, delta in ((0, 1.0), (1, 3.0)):
            records.append(
                {
                    "record_id": f"c001:layer{layer:02d}",
                    "physical_run": "c001-command-a",
                    "outcome_group": "silent_failure",
                    "analysis_split": "development",
                    "context_id": "c001",
                    "task_id": 1,
                    "episode_index": 2,
                    "phase": "middle",
                    "policy_step": 10,
                    "action_token_position": 3,
                    "safe_alarm_at_fault": False,
                    "safe_alarm_post_fault_any": False,
                    "selected_feature_l2": delta,
                    "selected_feature_normalized_l2": delta,
                    "absolute_monitor_increment_delta": delta,
                    "monitor_secant_sensitivity": delta,
                    "absolute_clean_gradient_cosine": delta,
                    "relu_gate_flip_fraction": delta,
                    "threshold_margin_after_fault": delta,
                }
            )
        branches = collapse_physical_failures(records)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]["member_interventions"], 2)
        self.assertEqual(branches[0]["selected_feature_l2"], 2.0)

    def test_physical_failure_collapse_rejects_outcome_disagreement(self) -> None:
        records = []
        for outcome in ("detected_failure", "silent_failure"):
            records.append(
                {
                    "physical_run": "shared",
                    "outcome_group": outcome,
                    "analysis_split": "holdout",
                    "context_id": "c",
                    "task_id": 0,
                    "episode_index": 0,
                    "phase": "late",
                    "policy_step": 2,
                    "action_token_position": 1,
                    "safe_alarm_at_fault": False,
                    "safe_alarm_post_fault_any": outcome == "detected_failure",
                    **{field: 1.0 for field in (
                        "selected_feature_l2",
                        "selected_feature_normalized_l2",
                        "absolute_monitor_increment_delta",
                        "monitor_secant_sensitivity",
                        "absolute_clean_gradient_cosine",
                        "relu_gate_flip_fraction",
                        "threshold_margin_after_fault",
                    )},
                }
            )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            collapse_physical_failures(records)


if __name__ == "__main__":
    unittest.main()
