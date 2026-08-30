import unittest

from embodied_silent_failures.analyze_language_campaign import (
    analysis_row,
    clustered_rate,
    expected_context_count,
)


class LanguageCampaignAnalysisTests(unittest.TestCase):
    def test_command_group_member_is_one_intervention_with_shared_physical_evidence(self) -> None:
        record = {
            "record_id": "c000:layer01",
            "status": "scored",
            "composition_verified": True,
            "context_id": "c000",
            "context": {
                "analysis_split": "development",
                "task_id": 0,
                "episode_index": 2,
                "phase": "early",
                "worker_shard": 0,
                "policy_step": 10,
                "action_token_position": 0,
            },
            "layer_index": 1,
            "local_measurements": {
                "site_id": "site",
                "executed_command": {"exact_equal": False, "difference_l2": 1.0},
                "injection": {},
                "propagation": [],
                "safe_feature": {},
            },
            "command_id": "same-command",
            "command_group_size": 27,
            "physical_run": "c000-command-same",
            "terminal_evidence": "observed_exact_command_branch",
            "control_success": True,
            "terminal_success": False,
            "monitor_horizon": "complete_physical_trace",
            "alarms": {
                "0.1": {
                    "post_fault_any": {"triggered": False},
                    "within_10_steps": {"triggered": False},
                    "within_25_steps": {"triggered": False},
                }
            },
            "alarm_at_fault": False,
            "alarm_before_fault": False,
            "control_alarm_at_fault": False,
        }

        row = analysis_row(record, "0.1")

        self.assertTrue(row["eligible_causal_outcome"])
        self.assertTrue(row["task_failure"])
        self.assertTrue(row["operational_silent_failure"])
        self.assertEqual(row["command_group_size"], 27)
        self.assertEqual(row["physical_run"], "c000-command-same")

    def test_failed_control_has_no_causal_outcome(self) -> None:
        record = {
            "record_id": "c000:layer00",
            "status": "scored",
            "composition_verified": True,
            "context_id": "c000",
            "context": {
                "analysis_split": "development",
                "task_id": 0,
                "episode_index": 2,
                "phase": "early",
                "worker_shard": 0,
                "policy_step": 10,
                "action_token_position": 0,
            },
            "layer_index": 0,
            "local_measurements": {
                "site_id": "site",
                "executed_command": {"exact_equal": True},
                "injection": {},
                "propagation": [],
                "safe_feature": {},
            },
            "control_success": False,
            "terminal_success": False,
            "monitor_horizon": "complete_physical_trace",
        }

        row = analysis_row(record, "0.1")

        self.assertFalse(row["eligible_causal_outcome"])
        self.assertIsNone(row["task_failure"])

    def test_bootstrap_resamples_trajectories_not_individual_rows(self) -> None:
        rows = [
            {"task_id": 0, "episode_index": 0, "event": True},
            {"task_id": 0, "episode_index": 0, "event": True},
            {"task_id": 0, "episode_index": 1, "event": False},
            {"task_id": 0, "episode_index": 1, "event": False},
        ]

        result = clustered_rate(rows, "event", samples=100, seed=3)

        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["trajectory_clusters"], 2)
        self.assertEqual(result["denominator"], 4)

    def test_expected_context_count_comes_from_score_coverage(self) -> None:
        scores = [
            {"coverage": {"planned_contexts": 105}},
            {"coverage": {"planned_contexts": 105}},
        ]

        self.assertEqual(expected_context_count(scores), 210)


if __name__ == "__main__":
    unittest.main()
