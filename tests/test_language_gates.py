from __future__ import annotations

import unittest

from embodied_silent_failures.language_gates import (
    command_signal_auc,
    equal_count_fifths,
    physical_command_branches,
    roc_auc,
)


class LanguageGateTests(unittest.TestCase):
    def test_auc_uses_average_ranks_for_ties(self) -> None:
        self.assertEqual(roc_auc([False, True], [0.0, 1.0]), 1.0)
        self.assertEqual(roc_auc([False, True], [1.0, 1.0]), 0.5)

    def test_cluster_bootstrap_uses_whole_trajectories(self) -> None:
        rows = [
            {
                "eligible_causal_outcome": True,
                "command_changed": changed,
                "signal": value,
                "task_id": task,
                "episode_index": episode,
            }
            for task, episode, changed, value in (
                (0, 0, False, 0.0),
                (0, 0, True, 1.0),
                (1, 0, False, 0.0),
                (1, 0, True, 1.0),
            )
        ]
        result = command_signal_auc(rows, "signal", bootstrap_samples=20, seed=3)
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["bootstrap_valid_samples"], 20)
        self.assertEqual(result["trajectory_cluster_bootstrap_95"], [1.0, 1.0])

    def test_physical_branches_collapse_exact_commands(self) -> None:
        rows = []
        scores = {}
        for layer in (3, 4):
            record_id = f"c001:layer{layer:02d}"
            rows.append(
                {
                    "record_id": record_id,
                    "eligible_causal_outcome": True,
                    "command_changed": True,
                    "physical_run": "c001-command-abc",
                    "analysis_split": "development",
                    "context_id": "c001",
                    "task_id": 0,
                    "episode_index": 2,
                    "phase": "middle",
                    "policy_step": 10,
                    "action_token_position": 2,
                    "command_id": "abc",
                    "layer_index": layer,
                    "task_failure": True,
                    "operational_silent_failure": True,
                    "safe_alarm_at_fault": False,
                    "safe_alarm_within_25": False,
                    "safe_alarm_post_fault_any": False,
                }
            )
            scores[record_id] = {
                "local_measurements": {
                    "clean_executed_command": [0.0] * 7,
                    "faulted_executed_command": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                }
            }
        branches = physical_command_branches(rows, scores)
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]["member_layers"], [3, 4])
        self.assertEqual(branches[0]["delta_dz"], 0.5)
        self.assertEqual(branches[0]["command_l2"], 0.5)

    def test_physical_branch_rejects_outcome_disagreement(self) -> None:
        rows = []
        scores = {}
        for layer, failure in ((0, False), (1, True)):
            record_id = f"r{layer}"
            rows.append(
                {
                    "record_id": record_id,
                    "eligible_causal_outcome": True,
                    "command_changed": True,
                    "physical_run": "same",
                    "analysis_split": "holdout",
                    "context_id": "c",
                    "task_id": 0,
                    "episode_index": 0,
                    "phase": "early",
                    "policy_step": 1,
                    "action_token_position": 0,
                    "command_id": "x",
                    "layer_index": layer,
                    "task_failure": failure,
                    "operational_silent_failure": False,
                    "safe_alarm_at_fault": False,
                    "safe_alarm_within_25": False,
                    "safe_alarm_post_fault_any": False,
                }
            )
            scores[record_id] = {
                "local_measurements": {
                    "clean_executed_command": [0.0] * 7,
                    "faulted_executed_command": [1.0] + [0.0] * 6,
                }
            }
        with self.assertRaisesRegex(ValueError, "disagree"):
            physical_command_branches(rows, scores)

    def test_equal_count_fifths_keep_every_branch(self) -> None:
        branches = [
            {
                "physical_run": f"run-{index}",
                "command_l2": float(index),
                "task_failure": index % 2 == 0,
                "operational_silent_failure": False,
            }
            for index in range(11)
        ]
        groups = equal_count_fifths(branches)
        self.assertEqual(sum(group["branches"] for group in groups), 11)
        self.assertEqual([group["branches"] for group in groups], [2, 2, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
