import importlib.util
import unittest

from embodied_silent_failures.language_composition import (
    add_context_indicators,
    binary_metrics,
    intervention_composition_rows,
    model_specifications,
    physical_consequence_rows,
)


class LanguageCompositionTests(unittest.TestCase):
    def test_physical_rows_keep_one_distinct_command(self) -> None:
        analysis = [
            {
                "analysis_split": "development",
                "worker_shard": 0,
                "context_id": "c000",
                "task_id": 0,
                "episode_index": 0,
                "phase": "early",
                "policy_step": 2,
                "action_token_position": 0,
                "command_id": "command",
                "physical_run": "c000-command",
                "task_failure": True,
                "operational_silent_failure": True,
                "safe_alarm_at_fault": False,
                "safe_alarm_within_25": False,
                "safe_alarm_post_fault_any": False,
                "eligible_causal_outcome": True,
                "command_changed": True,
                "record_id": "c000:layer00",
                "layer_index": 0,
            },
            {
                "analysis_split": "development",
                "worker_shard": 0,
                "context_id": "c000",
                "task_id": 0,
                "episode_index": 0,
                "phase": "early",
                "policy_step": 2,
                "action_token_position": 0,
                "command_id": "command",
                "physical_run": "c000-command",
                "task_failure": True,
                "operational_silent_failure": True,
                "safe_alarm_at_fault": False,
                "safe_alarm_within_25": False,
                "safe_alarm_post_fault_any": False,
                "eligible_causal_outcome": True,
                "command_changed": True,
                "record_id": "c000:layer01",
                "layer_index": 1,
            },
        ]
        local = {
            "clean_executed_command": [0.0] * 7,
            "faulted_executed_command": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        scores = {
            record_id: {
                "local_measurements": local,
                "threshold_at_fault": 10.0,
                "score_at_fault": score,
                "control_score_at_fault": 4.0,
                "context": {"phase_fraction": 0.25},
            }
            for record_id, score in (
                ("c000:layer00", 6.0),
                ("c000:layer01", 6.00001),
            )
        }

        rows = physical_consequence_rows(analysis, scores)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["member_interventions"], 2)
        self.assertAlmostEqual(rows[0]["member_score_spread"], 0.00001)

        interventions = intervention_composition_rows(analysis, scores)

        self.assertEqual(len(interventions), 2)
        self.assertAlmostEqual(interventions[0]["fault_margin_ratio"], 0.4)
        self.assertAlmostEqual(interventions[1]["score_shift_ratio"], 0.200001)
        self.assertTrue(interventions[0]["monitor_missed"])

    def test_context_specification_names_its_limited_proxies(self) -> None:
        rows = [{"task_id": 2, "phase": "middle"}]
        add_context_indicators(rows, (1, 2))
        specifications = model_specifications((1, 2))
        names = [name for name, _ in specifications["consequence"]["coarse_context"]]

        self.assertEqual(rows[0]["task_1"], 0.0)
        self.assertEqual(rows[0]["task_2"], 1.0)
        self.assertEqual(rows[0]["phase_middle"], 1.0)
        self.assertIn("clean_dx", names)
        self.assertIn("task_2", names)
        self.assertIn("phase_middle", names)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
        "numerical analysis dependencies are not installed",
    )
    def test_binary_metrics_uses_exactly_one_fifth(self) -> None:
        rows = [
            {"outcome": value}
            for value in (
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            )
        ]

        result = binary_metrics(
            rows,
            "outcome",
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        )

        self.assertEqual(result["top_fifth"]["rows"], 2)
        self.assertEqual(result["top_fifth"]["positive_outcomes"], 1)
        self.assertEqual(result["top_fifth"]["enrichment_over_uniform"], 5.0)


if __name__ == "__main__":
    unittest.main()
