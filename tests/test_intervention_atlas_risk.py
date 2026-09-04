import unittest

from embodied_silent_failures.intervention_atlas_risk import (
    GRAPH_FEATURES,
    flatten_record,
    model_features,
    rate_table,
)


def record(*, failure: bool, alarm: bool, topology: str = "shared") -> dict:
    change = {
        "normalized_difference_l2": 0.5,
    }
    return {
        "record_id": "c001:site-a",
        "context_id": "c001",
        "site_id": "site-a",
        "physical_run": "c001-control",
        "primary_eligible": True,
        "policy_failure": failure,
        "context": {
            "task_id": 1,
            "episode_index": 2,
            "phase": "middle",
            "phase_fraction": 0.5,
        },
        "sampling": {
            "stratum": f"{topology}:module_output:language:middle:direct",
            "site_inverse_probability_weight": 2.0,
        },
        "local_measurements": {
            "fault": {"comparison": change},
            "action_logits": change,
            "raw_action": change,
            "executed_command": change,
            "safe_input": change,
            "action_tokens": {"changed_token_count": 2},
        },
        "safe_contribution": {"faulted_minus_clean": -0.1},
        "safe_faulted_evidence": {"alarms": {"0.1": alarms(alarm)}},
        "safe_clean_evidence_same_suffix": {"alarms": {"0.1": alarms(False)}},
    }


def alarms(triggered: bool) -> dict:
    return {
        "within_5_steps": {"triggered": False},
        "within_10_steps": {"triggered": False},
        "within_25_steps": {"triggered": False},
        "post_fault_any": {"triggered": triggered},
    }


class InterventionAtlasRiskTests(unittest.TestCase):
    def test_flattens_policy_failure_and_monitor_miss_separately(self) -> None:
        row = flatten_record(record(failure=True, alarm=False), 0.1)

        self.assertTrue(row["policy_failure"])
        self.assertTrue(row["safe_miss_given_failure"])
        self.assertTrue(row["silent_failure"])
        self.assertEqual(row["changed_action_token_fraction"], 2 / 7)

    def test_graph_features_come_from_predeclared_stratum_fields(self) -> None:
        row = flatten_record(record(failure=False, alarm=False), 0.1)
        features = model_features(row, GRAPH_FEATURES)

        self.assertEqual(features["topology=shared"], 1.0)
        self.assertEqual(features["owner=language"], 1.0)

    def test_rate_table_separates_three_probabilities(self) -> None:
        rows = [
            flatten_record(record(failure=True, alarm=False), 0.1),
            flatten_record(record(failure=True, alarm=True), 0.1),
            flatten_record(record(failure=False, alarm=False), 0.1),
        ]
        for row in rows:
            row["graph_population_weight"] = 1.0

        rates = rate_table(rows)["all"]["sampled_population"]

        self.assertEqual(rates["policy_failure_probability"], 2 / 3)
        self.assertEqual(rates["safe_miss_given_policy_failure"], 1 / 2)
        self.assertEqual(rates["silent_failure_probability"], 1 / 3)
        timing = rate_table(rows)["all"]["detection_timing_given_policy_failure"]
        self.assertEqual(timing["within_25_steps"]["detected"], 0)


if __name__ == "__main__":
    unittest.main()
