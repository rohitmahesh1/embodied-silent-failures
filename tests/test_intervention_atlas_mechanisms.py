import unittest

from embodied_silent_failures.intervention_atlas_mechanisms import (
    alarm_horizon_audit,
    context_outcome_audit,
    mechanism_features,
)


class InterventionAtlasMechanismTests(unittest.TestCase):
    def test_recovery_features_preserve_missing_horizon(self) -> None:
        row = {
            "task_id": 2,
            "phase_fraction": 0.5,
            "comparisons": {},
        }

        features = mechanism_features(row, "task_phase_recovery")

        self.assertEqual(features["task=2"], 1.0)
        self.assertEqual(features["h25:missing"], 1.0)

    def test_alarm_horizon_counts_alarm_after_control_end(self) -> None:
        result = alarm_horizon_audit(
            [
                {
                    "safe_alarm": True,
                    "first_alarm_minimum": 30,
                    "fault_step": 10,
                    "faulted_observation_steps": 40,
                    "control_observation_steps": 15,
                    "task_id": 0,
                }
            ]
        )

        self.assertEqual(result["eventual_alarms"], 1)
        self.assertEqual(result["alarms_after_successful_control_ended"], 1)

    def test_context_audit_reports_mixed_contexts_by_split(self) -> None:
        rows = [
            {
                "analysis_split": split,
                "context_id": f"{split}-mixed",
                "task_id": 0,
                "phase": "early",
                "safe_alarm": alarm,
            }
            for split in ("development", "holdout")
            for alarm in (False, True)
        ]

        result = context_outcome_audit(rows, permutations=10, seed=7)

        self.assertEqual(
            result["development"]["context_outcome_composition"]["mixed"], 1
        )
        self.assertEqual(
            result["holdout"]["context_outcome_composition"]["mixed"], 1
        )


if __name__ == "__main__":
    unittest.main()
