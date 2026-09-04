import unittest

from embodied_silent_failures.intervention_atlas_alarm_specificity import (
    _paired_summary,
)


class InterventionAtlasAlarmSpecificityTests(unittest.TestCase):
    def test_equal_horizon_excludes_late_alarm(self) -> None:
        result = _paired_summary(
            [
                {
                    "faulted_first": 12,
                    "control_first": None,
                    "faulted_length": 20,
                    "control_length": 10,
                    "fault_step": 5,
                }
            ],
            alpha="0.1",
        )

        self.assertEqual(result["own_terminal_horizon"]["faulted_only"], 1)
        self.assertEqual(
            result["equal_observation_horizon"]["faulted_only"], 0
        )
        self.assertEqual(
            result["horizon_audit"][
                "faulted_alarms_only_after_common_horizon"
            ],
            1,
        )

    def test_paired_test_counts_only_discordant_pairs(self) -> None:
        result = _paired_summary(
            [
                {
                    "faulted_first": 8,
                    "control_first": None,
                    "faulted_length": 10,
                    "control_length": 10,
                    "fault_step": 5,
                },
                {
                    "faulted_first": 8,
                    "control_first": 8,
                    "faulted_length": 10,
                    "control_length": 10,
                    "fault_step": 5,
                },
            ],
            alpha="0.1",
        )

        comparison = result["equal_observation_horizon"]
        self.assertEqual(comparison["faulted_only"], 1)
        self.assertEqual(comparison["both"], 1)
        self.assertEqual(comparison["discordant_pairs"], 1)
        self.assertEqual(comparison["exact_paired_binomial_p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
