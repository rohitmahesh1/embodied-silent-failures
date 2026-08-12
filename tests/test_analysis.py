import unittest

from embodied_silent_failures.analysis import (
    BASELINE_FAILURE,
    DETECTED_FAULT_FAILURE,
    FALSE_ALARM,
    PRESERVED_SUCCESS,
    SILENT_FAULT_FAILURE,
    Alarm,
    TREATMENT_CONDITIONS,
    alarm_from_score_band,
    alarm_from_scores,
    classify_pair,
    summarize_by_fault_field,
    summarize_outcomes,
)
from embodied_silent_failures.score_safe import alarm_windows


class AnalysisTests(unittest.TestCase):
    def result(
        self,
        condition: str,
        success: bool,
        episode_index: int = 4,
        bit_class: str = "mantissa",
    ) -> dict:
        value = {
            "condition": condition,
            "task_id": 2,
            "episode_index": episode_index,
            "initial_state_sha256": f"state-{episode_index}",
            "trial_seed": 17 + episode_index,
            "success": success,
        }
        if condition in TREATMENT_CONDITIONS:
            value["fault"] = {
                "trial_seed": value["trial_seed"],
                "bit_class": bit_class,
            }
        return value

    def pair(self, clean_success: bool, fault_success: bool, alarm: bool):
        return classify_pair(
            self.result("clean", clean_success),
            self.result("activation_fault", fault_success),
            Alarm(alarm, 12 if alarm else None),
        )

    def test_alarm_reports_first_threshold_crossing(self) -> None:
        self.assertEqual(alarm_from_scores([0.1, 0.5, 0.7], 0.5), Alarm(True, 1))
        self.assertEqual(alarm_from_scores([0.1, 0.2], 0.5), Alarm(False, None))

    def test_time_varying_alarm_respects_the_post_fault_window(self) -> None:
        scores = [0.9, 0.1, 0.4, 0.8]
        thresholds = [0.5, 0.5, 0.3, 0.7]

        self.assertEqual(
            alarm_from_score_band(scores, thresholds, start_step=1, stop_step=3),
            Alarm(True, 2),
        )
        self.assertEqual(
            alarm_from_score_band(scores, thresholds, start_step=1, stop_step=2),
            Alarm(False, None),
        )

    def test_time_varying_alarm_rejects_invalid_windows_and_bands(self) -> None:
        with self.assertRaises(ValueError):
            alarm_from_score_band([0.1], [0.2], start_step=1)
        with self.assertRaises(ValueError):
            alarm_from_score_band([0.1, 0.2], [0.2], stop_step=2)

    def test_safe_alarm_windows_start_at_the_fault(self) -> None:
        alarms = alarm_windows(
            scores=[0.9, 0.1, 0.2, 0.1, 0.1, 0.1, 0.8],
            alphas=[0.1],
            bands=[[0.5] * 7],
            fault_step=1,
        )["0.1"]

        self.assertFalse(alarms["within_5_steps"]["triggered"])
        self.assertTrue(alarms["post_fault_any"]["triggered"])
        self.assertEqual(alarms["post_fault_any"]["first_step"], 6)

    def test_safe_alarm_windows_report_nonfinite_policy_separately(self) -> None:
        values = [0.1, float("nan"), 0.1]
        arguments = {
            "scores": values,
            "alphas": [0.1],
            "bands": [[0.5] * len(values)],
            "fault_step": 1,
        }

        native = alarm_windows(**arguments)["0.1"]["post_fault_any"]
        guarded = alarm_windows(
            **arguments, nonfinite_is_alarm=True
        )["0.1"]["post_fault_any"]

        self.assertFalse(native["triggered"])
        self.assertTrue(guarded["triggered"])
        self.assertEqual(guarded["first_step"], 1)

    def test_pair_categories_preserve_the_causal_denominator(self) -> None:
        cases = [
            (True, True, False, PRESERVED_SUCCESS),
            (True, True, True, FALSE_ALARM),
            (True, False, True, DETECTED_FAULT_FAILURE),
            (True, False, False, SILENT_FAULT_FAILURE),
            (False, False, False, BASELINE_FAILURE),
        ]
        for clean, fault, alarm, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.pair(clean, fault, alarm).category, expected)

    def test_pair_rejects_different_initial_states(self) -> None:
        clean = self.result("clean", True, episode_index=1)
        fault = self.result("activation_fault", False, episode_index=2)
        with self.assertRaises(ValueError):
            classify_pair(clean, fault, Alarm(False, None))

    def test_summary_separates_policy_vulnerability_and_residual_risk(self) -> None:
        outcomes = [
            self.pair(True, True, False),
            self.pair(True, False, True),
            self.pair(True, False, False),
            self.pair(False, False, False),
        ]
        summary = summarize_outcomes(outcomes)

        self.assertEqual(summary["eligible_clean_successes"], 3)
        self.assertEqual(summary["excluded_baseline_failures"], 1)
        self.assertEqual(summary["policy_vulnerability"]["estimate"], 2 / 3)
        self.assertEqual(
            summary["monitor_miss_given_fault_failure"]["estimate"], 1 / 2
        )
        self.assertEqual(summary["residual_silent_risk"]["estimate"], 1 / 3)

    def test_summary_can_be_stratified_by_fault_record(self) -> None:
        mantissa = self.pair(True, False, False)
        exponent = classify_pair(
            self.result("clean", True),
            self.result("activation_fault", False, bit_class="exponent"),
            Alarm(True, 4),
        )
        grouped = summarize_by_fault_field([mantissa, exponent], "bit_class")

        self.assertEqual(set(grouped), {"exponent", "mantissa"})
        self.assertEqual(grouped["mantissa"]["residual_silent_risk"]["estimate"], 1)
        self.assertEqual(grouped["exponent"]["residual_silent_risk"]["estimate"], 0)

    def test_pair_accepts_stale_image_as_a_supported_treatment(self) -> None:
        outcome = classify_pair(
            self.result("clean", True),
            self.result("stale_image", False),
            Alarm(False, None),
        )

        self.assertEqual(outcome.category, SILENT_FAULT_FAILURE)


if __name__ == "__main__":
    unittest.main()
