from __future__ import annotations

import unittest

from embodied_silent_failures.safe_trajectory_geometry import (
    physical_population,
    summarize_trajectory_arrays,
)


def _site_record(
    *,
    run: str,
    site: str,
    representative: str,
    failure: bool,
    alarm: bool,
) -> dict:
    return {
        "primary_eligible": True,
        "physical_run": run,
        "context_id": "c0001",
        "site_id": site,
        "representative_site_id": representative,
        "policy_failure": failure,
        "context": {
            "task_id": 2,
            "episode_index": 3,
            "phase": "middle",
            "policy_step": 10,
        },
        "safe_faulted_evidence": {
            "alarms": {
                "0.05": {
                    "post_fault_any": {
                        "triggered": alarm,
                        "first_step": 40 if alarm else None,
                    },
                    "within_25_steps": {"triggered": False},
                }
            }
        },
    }


class SafeTrajectoryGeometryTests(unittest.TestCase):
    def test_population_counts_each_noncontrol_branch_once(self) -> None:
        monitor = {"primary_alpha": 0.05}
        document = {
            "monitor": monitor,
            "analysis_split": "holdout",
            "records": [
                _site_record(
                    run="c0001-control",
                    site="control-site",
                    representative="control-site",
                    failure=False,
                    alarm=False,
                ),
                _site_record(
                    run="c0001-command-a",
                    site="site-a",
                    representative="site-a",
                    failure=True,
                    alarm=False,
                ),
                _site_record(
                    run="c0001-command-a",
                    site="site-b",
                    representative="site-a",
                    failure=True,
                    alarm=False,
                ),
            ],
        }
        population, selected_monitor = physical_population([document])
        self.assertEqual(selected_monitor, monitor)
        self.assertEqual(len(population), 1)
        self.assertEqual(population[0]["member_site_count"], 2)
        self.assertEqual(population[0]["outcome_group"], "silent_failure")
        self.assertFalse(population[0]["safe_alarm_within_25_steps"])

    def test_window_geometry_separates_alignment_and_cancellation(self) -> None:
        import numpy as np

        arrays = {
            "selected_feature_l2": np.asarray([4.0, 4.0]),
            "selected_feature_normalized_l2": np.asarray([0.4, 0.4]),
            "monitor_increment_delta": np.asarray([0.3, -0.2]),
            "absolute_monitor_increment_delta": np.asarray([0.3, 0.2]),
            "clean_gradient_l2": np.asarray([1.0, 1.0]),
            "clean_gradient_dot_delta": np.asarray([0.2, -0.1]),
            "clean_gradient_cosine": np.asarray([0.05, -0.025]),
            "clean_linearization_error": np.asarray([0.1, -0.1]),
            "relu_gate_flip_fraction": np.asarray([0.0, 0.5]),
        }
        summary = summarize_trajectory_arrays(arrays)

        self.assertEqual(summary["window_steps"], 2)
        self.assertGreater(summary["safe_response_cancellation_fraction"], 0)
        self.assertLess(summary["gradient_alignment_fraction"], 1)
        self.assertAlmostEqual(
            summary["safe_response_signed_sum"],
            float(arrays["monitor_increment_delta"].sum()),
        )


if __name__ == "__main__":
    unittest.main()
