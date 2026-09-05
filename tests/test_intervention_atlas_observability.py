import unittest

import numpy as np

from embodied_silent_failures.intervention_atlas_observability import (
    fresh_evidence_shift,
    observability_rows,
    window_summary,
)


def _record(
    record_id: str,
    physical_run: str,
    *,
    failure: bool,
    site_id: str,
) -> dict:
    return {
        "record_id": record_id,
        "physical_run": physical_run,
        "context_id": "c0001",
        "site_id": site_id,
        "representative_site_id": "site-a",
        "primary_eligible": True,
        "policy_failure": failure,
        "context": {
            "task_id": 0,
            "episode_index": 2,
            "phase": "early",
            "policy_step": 1,
        },
    }


class InterventionAtlasObservabilityTests(unittest.TestCase):
    def test_fresh_evidence_removes_prior_cumulative_score(self) -> None:
        shift = fresh_evidence_shift(
            np.asarray([10.0, 12.0, 15.0]),
            np.asarray([10.0, 11.0, 13.0]),
            fault_step=1,
            window_steps=2,
        )

        self.assertEqual(shift, 2.0)

    def test_rows_keep_sites_but_exclude_control_as_physical_fault(self) -> None:
        records = [
            _record("a", "c0001-command-a", failure=True, site_id="site-a"),
            _record("b", "c0001-command-a", failure=True, site_id="site-b"),
            _record("c", "c0001-control", failure=False, site_id="site-c"),
        ]
        control = np.arange(120, dtype=float)
        physical = {
            "c0001-control": {
                "scores": control,
                "record": {"success": True},
            },
            "c0001-command-a": {
                "scores": control + 1,
                "record": {"success": False},
            },
        }
        site_scores = {
            "a": control + 1,
            "b": control + 2,
            "c": control + 3,
        }

        site_rows, physical_rows = observability_rows(
            [("development", records)], site_scores, physical
        )

        self.assertEqual(len(site_rows), 3)
        self.assertEqual(len(physical_rows), 1)
        self.assertEqual(physical_rows[0]["member_site_count"], 2)

    def test_window_summary_reports_within_context_concordance(self) -> None:
        rows = []
        for index, (failure, value) in enumerate(
            [(True, 2.0), (False, 1.0), (True, 4.0), (False, 3.0)]
        ):
            rows.append(
                {
                    "policy_failure": failure,
                    "task_id": 0,
                    "context_id": f"context-{index // 2}",
                    "fresh_evidence_shift": {"25": value},
                }
            )

        result = window_summary(rows, 25)

        self.assertEqual(result["failure_classification"]["pooled_roc_auc"], 0.75)
        self.assertEqual(
            result["failure_classification"]["within_context"]["roc_auc"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
