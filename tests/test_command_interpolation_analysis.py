from __future__ import annotations

import unittest

from embodied_silent_failures.command_interpolation_analysis import (
    branch_boundary_summary,
)


class CommandInterpolationAnalysisTests(unittest.TestCase):
    def test_boundary_summary_includes_known_endpoints(self) -> None:
        records = [
            {
                "physical_run": "run-a",
                "context_id": "c001",
                "worker_shard": 0,
                "analysis_split": "development",
                "interpolation": interpolation,
                "success": success,
            }
            for interpolation, success in (
                (0.0, True),
                (0.25, True),
                (0.5, True),
                (0.75, False),
                (1.0, False),
            )
        ]
        summary = branch_boundary_summary(records)
        self.assertEqual(summary["outcome_patterns"], {"SSSFF": 1})
        self.assertEqual(summary["monotone_branches"], 1)
        self.assertEqual(summary["endpoint_contract_branches"], 1)
        self.assertEqual(
            summary["records"][0]["first_observed_failure_lambda"], 0.75
        )

    def test_boundary_summary_marks_nonmonotone_sequence(self) -> None:
        records = [
            {
                "physical_run": "run-a",
                "context_id": "c001",
                "worker_shard": 0,
                "analysis_split": "development",
                "interpolation": interpolation,
                "success": success,
            }
            for interpolation, success in (
                (0.0, True),
                (0.25, False),
                (0.5, True),
                (1.0, False),
            )
        ]
        summary = branch_boundary_summary(records)
        self.assertEqual(summary["monotone_branches"], 0)


if __name__ == "__main__":
    unittest.main()
