import unittest

from embodied_silent_failures.intervention_atlas_followups import (
    physical_equivalence_audit,
    site_rates,
    stability_features,
)


class InterventionAtlasFollowupTests(unittest.TestCase):
    def test_stability_interactions_are_explicit(self) -> None:
        row = {"site_id": "site-a", "task_id": 2, "phase": "late"}

        features = stability_features(row, "site_context_interactions")

        self.assertEqual(features["site=site-a"], 1.0)
        self.assertEqual(features["task=2"], 1.0)
        self.assertEqual(features["phase=late"], 1.0)
        self.assertEqual(features["site_task=site-a|2"], 1.0)
        self.assertEqual(features["site_phase=site-a|late"], 1.0)

    def test_task_and_phase_can_be_inspected_separately(self) -> None:
        row = {"site_id": "site-a", "task_id": 2, "phase": "late"}

        self.assertEqual(stability_features(row, "task_only"), {"task=2": 1.0})
        self.assertEqual(
            stability_features(row, "phase_only"), {"phase=late": 1.0}
        )

    def test_site_rates_can_condition_on_policy_failure(self) -> None:
        rows = [
            {"site_id": "a", "policy_failure": True, "miss": True},
            {"site_id": "a", "policy_failure": False, "miss": False},
            {"site_id": "a", "policy_failure": True, "miss": False},
        ]

        rates = site_rates(
            rows,
            "miss",
            select=lambda row: row["policy_failure"],
        )

        self.assertEqual(rates, {"a": 0.5})

    def test_physical_equivalence_requires_matching_every_context(self) -> None:
        rows = [
            {
                "site_id": site,
                "task_id": 0,
                "episode_index": 0,
                "phase": phase,
                "physical_run": run,
            }
            for site, phase, run in (
                ("a", "early", "early-1"),
                ("a", "late", "late-1"),
                ("b", "early", "early-1"),
                ("b", "late", "late-1"),
                ("c", "early", "early-2"),
                ("c", "late", "late-1"),
            )
        ]

        result = physical_equivalence_audit(rows)

        self.assertEqual(result["graph_sites"], 3)
        self.assertEqual(result["behavioral_equivalence_classes"], 2)
        self.assertEqual(result["class_sizes_descending"], [2, 1])


if __name__ == "__main__":
    unittest.main()
