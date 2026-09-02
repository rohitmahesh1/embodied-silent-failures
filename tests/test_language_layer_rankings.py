from __future__ import annotations

import unittest

from embodied_silent_failures.language_layer_rankings import (
    LANGUAGE_BLOCK_COUNT,
    expected_protection,
    freeze_rankings,
    spearman,
    tie_aware_protection_weights,
)


def _analysis(split: str = "development") -> dict:
    records = []
    for context in range(4):
        for layer in range(LANGUAGE_BLOCK_COUNT):
            failure = layer >= 16 and context < 2
            residual = layer >= 24 and context == 0
            records.append(
                {
                    "record_id": f"c{context}:l{layer}",
                    "eligible_causal_outcome": True,
                    "task_id": context % 2,
                    "episode_index": context,
                    "layer_index": layer,
                    "task_failure": failure,
                    "operational_silent_failure": residual,
                }
            )
    return {"analysis_split": split, "records": records}


class LanguageLayerRankingsTest(unittest.TestCase):
    def test_freezes_distinct_development_rankings(self) -> None:
        frozen = freeze_rankings(_analysis())
        vulnerability = frozen["rankings"]["conventional_vulnerability"]
        residual = frozen["rankings"]["monitor_aware_residual"]

        self.assertEqual(vulnerability["layers"]["20"]["rate"], 0.5)
        self.assertEqual(residual["layers"]["20"]["rate"], 0.0)
        self.assertLess(
            frozen["development_ranking_comparison"][
                "spearman_with_average_ties"
            ],
            1.0,
        )

    def test_rejects_holdout_as_ranking_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "development"):
            freeze_rankings(_analysis("holdout"))

    def test_tie_policy_spends_exact_budget_without_layer_order(self) -> None:
        scores = {layer: 1.0 if layer >= 16 else 0.0 for layer in range(32)}
        weights = tie_aware_protection_weights(scores, 8)

        self.assertAlmostEqual(sum(weights.values()), 8.0)
        self.assertTrue(all(weights[layer] == 0.5 for layer in range(16, 32)))
        self.assertTrue(all(weights[layer] == 0.0 for layer in range(16)))

    def test_expected_capture_uses_frozen_tie_weights(self) -> None:
        rows = [
            {
                "layer_index": layer,
                "operational_silent_failure": layer >= 24,
            }
            for layer in range(32)
        ]
        weights = {layer: 1.0 if layer >= 24 else 0.0 for layer in range(32)}

        result = expected_protection(
            rows, "operational_silent_failure", weights
        )

        self.assertEqual(result["events"], 8)
        self.assertEqual(result["expected_events_captured"], 8.0)
        self.assertEqual(result["expected_capture_fraction"], 1.0)

    def test_spearman_uses_average_ties(self) -> None:
        left = {layer: float(layer // 4) for layer in range(32)}
        right = dict(left)
        reverse = {layer: -value for layer, value in left.items()}

        self.assertAlmostEqual(spearman(left, right), 1.0)
        self.assertAlmostEqual(spearman(left, reverse), -1.0)


if __name__ == "__main__":
    unittest.main()
