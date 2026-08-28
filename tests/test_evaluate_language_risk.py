import unittest

import numpy as np
from sklearn import metrics as sklearn_metrics
from sklearn.calibration import calibration_curve

from embodied_silent_failures.evaluate_language_risk import _bootstrap, _metrics


class EvaluateLanguageRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.asarray([0, 1, 0, 1], dtype=int)
        self.probabilities = np.asarray([0.1, 0.8, 0.2, 0.7], dtype=float)

    def test_metrics_include_ranking_calibration_and_tie_policy(self) -> None:
        result = _metrics(
            self.labels,
            self.probabilities,
            np,
            sklearn_metrics,
            include_calibration=True,
            calibration_curve=calibration_curve,
        )

        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["average_precision"], 1.0)
        self.assertTrue(result["calibration_by_predicted_risk_fifth"])
        self.assertEqual(result["top_risk_group"]["residual_interventions"], 1)

    def test_bootstrap_resamples_whole_trajectories(self) -> None:
        rows = [
            {
                "task_id": index // 2,
                "episode_index": 0,
                "operational_silent_failure": bool(label),
            }
            for index, label in enumerate(self.labels)
        ]

        result = _bootstrap(
            rows,
            self.probabilities,
            samples=20,
            seed=4,
            np=np,
            sklearn_metrics=sklearn_metrics,
        )

        self.assertEqual(result["requested_samples"], 20)
        self.assertEqual(result["brier_score_valid_samples"], 20)
        self.assertEqual(result["roc_auc_valid_samples"], 20)


if __name__ == "__main__":
    unittest.main()
