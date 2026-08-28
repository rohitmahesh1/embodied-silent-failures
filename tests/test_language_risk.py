import math
import unittest

from embodied_silent_failures.language_risk import (
    feature_row,
    predict_probability,
    top_risk_group,
    transform,
)


class LanguageRiskTests(unittest.TestCase):
    def test_feature_transforms_are_explicit(self) -> None:
        self.assertEqual(transform(2.0, "identity"), 2.0)
        self.assertAlmostEqual(transform(2.0, "log1p"), math.log(3.0))
        self.assertAlmostEqual(transform(-2.0, "signed_log1p"), -math.log(3.0))
        with self.assertRaises(ValueError):
            transform(-1.0, "log1p")

    def test_json_logistic_model_scores_without_sklearn(self) -> None:
        model = {
            "features": [{"name": "value", "transform": "identity"}],
            "standardization": {"mean": [2.0], "scale": [2.0]},
            "logistic_regression": {"coefficients": [2.0], "intercept": -1.0},
        }

        probability = predict_probability(model, {"value": 4.0})

        self.assertAlmostEqual(probability, 1 / (1 + math.exp(-1.0)))

    def test_top_group_includes_boundary_ties(self) -> None:
        indices, threshold = top_risk_group([0.9, 0.8, 0.8, 0.1], fraction=0.5)

        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(threshold, 0.8)

    def test_feature_row_rejects_missing_eligible_measurement(self) -> None:
        with self.assertRaises(ValueError):
            feature_row({}, (("missing", "identity"),))


if __name__ == "__main__":
    unittest.main()
