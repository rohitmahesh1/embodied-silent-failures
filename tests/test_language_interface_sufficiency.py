import importlib.util
import unittest

from embodied_silent_failures.language_interface_sufficiency import (
    feature_map,
    interface_rows,
    regression_cluster_bootstrap,
)


def measurement(layer: int, value: float, *, exact: bool = False) -> dict:
    return {
        "layer_index": layer,
        "changed_element_count": 0 if exact else 100,
        "difference_l2": 0.0 if exact else value,
        "normalized_difference_l2": 0.0 if exact else value / 10,
        "maximum_absolute_difference": 0.0 if exact else value / 20,
        "exact_equal": exact,
        "finite": True,
    }


class LanguageInterfaceSufficiencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = [
            {
                "eligible_causal_outcome": True,
                "record_id": "c000:layer01",
                "context_id": "c000",
                "task_id": 2,
                "episode_index": 3,
                "phase": "middle",
                "action_token_position": 4,
            }
        ]
        propagation = [measurement(0, 0.0, exact=True)] + [
            measurement(layer, float(layer + 1)) for layer in range(1, 32)
        ]
        local = {
            "layer_index": 1,
            "injection": {
                key: value
                for key, value in propagation[1].items()
                if key != "layer_index"
            },
            "propagation": propagation,
            "safe_feature": {
                key: value
                for key, value in measurement(31, 5.0).items()
                if key != "layer_index"
            },
            "executed_command": {"exact_equal": False},
        }
        self.scores = {
            "c000:layer01": {"local_measurements": local},
        }

    def test_rows_follow_only_downstream_interfaces(self) -> None:
        rows = interface_rows(self.analysis, self.scores)

        self.assertEqual(len(rows["block_transition"]), 30)
        self.assertEqual(rows["block_transition"][0]["boundary"], "block_1_to_2")
        self.assertEqual(rows["block_transition"][0]["history"], [])
        self.assertEqual(len(rows["block_transition"][1]["history"]), 1)
        self.assertEqual(len(rows["safe_feature_endpoint"]), 1)
        self.assertTrue(rows["command_change_endpoint"][0]["command_changed"])

    def test_feature_ladders_are_strictly_nested(self) -> None:
        row = interface_rows(self.analysis, self.scores)["block_transition"][2]

        local = feature_map(row, "local")
        history = feature_map(row, "history")
        context = feature_map(row, "context")

        self.assertLess(set(local), set(history))
        self.assertLess(set(history), set(context))
        self.assertIn("task=2", context)
        self.assertIn("action_token=4", context)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
        "numerical analysis dependencies are not installed",
    )
    def test_regression_bootstrap_preserves_trajectory_clusters(self) -> None:
        rows = [
            {
                "task_id": 0,
                "episode_index": episode,
                "target_log_normalized_l2": target,
            }
            for episode, target in ((0, 0.0), (0, 1.0), (1, 2.0), (1, 3.0))
        ]
        result = regression_cluster_bootstrap(
            rows,
            {
                "local": [1.0, 2.0, 3.0, 4.0],
                "history": [0.0, 1.0, 2.0, 3.0],
                "context": [0.0, 1.0, 2.0, 3.0],
            },
            samples=20,
            seed=1,
        )

        self.assertEqual(
            result["history"]["relative_mse_reduction_from_local"], 1.0
        )
        self.assertEqual(result["history"]["valid_samples"], 20)

        no_bootstrap = regression_cluster_bootstrap(
            rows,
            {
                "local": [1.0, 2.0, 3.0, 4.0],
                "history": [0.0, 1.0, 2.0, 3.0],
                "context": [0.0, 1.0, 2.0, 3.0],
            },
            samples=0,
            seed=1,
        )
        self.assertEqual(
            no_bootstrap["history"]["trajectory_cluster_bootstrap_95"], []
        )


if __name__ == "__main__":
    unittest.main()
