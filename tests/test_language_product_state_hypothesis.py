import importlib.util
import subprocess
import sys
import unittest


class LanguageProductStateHypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("numpy") is None:
            raise unittest.SkipTest("NumPy is required")
        import numpy as np

        cls.np = np

    def _row(self, *, layer: int = 2, failed: bool = False) -> dict:
        np = self.np
        state = {
            "robot0_proprio-state": {
                "before": np.asarray([1.0, 2.0]),
                "control_after": np.asarray([1.1, 2.1]),
                "after": np.asarray([1.2, 2.2]),
            },
            "object-state": {
                "before": np.asarray([3.0]),
                "control_after": np.asarray([3.1]),
                "after": np.asarray([3.2]),
            },
            "simulator_state": {
                "before": np.asarray([4.0, 5.0, 6.0]),
                "control_after": np.asarray([4.1, 5.1, 6.1]),
                "after": np.asarray([4.2, 5.2, 6.2]),
            },
        }
        return {
            "record_id": f"c000:layer{layer:02d}",
            "physical_run": "c000-command-a",
            "analysis_split": "development",
            "context_id": "c000",
            "task_id": 1,
            "episode_index": 3,
            "phase": "middle",
            "action_token_position": 4,
            "layer_index": layer,
            "eligible_causal_outcome": True,
            "command_changed": True,
            "task_failure": failed,
            "operational_silent_failure": failed,
            "control_command": np.zeros(7),
            "fault_command": np.ones(7),
            "safe_threshold_at_fault": 10.0,
            "score_at_fault": "4.0",
            "control_score_at_fault": "3.0",
            "score_change_from_control_at_fault": "1.0",
            "safe_feature_normalized_l2": "0.5",
            "injection_l2": "2.0",
            "injection_normalized_l2": "0.2",
            "final_propagation_l2": "3.0",
            "final_propagation_normalized_l2": "0.3",
            "state": state,
        }

    def test_feature_ladder_only_adds_declared_provenance(self) -> None:
        from embodied_silent_failures.language_product_state_hypothesis import (
            feature_names,
            feature_vector,
            state_widths,
        )

        row = self._row()
        widths = state_widths([row])
        product = feature_vector(
            self.np, row, "observation_product", "product", widths
        )
        origin = feature_vector(
            self.np, row, "observation_product", "product_and_origin", widths
        )
        path = feature_vector(
            self.np,
            row,
            "observation_product",
            "product_origin_and_path",
            widths,
        )

        self.assertEqual(len(origin) - len(product), 39)
        self.assertEqual(len(path) - len(origin), 4)
        self.assertEqual(
            len(path),
            len(
                feature_names(
                    "observation_product", "product_origin_and_path", widths
                )
            ),
        )
        self.assertAlmostEqual(product[13], 0.0)
        self.assertAlmostEqual(product[20], 1.0)

    def test_alias_audit_does_not_treat_layers_as_physical_repeats(self) -> None:
        from embodied_silent_failures.language_product_state_models import (
            alias_audit,
            alias_weights,
        )

        first = self._row(layer=2, failed=True)
        second = self._row(layer=3, failed=True)
        third = self._row(layer=4, failed=False)
        third["physical_run"] = "c000-command-b"
        audit = alias_audit([first, second, third], "task_failure")
        weights = alias_weights(self.np, [first, second, third])

        self.assertEqual(audit["physical_branches"], 2)
        self.assertEqual(audit["branches_with_multiple_source_layers"], 1)
        self.assertEqual(audit["alias_groups_with_mixed_outcome"], 0)
        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])

    def test_monitor_miss_population_conditions_on_task_failure(self) -> None:
        from embodied_silent_failures.language_product_state_hypothesis import (
            eligible_rows,
        )

        failed = self._row(failed=True)
        succeeded = self._row(layer=3, failed=False)
        rows = eligible_rows(
            [failed, succeeded], "monitor_miss_given_failure"
        )

        self.assertEqual(rows, [failed])
        self.assertTrue(rows[0]["monitor_miss_given_failure"])

    def test_state_archive_uses_shared_offsets_for_both_moments(self) -> None:
        from embodied_silent_failures.language_product_state_hypothesis import (
            _unpack_state,
        )

        archive = {
            "numeric_state_offsets": self.np.asarray([0, 2, 3]),
            "numeric_state_shapes": self.np.asarray([[2], [1]]),
            "numeric_state_before_values": self.np.asarray([1.0, 2.0, 3.0]),
            "numeric_state_after_values": self.np.asarray([4.0, 5.0, 6.0]),
        }

        self.assertEqual(
            _unpack_state(self.np, archive, "before", 0).tolist(), [1.0, 2.0]
        )
        self.assertEqual(
            _unpack_state(self.np, archive, "after", 1).tolist(), [6.0]
        )

    def test_analysis_cli_loads_before_reading_artifacts(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_silent_failures.analyze_language_product_state_hypothesis",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--shard-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
