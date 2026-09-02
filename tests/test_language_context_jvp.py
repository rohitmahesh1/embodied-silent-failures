import unittest

from embodied_silent_failures.language_context_jvp import (
    approximation_metrics,
    clean_full_prompt_states,
    output_names,
    ragged_call,
    scaled_source,
    sparse_rows,
    sparse_value,
)


class LanguageContextJvpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("NumPy is required") from error
        cls.np = np

    def test_ragged_call_recovers_the_requested_shape(self) -> None:
        np = self.np
        arrays = {
            "clean_example_values": np.asarray([1, 2, 3, 4, 5]),
            "clean_example_offsets": np.asarray([0, 4, 5]),
            "clean_example_shapes": np.asarray([[2, 2], [1, 1]]),
            "clean_example_call_indices": np.asarray([0, 3]),
        }
        result = ragged_call(np, arrays, "example", 3)
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.tolist(), [[5]])

    def test_sparse_rows_preserve_the_full_coordinate(self) -> None:
        np = self.np
        arrays = {
            "fault_values": np.asarray([[10], [20]]),
            "fault_values_row_intervention": np.asarray([1, 0]),
            "fault_values_row_layer": np.asarray([4, 4]),
            "fault_values_row_token": np.asarray([2, 2]),
        }
        index = sparse_rows(np, arrays, "values")
        self.assertEqual(
            sparse_value(arrays, index, "values", 0, 4, 2).tolist(), [20]
        )

    def test_output_names_follow_recorded_attention_and_mlp_cuts(self) -> None:
        names = output_names(30)
        self.assertEqual(
            names,
            [
                {"family": "post_attention_residual", "layer_index": 31},
                {"family": "post_block_residual", "layer_index": 31},
                {"family": "current_token_key", "layer_index": 31},
                {"family": "current_token_value", "layer_index": 31},
                {"family": "selected_token_final_feature", "layer_index": 31},
                {"family": "selected_token_action_logits", "layer_index": None},
            ],
        )

    def test_approximation_metrics_distinguish_direction_and_scale(self) -> None:
        np = self.np
        metrics = approximation_metrics(
            np, np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])
        )
        self.assertAlmostEqual(metrics["normalized_error"], 1.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertAlmostEqual(metrics["norm_ratio"], 2.0)

    def test_clean_full_prompt_states_preserve_full_sequence(self) -> None:
        try:
            torch = __import__("torch")
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is required") from error

        class Layer(torch.nn.Module):
            def forward(self, hidden_states, **_kwargs):
                return (hidden_states + 1,)

        context = {
            "initial_language_input": torch.zeros((1, 3, 4)),
            "full_attention_mask": None,
            "full_position_ids": torch.arange(3).reshape(1, 3),
            "full_cache_position": torch.arange(3),
        }
        states = clean_full_prompt_states(torch, [Layer(), Layer()], context)
        self.assertEqual([tuple(value.shape) for value in states], [(1, 3, 4)] * 2)
        self.assertTrue(torch.equal(states[-1], torch.full((1, 3, 4), 2.0)))

    def test_scaled_source_records_the_realized_low_precision_step(self) -> None:
        try:
            torch = __import__("torch")
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is required") from error

        clean = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        fault = torch.tensor([2.0, 4.0], dtype=torch.bfloat16)
        halfway = scaled_source(torch, clean, fault, 0.5)
        self.assertEqual(halfway.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(halfway, torch.tensor([1.5, 3.0], dtype=torch.bfloat16))
        )
        self.assertTrue(torch.equal(scaled_source(torch, clean, fault, 1.0), fault))

    def test_scaled_source_rejects_extrapolation(self) -> None:
        try:
            torch = __import__("torch")
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is required") from error

        value = torch.ones(2, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            scaled_source(torch, value, value, 1.5)


if __name__ == "__main__":
    unittest.main()
