import unittest

import numpy as np

from embodied_silent_failures.analyze_language_interface_atlas import spectrum_summary
from embodied_silent_failures.language_interface_atlas import (
    bfloat16_words_to_float32,
    context_arrays,
)


def _bfloat16_words(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(
        np.int16
    )


class _Archive(dict):
    pass


class LanguageInterfaceAtlasTests(unittest.TestCase):
    def test_bfloat16_words_decode_to_float32(self) -> None:
        values = np.asarray([0.0, 1.0, -2.5, 3.25], dtype=np.float32)
        decoded = bfloat16_words_to_float32(np, _bfloat16_words(values))

        np.testing.assert_array_equal(decoded, values)

    def test_context_arrays_recover_declared_signed_cuts(self) -> None:
        hidden = 2
        heads = 1
        head_dim = 2
        token = 2
        clean_residual = np.zeros((32, 7, hidden), dtype=np.float32)
        clean_keys = np.zeros((32, 7, heads, head_dim), dtype=np.float32)
        clean_values = np.zeros_like(clean_keys)
        source_residual = clean_residual.copy()
        rows = []
        key_rows = []
        value_rows = []
        owners = []
        layers = []
        tokens = []
        cache_owners = []
        cache_layers = []
        cache_tokens = []
        for owner in range(32):
            source_residual[owner, token] = owner + 1
            for current_token in range(token, 7):
                first_layer = owner if current_token == token else 0
                for layer in range(first_layer, 32):
                    rows.append(np.full(hidden, owner + layer + current_token + 1.0))
                    owners.append(owner)
                    layers.append(layer)
                    tokens.append(current_token)
                    if current_token == token and layer == owner:
                        rows[-1] = source_residual[owner, token].copy()
            for current_token in range(token, 7):
                first_layer = owner + 1 if current_token == token else 0
                for layer in range(first_layer, 32):
                    key_rows.append(
                        np.full((heads, head_dim), owner + layer + current_token + 2.0)
                    )
                    value_rows.append(
                        np.full((heads, head_dim), owner + layer + current_token + 3.0)
                    )
                    cache_owners.append(owner)
                    cache_layers.append(layer)
                    cache_tokens.append(current_token)

        archive = _Archive(
            fault_layer=np.arange(32, dtype=np.int16),
            clean_residuals=_bfloat16_words(clean_residual),
            source_residuals=_bfloat16_words(source_residual),
            clean_attention_cache_keys=_bfloat16_words(clean_keys),
            clean_attention_cache_values=_bfloat16_words(clean_values),
            fault_residuals=_bfloat16_words(np.asarray(rows, dtype=np.float32)),
            fault_residuals_row_intervention=np.asarray(owners, dtype=np.int16),
            fault_residuals_row_layer=np.asarray(layers, dtype=np.int16),
            fault_residuals_row_token=np.asarray(tokens, dtype=np.int8),
            fault_attention_cache_keys=_bfloat16_words(
                np.asarray(key_rows, dtype=np.float32)
            ),
            fault_attention_cache_keys_row_intervention=np.asarray(
                cache_owners, dtype=np.int16
            ),
            fault_attention_cache_keys_row_layer=np.asarray(
                cache_layers, dtype=np.int16
            ),
            fault_attention_cache_keys_row_token=np.asarray(
                cache_tokens, dtype=np.int8
            ),
            fault_attention_cache_values=_bfloat16_words(
                np.asarray(value_rows, dtype=np.float32)
            ),
            fault_attention_cache_values_row_intervention=np.asarray(
                cache_owners, dtype=np.int16
            ),
            fault_attention_cache_values_row_layer=np.asarray(
                cache_layers, dtype=np.int16
            ),
            fault_attention_cache_values_row_token=np.asarray(
                cache_tokens, dtype=np.int8
            ),
            clean_action_logits=np.zeros((1, 7, 256), dtype=np.float32),
            fault_action_logits=np.ones((32, 7, 256), dtype=np.float32),
        )
        local = {
            "context": {"context_id": "c000", "action_token_position": token},
            "interventions": [
                {
                    "status": "complete",
                    "layer_index": layer,
                    "clean_executed_command": [0.0] * 7,
                    "faulted_executed_command": [float(layer)] * 7,
                }
                for layer in range(32)
            ],
        }
        score_document = {
            "monitor": {"primary_alpha": 0.1},
            "records": [
                {
                    "record_id": f"c000-l{layer:02d}",
                    "status": "scored",
                    "composition_verified": True,
                    "control_success": True,
                    "terminal_success": layer % 2 == 0,
                    "monitor_horizon": "complete_physical_trace",
                    "context_id": "c000",
                    "context": {
                        "context_id": "c000",
                        "analysis_split": "development",
                        "task_id": 0,
                        "episode_index": 0,
                        "phase": "early",
                        "worker_shard": 0,
                        "policy_step": 1,
                        "action_token_position": token,
                    },
                    "layer_index": layer,
                    "local_measurements": {
                        "executed_command": {"exact_equal": layer == 0},
                        "propagation": [],
                    },
                    "alarms": {
                        "0.1": {
                            "within_10_steps": {"triggered": False},
                            "within_25_steps": {"triggered": False},
                            "post_fault_any": {"triggered": False},
                        }
                    },
                    "alarm_at_fault": False,
                    "alarm_before_fault": False,
                    "control_alarm_at_fault": False,
                    "score_at_fault": float(layer),
                    "control_score_at_fault": 0.0,
                    "score_change_from_control_at_fault": float(layer),
                }
                for layer in range(32)
            ],
        }

        safe_features = {
            "clean": np.zeros((7, hidden), dtype=np.float32),
            "fault": np.ones((32, 7, hidden), dtype=np.float32),
        }

        arrays = context_arrays(
            np, archive, local, score_document, safe_features=safe_features
        )

        self.assertEqual(arrays["injection_residual_delta"].shape, (32, hidden))
        self.assertEqual(arrays["immediate_residual_delta"].shape, (31, hidden))
        self.assertEqual(
            arrays["immediate_key_delta"].shape, (31, heads, head_dim)
        )
        self.assertEqual(arrays["final_residual_delta"].shape, (32, hidden))
        self.assertEqual(arrays["safe_feature_delta"].shape, (32, hidden))
        self.assertEqual(arrays["residual_path_delta"].shape, (528, hidden))
        self.assertEqual(
            arrays["cache_key_path_delta"].shape, (496, heads, head_dim)
        )
        self.assertEqual(
            arrays["cache_value_path_delta"].shape, (496, heads, head_dim)
        )
        self.assertEqual(
            list(
                zip(
                    arrays["residual_path_source_layer"][:3],
                    arrays["residual_path_boundary_layer"][:3],
                    strict=True,
                )
            ),
            [(0, 0), (0, 1), (0, 2)],
        )
        np.testing.assert_array_equal(arrays["safe_feature_delta"], 1.0)
        np.testing.assert_array_equal(arrays["action_logit_delta"], 1.0)
        np.testing.assert_array_equal(arrays["command_delta"][3], 3.0)
        self.assertEqual(arrays["task_failure"].tolist()[:3], [0, 1, 0])
        self.assertEqual(
            arrays["operational_silent_failure"].tolist()[:3], [0, 1, 0]
        )

    def test_spectrum_summary_separates_mean_and_variation(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        summary = spectrum_summary(np, matrix)

        self.assertAlmostEqual(summary["uncentered"]["effective_rank"], 1.0)
        self.assertEqual(summary["uncentered"]["components_99"], 1)
        self.assertEqual(summary["centered"]["effective_rank"], 0.0)
        self.assertEqual(summary["centered"]["components_99"], None)


if __name__ == "__main__":
    unittest.main()
