from __future__ import annotations

import unittest

import numpy as np

from embodied_silent_failures.action_monitor_geometry import (
    action_monitor_arrays,
    bfloat16_words_to_float32,
    conditional_js_divergence,
    summarize_action_monitor_arrays,
)


class ActionMonitorGeometryTests(unittest.TestCase):
    def test_js_is_zero_for_equal_logits_and_positive_for_changed_choice(self) -> None:
        clean = np.zeros((1, 256), dtype=np.float32)
        faulted = clean.copy()
        self.assertAlmostEqual(float(conditional_js_divergence(np, clean, clean)[0]), 0.0)

        clean[0, 3] = 10.0
        faulted[0, 9] = 10.0
        self.assertGreater(float(conditional_js_divergence(np, clean, faulted)[0]), 0.5)

    def test_summary_aligns_primary_measurement_to_safe_final_token(self) -> None:
        clean = np.zeros((2, 7, 256), dtype=np.float32)
        faulted = clean.copy()
        clean[:, :, 0] = 5.0
        faulted[0, 6, 1] = 8.0
        zeros = np.zeros((2, 7), dtype=np.float32)
        commands = np.zeros((2, 7), dtype=np.float32)
        arrays = action_monitor_arrays(
            np,
            clean_logits=clean,
            faulted_logits=faulted,
            clean_tokens=np.zeros((2, 7), dtype=np.int32),
            faulted_tokens=np.zeros((2, 7), dtype=np.int32),
            clean_raw_action=commands,
            faulted_raw_action=commands,
            clean_command=commands,
            faulted_command=commands,
            clean_full_log_normalizer=zeros,
            faulted_full_log_normalizer=zeros,
        )
        summary = summarize_action_monitor_arrays(arrays)

        self.assertGreater(summary["same_feature_action_js_at_fault"], 0.0)
        self.assertTrue(summary["same_feature_action_argmax_changed_at_fault"])
        self.assertEqual(summary["window_steps"], 2)

    def test_bfloat16_words_decode_without_numpy_bfloat16_support(self) -> None:
        float_words = np.asarray([1.0, -2.5], dtype=np.float32).view(np.uint32)
        bfloat16_words = (float_words >> 16).astype(np.uint16)

        decoded = bfloat16_words_to_float32(np, bfloat16_words)

        np.testing.assert_array_equal(decoded, np.asarray([1.0, -2.5]))


if __name__ == "__main__":
    unittest.main()
