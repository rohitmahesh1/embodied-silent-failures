import unittest

import numpy as np

from embodied_silent_failures.language_interface_prediction import (
    balanced_signed_sketch,
    compose_path,
    fit_transition,
    predict_transition,
    trajectory_folds,
    vector_metrics,
)
from embodied_silent_failures.language_interface_prediction_data import (
    cumulative_cache_state,
    endpoint_features,
    local_risk_features,
    task_block_physical_state,
)


class LanguageInterfacePredictionTests(unittest.TestCase):
    def test_balanced_signed_sketch_is_deterministic_and_linear(self) -> None:
        left = np.arange(16, dtype=np.float32).reshape(2, 8)
        right = np.flip(left, axis=1).copy()

        combined = balanced_signed_sketch(
            np, left + right, width=4, seed=17
        )
        separate = balanced_signed_sketch(
            np, left, width=4, seed=17
        ) + balanced_signed_sketch(np, right, width=4, seed=17)

        np.testing.assert_array_equal(combined, separate)
        np.testing.assert_array_equal(
            separate,
            balanced_signed_sketch(np, left, width=4, seed=17)
            + balanced_signed_sketch(np, right, width=4, seed=17),
        )

    def test_trajectory_folds_hold_out_whole_balanced_trajectories(self) -> None:
        contexts = [
            {"task_id": task, "episode_index": episode, "phase": phase}
            for task in range(2)
            for episode in (3, 7, 11)
            for phase in ("early", "late")
        ]

        folds = trajectory_folds(contexts)

        self.assertEqual(len(folds), 3)
        for fold in folds:
            trajectories = {
                (contexts[index]["task_id"], contexts[index]["episode_index"])
                for index in fold
            }
            self.assertEqual(len(trajectories), 2)
            for trajectory in trajectories:
                self.assertEqual(
                    sum(
                        (context["task_id"], context["episode_index"])
                        == trajectory
                        for context in contexts
                    ),
                    2,
                )

    def test_linear_transition_composes_known_maps(self) -> None:
        rng = np.random.default_rng(4)
        state = rng.normal(size=(80, 3)).astype(np.float32)
        clean = rng.normal(size=(80, 3)).astype(np.float32)
        tokens = np.arange(80) % 7
        target = state * 1.25
        model = fit_transition(
            np,
            state,
            clean,
            tokens,
            target,
            alpha=1e-8,
            kind="linear",
        )

        predicted = predict_transition(np, model, state, clean, tokens)
        np.testing.assert_allclose(predicted, target, rtol=1e-5, atol=1e-5)
        composed = compose_path(
            np,
            [model] * 31,
            np.ones(3, dtype=np.float32),
            np.ones((32, 3), dtype=np.float32),
            2,
            source_layer=29,
        )
        np.testing.assert_allclose(composed, 1.25**2, rtol=1e-5)

    def test_identity_transitions_fit_a_small_correction(self) -> None:
        rng = np.random.default_rng(12)
        state = rng.normal(size=(80, 3)).astype(np.float32)
        clean = rng.normal(size=(80, 3)).astype(np.float32)
        tokens = np.arange(80) % 7
        target = state * 1.25

        for kind in ("identity_linear", "identity_state_conditioned"):
            model = fit_transition(
                np,
                state,
                clean,
                tokens,
                target,
                alpha=1e-8,
                kind=kind,
            )
            predicted = predict_transition(np, model, state, clean, tokens)
            np.testing.assert_allclose(predicted, target, rtol=1e-5, atol=1e-5)

    def test_vector_metrics_report_zero_baseline_relative_error(self) -> None:
        actual = np.asarray([[1.0, 0.0], [0.0, 2.0]])

        exact = vector_metrics(np, actual, actual)
        zero = vector_metrics(np, actual, np.zeros_like(actual))

        self.assertAlmostEqual(exact["normalized_mse_against_zero"], 0.0)
        self.assertAlmostEqual(zero["normalized_mse_against_zero"], 1.0)

    def test_physical_state_uses_disjoint_task_blocks(self) -> None:
        first = task_block_physical_state(
            np, [1.0, 2.0], task_id=0, coordinate_width=3, task_count=2
        )
        second = task_block_physical_state(
            np, [1.0, 2.0], task_id=1, coordinate_width=3, task_count=2
        )

        np.testing.assert_array_equal(first, [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(second, [0.0, 0.0, 0.0, 1.0, 2.0, 0.0])

    def test_source_context_adds_the_clean_source_boundary(self) -> None:
        path = np.zeros((32, 32, 3), dtype=np.float32)
        path[np.arange(32), np.arange(32)] = np.arange(32)[:, None]
        clean = np.repeat(np.arange(32)[:, None], 3, axis=1).astype(np.float32)
        state = np.ones((32, 3), dtype=np.float32)
        rows = [
            {
                "context": {"action_token_position": 2},
                "path": path,
                "clean": clean,
            }
        ]

        features = endpoint_features(
            np, rows, [0], state, kind="source_context"
        )

        self.assertEqual(features.shape, (32, 51))
        np.testing.assert_array_equal(features[5, 16:19], clean[5])

    def test_local_risk_features_follow_eligibility_mask(self) -> None:
        path = np.zeros((32, 32, 3), dtype=np.float32)
        path[np.arange(32), np.arange(32), 0] = np.arange(32)
        eligible = np.zeros(32, dtype=int)
        eligible[[2, 7]] = 1
        silent = np.zeros(32, dtype=int)
        silent[7] = 1
        rows = [
            {
                "context": {
                    "context_id": "c000",
                    "task_id": 0,
                    "episode_index": 1,
                    "phase": "early",
                    "action_token_position": 0,
                },
                "path": path,
                "physical": np.asarray([1.0, 2.0]),
                "record_id": np.asarray([f"r{index}" for index in range(32)]),
                "eligible": eligible,
                "silent_failure": silent,
            }
        ]

        features, records, labels = local_risk_features(
            np, rows, [0], fork_width=5
        )

        self.assertEqual(features.shape[0], 2)
        self.assertEqual([row["source_layer"] for row in records], [2, 7])
        self.assertEqual(labels.tolist(), [0, 1])

    def test_cache_state_accumulates_every_live_layer_entry(self) -> None:
        sources = []
        boundaries = []
        sketches = []
        for source in range(31):
            for boundary in range(source + 1, 32):
                sources.append(source)
                boundaries.append(boundary)
                sketches.append([float(boundary), 1.0])

        state = cumulative_cache_state(
            np,
            sketches,
            sources,
            boundaries,
            width=2,
        )

        np.testing.assert_array_equal(state[0, 1], [1.0, 1.0])
        np.testing.assert_array_equal(state[0, 3], [6.0, 3.0])
        np.testing.assert_array_equal(state[2, 3], [3.0, 1.0])
        np.testing.assert_array_equal(state[31, 31], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
