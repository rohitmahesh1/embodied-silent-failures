from __future__ import annotations

import unittest
from types import SimpleNamespace

from embodied_silent_failures.language_policy import action_vocabulary_bounds


class LanguagePolicyTests(unittest.TestCase):
    def test_action_vocabulary_excludes_padded_model_outputs(self) -> None:
        model = SimpleNamespace(
            vocab_size=32_000,
            config=SimpleNamespace(n_action_bins=256),
        )

        self.assertEqual(action_vocabulary_bounds(model, 32_064), (31_744, 32_000))

    def test_action_vocabulary_must_fit_language_head(self) -> None:
        model = SimpleNamespace(
            vocab_size=32_000,
            config=SimpleNamespace(n_action_bins=256),
        )

        with self.assertRaisesRegex(ValueError, "outside the language-head output"):
            action_vocabulary_bounds(model, 31_999)


if __name__ == "__main__":
    unittest.main()
