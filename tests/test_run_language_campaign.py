import unittest

from embodied_silent_failures.run_language_campaign import _select_contexts


class RunLanguageCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "contexts": [
                {"context_id": "c000", "worker_shard": 0},
                {"context_id": "c001", "worker_shard": 1},
                {"context_id": "c002", "worker_shard": 0},
            ]
        }

    def test_selects_named_contexts_in_manifest_order(self) -> None:
        selected = _select_contexts(self.manifest, 0, ["c002", "c000"])

        self.assertEqual([context["context_id"] for context in selected], ["c000", "c002"])

    def test_rejects_context_from_the_other_worker(self) -> None:
        with self.assertRaises(ValueError):
            _select_contexts(self.manifest, 0, ["c001"])

    def test_rejects_unknown_context(self) -> None:
        with self.assertRaises(ValueError):
            _select_contexts(self.manifest, 0, ["missing"])


if __name__ == "__main__":
    unittest.main()
