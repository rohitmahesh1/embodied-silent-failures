import unittest

from embodied_silent_failures.run_language_campaign import (
    _immutable_run_identity,
    _select_contexts,
)


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

    def test_resume_identity_ignores_only_the_launch_time(self) -> None:
        run = {
            "campaign": "language",
            "condition": "activation_fault",
            "worker_shard": 0,
            "context_ids": ["c000"],
            "limits": {"maximum_contexts": None},
            "execution": {"started_at": "first", "manifest_file_sha256": "abc"},
        }
        resumed = {
            **run,
            "execution": {"started_at": "second", "manifest_file_sha256": "abc"},
        }

        self.assertEqual(_immutable_run_identity(run), _immutable_run_identity(resumed))

    def test_resume_identity_includes_contexts_and_provenance(self) -> None:
        run = {
            "campaign": "language",
            "condition": "activation_fault",
            "worker_shard": 0,
            "context_ids": ["c000"],
            "limits": {},
            "execution": {"manifest_file_sha256": "abc"},
        }

        changed_context = {**run, "context_ids": ["c001"]}
        changed_manifest = {
            **run,
            "execution": {"manifest_file_sha256": "different"},
        }
        self.assertNotEqual(
            _immutable_run_identity(run), _immutable_run_identity(changed_context)
        )
        self.assertNotEqual(
            _immutable_run_identity(run), _immutable_run_identity(changed_manifest)
        )


if __name__ == "__main__":
    unittest.main()
