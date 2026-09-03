import unittest

from embodied_silent_failures.run_intervention_atlas import (
    _immutable_identity,
    _select_contexts,
)


class RunInterventionAtlasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "counts": {"worker_count": 2},
            "contexts": [
                {"context_id": "c0", "worker_shard": 0},
                {"context_id": "c1", "worker_shard": 1},
            ],
        }

    def test_context_selection_never_crosses_worker_shards(self) -> None:
        self.assertEqual(
            _select_contexts(self.manifest, 0, []),
            [{"context_id": "c0", "worker_shard": 0}],
        )
        with self.assertRaisesRegex(ValueError, "another worker shard"):
            _select_contexts(self.manifest, 0, ["c1"])

    def test_resume_identity_ignores_only_start_time(self) -> None:
        left = {
            "campaign": "atlas",
            "worker_shard": 0,
            "context_ids": ["c0"],
            "limits": {},
            "execution": {"started_at": "first", "manifest": "same"},
        }
        right = {
            **left,
            "execution": {"started_at": "second", "manifest": "same"},
        }
        self.assertEqual(_immutable_identity(left), _immutable_identity(right))
        right["execution"]["manifest"] = "different"
        self.assertNotEqual(_immutable_identity(left), _immutable_identity(right))


if __name__ == "__main__":
    unittest.main()
