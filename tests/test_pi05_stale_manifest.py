import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.pi05_stale_manifest import (
    build_manifest,
    load_manifest,
)
from embodied_silent_failures.plan import Trial


class Pi05StaleManifestTests(unittest.TestCase):
    def _baseline(self, root: Path) -> None:
        (root / "run.json").write_text(
            json.dumps(
                {
                    "condition": "clean",
                    "configuration": {"model": "pi0.5", "replan_steps": 5},
                    "repository_states": {
                        "experiment_code": {"revision": "revision"}
                    },
                }
            ),
            encoding="utf-8",
        )
        for task_id in range(10):
            for episode_index in range(2):
                value = {
                    "status": "complete",
                    "condition": "clean",
                    "model": "pi0.5",
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "success": episode_index == 0,
                    "model_decisions": 8 + task_id,
                }
                (root / f"task{task_id}--ep{episode_index}.complete.json").write_text(
                    json.dumps(value), encoding="utf-8"
                )

    def test_selection_is_task_balanced_outcome_blind_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._baseline(root)
            first = root / "first.json"
            second = root / "second.json"

            left = build_manifest(root, first, per_task=1, seed=19)
            right = build_manifest(root, second, per_task=1, seed=19)
            loaded = load_manifest(first)

        self.assertEqual(left["trials"], right["trials"])
        self.assertEqual(len(loaded.specs), 10)
        self.assertEqual({trial.task_id for trial in loaded.specs}, set(range(10)))
        self.assertEqual(
            {trial.episode_index for trial in loaded.specs}, {0}
        )
        self.assertEqual(
            sorted(spec.order_bit for spec in loaded.specs.values()),
            [0] * 5 + [1] * 5,
        )
        for trial, spec in loaded.specs.items():
            with self.subTest(trial=trial):
                self.assertGreaterEqual(spec.intervention_decision, 1)
                self.assertLess(spec.intervention_decision, spec.clean_decisions)
                self.assertEqual(spec.source_decision, spec.intervention_decision - 1)
                self.assertEqual(
                    spec.intervention_environment_step,
                    spec.intervention_decision * 5,
                )

    def test_loader_rejects_changed_derived_site_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._baseline(root)
            path = root / "sites.json"
            build_manifest(root, path, per_task=1, seed=19)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["trials"][0]["source_decision"] += 1
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_manifest_uses_trial_identity_as_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._baseline(root)
            path = root / "sites.json"
            build_manifest(root, path, per_task=1, seed=19)

            manifest = load_manifest(path)

        self.assertIn(Trial(4, 0), manifest.specs)


if __name__ == "__main__":
    unittest.main()
