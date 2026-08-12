import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.plan import Trial
from embodied_silent_failures.stale_image_manifest import (
    load_stale_image_manifest,
)


class StaleImageManifestTests(unittest.TestCase):
    def test_loads_explicit_per_trial_stale_image_specs(self) -> None:
        value = {
            "schema_version": 1,
            "selection_basis": "manual_curation",
            "trials": [
                {
                    "task_id": 2,
                    "episode_index": 4,
                    "stale_image": {
                        "policy_step": 80,
                        "image_lag": 4,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            manifest = load_stale_image_manifest(path)

        self.assertEqual(manifest.selection_basis, "manual_curation")
        spec = manifest.specs[Trial(2, 4)]
        self.assertEqual(spec.policy_step, 80)
        self.assertEqual(spec.image_lag, 4)
        self.assertEqual(spec.source_policy_step, 76)

    def test_selects_smallest_gripper_changing_lag_from_probe_records(self) -> None:
        value = {
            "schema_version": 1,
            "selection_basis": "probe_records",
            "records": [
                {
                    "task_id": 0,
                    "episode_index": 1,
                    "policy_step": 64,
                    "clean_reproduces_trace": True,
                    "candidates": [
                        {
                            "image_lag": 4,
                            "source_policy_step": 60,
                            "action_change": {"gripper_changed": True},
                        },
                        {
                            "image_lag": 1,
                            "source_policy_step": 63,
                            "action_change": {"gripper_changed": True},
                        },
                    ],
                },
                {
                    "task_id": 0,
                    "episode_index": 2,
                    "policy_step": 12,
                    "clean_reproduces_trace": False,
                    "candidates": [
                        {
                            "image_lag": 1,
                            "source_policy_step": 11,
                            "action_change": {"gripper_changed": True},
                        }
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            manifest = load_stale_image_manifest(path)

        spec = manifest.specs[Trial(0, 1)]
        self.assertEqual(spec.policy_step, 64)
        self.assertEqual(spec.image_lag, 1)
        self.assertEqual(spec.source_policy_step, 63)
        self.assertEqual(len(manifest.specs), 1)

    def test_rejects_probe_without_any_clean_reproducing_changed_trials(self) -> None:
        value = {
            "schema_version": 1,
            "selection_basis": "probe_records",
            "records": [
                {
                    "task_id": 0,
                    "episode_index": 1,
                    "policy_step": 64,
                    "clean_reproduces_trace": True,
                    "candidates": [
                        {
                            "image_lag": 1,
                            "source_policy_step": 63,
                            "action_change": {"gripper_changed": False},
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_stale_image_manifest(path)


if __name__ == "__main__":
    unittest.main()
