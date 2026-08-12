import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.fault_manifest import load_fault_manifest
from embodied_silent_failures.plan import Trial


class FaultManifestTests(unittest.TestCase):
    def test_loads_exact_per_trial_faults(self) -> None:
        value = {
            "schema_version": 1,
            "selection_basis": "action_impact_only",
            "trials": [
                {
                    "task_id": 2,
                    "episode_index": 4,
                    "fault": {
                        "site": "final_hidden",
                        "layer": None,
                        "policy_step": 80,
                        "generation_step": 6,
                        "bit_index": 15,
                        "seed": 0,
                        "feature_index": 123,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faults.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            manifest = load_fault_manifest(path)

        self.assertEqual(manifest.selection_basis, "action_impact_only")
        self.assertEqual(manifest.specs[Trial(2, 4)].feature_index, 123)

    def test_rejects_duplicates(self) -> None:
        fault = {
            "site": "final_hidden",
            "layer": None,
            "policy_step": 50,
            "generation_step": 6,
            "bit_index": 15,
            "seed": 0,
        }
        value = {
            "schema_version": 1,
            "selection_basis": "action_impact_only",
            "trials": [
                {"task_id": 0, "episode_index": 0, "fault": fault},
                {"task_id": 0, "episode_index": 0, "fault": fault},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faults.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fault_manifest(path)


if __name__ == "__main__":
    unittest.main()
