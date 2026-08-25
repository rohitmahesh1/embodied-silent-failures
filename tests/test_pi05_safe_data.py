import json
import pickle
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from embodied_silent_failures.artifacts import artifact_record
from embodied_silent_failures.pi05_safe_data import export_features


def _write_run(root: Path, task_id: int, episode_index: int) -> None:
    run = {
        "condition": "clean",
        "configuration": {
            "model": "pi0.5",
            "replan_steps": 1,
            "seed": 7,
        },
        "repository_states": {
            "experiment_code": {"revision": "experiment-revision"},
            "openpi": {"revision": "openpi-revision"},
        },
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    name = f"task{task_id}--ep{episode_index}--succ1.pkl"
    payload = {
        "model": "pi0.5",
        "condition": "clean",
        "task_id": task_id,
        "episode_idx": episode_index,
        "episode_success": True,
        "replan_steps": 1,
        "decisions": {
            "pre_velocity": np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(
                2, 3, 4, 5
            )
        },
    }
    pickle_path = root / name
    with pickle_path.open("wb") as file:
        pickle.dump(payload, file)
    completion = {
        "status": "complete",
        "condition": "clean",
        "model": "pi0.5",
        "task_id": task_id,
        "episode_index": episode_index,
        "success": True,
        "replan_steps": 1,
        "model_decisions": 2,
        "files": {"pickle": name},
        "artifact_manifest": [artifact_record(pickle_path)],
    }
    (root / f"task{task_id}--ep{episode_index}.complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )


@unittest.skipIf(np is None, "NumPy is installed in the pi0.5 runtime")
class Pi05SafeDataTests(unittest.TestCase):
    def test_split_campaign_is_merged_with_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_run(first, 0, 0)
            _write_run(second, 0, 1)

            manifest = export_features(
                [second, first], root / "features", expected_replan_steps=1
            )

        self.assertEqual(manifest["rollouts"], 2)
        self.assertEqual(manifest["decisions"], 4)
        self.assertEqual(len(manifest["source"]["runs"]), 2)
        self.assertEqual(
            [
                (item["task_id"], item["episode_index"])
                for item in manifest["source_rollouts"]
            ],
            [(0, 0), (0, 1)],
        )

    def test_split_campaign_rejects_duplicate_rollout_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_run(first, 0, 0)
            _write_run(second, 0, 0)

            with self.assertRaisesRegex(ValueError, "duplicate task/episode"):
                export_features(
                    [first, second], root / "features", expected_replan_steps=1
                )

    def test_split_campaign_rejects_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_run(first, 0, 0)
            _write_run(second, 0, 1)
            run_path = second / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["configuration"]["seed"] = 8
            run_path.write_text(json.dumps(run), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "configuration differs"):
                export_features(
                    [first, second], root / "features", expected_replan_steps=1
                )


if __name__ == "__main__":
    unittest.main()
