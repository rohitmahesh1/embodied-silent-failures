import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.prepare_graph_campaign import (
    _clean_stages,
    _paired_stages,
)


def _completion(task_id: int, episode_index: int, success: bool, steps: int) -> dict:
    return {
        "status": "complete",
        "condition": "clean",
        "task_id": task_id,
        "episode_index": episode_index,
        "success": success,
        "policy_steps": steps,
        "_path": f"task{task_id}--ep{episode_index}.complete.json",
        "_sha256": f"sha-{task_id}-{episode_index}",
    }


class GraphCampaignTests(unittest.TestCase):
    def test_clean_stages_are_balanced_and_disjoint(self) -> None:
        clean = {}
        for task_id in range(10):
            for episode_index in range(45):
                success = episode_index >= 20
                steps = 420 if episode_index in (0, 20) else 250
                clean[(task_id, episode_index)] = _completion(
                    task_id, episode_index, success, steps
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            stages = _clean_stages(clean, root, seed=13)
            manifests = [
                json.loads(Path(stage["manifest"]).read_text(encoding="utf-8"))
                for stage in stages
            ]

        census = manifests[:5]
        temporal = manifests[5:]
        census_trials = {
            (item["task_id"], item["episode_index"])
            for manifest in census
            for item in manifest["trials"]
        }
        temporal_trials = {
            (item["task_id"], item["episode_index"])
            for manifest in temporal
            for item in manifest["trials"]
        }
        self.assertEqual(len(census_trials), 100)
        self.assertEqual(len(temporal_trials), 20)
        self.assertFalse(census_trials & temporal_trials)
        for manifest in census:
            strata = {
                (item["task_id"], item["source_success"])
                for item in manifest["trials"]
            }
            self.assertEqual(len(manifest["trials"]), 20)
            self.assertEqual(len(strata), 20)

    def test_paired_stages_select_only_stale_failures_with_successful_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale_dir = root / "stale"
            control_dir = root / "control"
            manifest_dir = root / "campaign" / "manifests"
            stale_dir.mkdir()
            control_dir.mkdir()
            manifest_dir.mkdir(parents=True)
            source_trials = []
            for episode_index in range(26):
                task_id = episode_index % 10
                key = f"task{task_id}--ep{episode_index}"
                common = {
                    "status": "complete",
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "trial_seed": episode_index,
                    "initial_state_sha256": key,
                }
                stale = {
                    **common,
                    "condition": "stale_image",
                    "success": False,
                    "fault": {"policy_step": 10},
                }
                control = {
                    **common,
                    "condition": "current_image_control",
                    "success": True,
                }
                (stale_dir / f"{key}.complete.json").write_text(
                    json.dumps(stale), encoding="utf-8"
                )
                (control_dir / f"{key}.complete.json").write_text(
                    json.dumps(control), encoding="utf-8"
                )
                source_trials.append(
                    {
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "stale_image": {
                            "policy_step": 10,
                            "image_lag": 1,
                            "source_policy_step": 9,
                        },
                    }
                )
            source = root / "source.json"
            source.write_text(
                json.dumps({"schema_version": 1, "trials": source_trials}),
                encoding="utf-8",
            )

            stages, summary = _paired_stages(
                stale_dir, control_dir, source, root / "campaign"
            )

        self.assertEqual(summary["pair_count"], 26)
        self.assertEqual(len(stages), 20)
        self.assertEqual(sum(item["expected_trials"] for item in stages), 52)


if __name__ == "__main__":
    unittest.main()
