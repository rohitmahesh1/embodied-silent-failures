import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.artifacts import (
    completion_path,
    exclusion_path,
    prepare_trial,
    safe_stem,
    write_json_atomic,
)
from embodied_silent_failures.plan import Trial


class ArtifactTests(unittest.TestCase):
    def test_safe_stem_matches_safe_dataset_naming(self) -> None:
        trial = Trial(task_id=3, episode_index=17)
        self.assertEqual(safe_stem(trial, True), "task3--ep17--succ1")
        self.assertEqual(safe_stem(trial, False), "task3--ep17--succ0")

    def test_prepare_trial_removes_incomplete_outputs(self) -> None:
        trial = Trial(task_id=2, episode_index=4)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            stale = output_dir / "task2--ep4--succ0.csv"
            temporary = output_dir / ".task2--ep4--succ0.partial.tmp.mp4"
            unrelated = output_dir / "task2--ep5--succ0.csv"
            similarly_named = output_dir / "task2--ep40--succ0.csv"
            stale.write_text("partial", encoding="utf-8")
            temporary.write_text("partial", encoding="utf-8")
            unrelated.write_text("keep", encoding="utf-8")
            similarly_named.write_text("keep", encoding="utf-8")

            self.assertIsNone(prepare_trial(output_dir, trial, resume=True))
            self.assertFalse(stale.exists())
            self.assertFalse(temporary.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(similarly_named.exists())

    def test_prepare_trial_skips_valid_completion_when_resuming(self) -> None:
        trial = Trial(task_id=1, episode_index=9)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            csv_path = output_dir / "task1--ep9--succ0.csv"
            pickle_path = output_dir / "task1--ep9--succ0.pkl"
            csv_path.touch()
            pickle_path.touch()
            marker = completion_path(output_dir, trial)
            write_json_atomic(
                marker,
                {
                    "status": "complete",
                    "task_id": 1,
                    "episode_index": 9,
                    "files": {
                        "csv": csv_path.name,
                        "pickle": pickle_path.name,
                        "video": None,
                    },
                },
            )

            self.assertEqual(prepare_trial(output_dir, trial, resume=True), "complete")
            with self.assertRaises(FileExistsError):
                prepare_trial(output_dir, trial, resume=False)

    def test_prepare_trial_skips_valid_exclusion_when_resuming(self) -> None:
        trial = Trial(task_id=3, episode_index=7)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            marker = exclusion_path(output_dir, trial)
            write_json_atomic(
                marker,
                {
                    "status": "excluded",
                    "task_id": 3,
                    "episode_index": 7,
                    "reason": "counterfactual_replay_diverged_before_intervention",
                },
            )

            self.assertEqual(prepare_trial(output_dir, trial, resume=True), "excluded")
            with self.assertRaises(FileExistsError):
                prepare_trial(output_dir, trial, resume=False)

    def test_prepare_trial_rejects_missing_completed_artifacts(self) -> None:
        trial = Trial(task_id=1, episode_index=9)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            marker = completion_path(output_dir, trial)
            write_json_atomic(
                marker,
                {
                    "status": "complete",
                    "task_id": 1,
                    "episode_index": 9,
                    "files": {
                        "csv": "task1--ep9--succ0.csv",
                        "pickle": "task1--ep9--succ0.pkl",
                        "video": None,
                    },
                },
            )

            with self.assertRaises(FileNotFoundError):
                prepare_trial(output_dir, trial, resume=True)

    def test_prepare_trial_requires_referenced_evidence_artifacts(self) -> None:
        trial = Trial(task_id=1, episode_index=9)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            output_dir.mkdir()
            csv_path = output_dir / "task1--ep9--succ0.csv"
            pickle_path = output_dir / "task1--ep9--succ0.pkl"
            csv_path.touch()
            pickle_path.touch()
            evidence_dir = Path(directory) / "evidence" / "task1--ep9"
            evidence_dir.mkdir(parents=True)
            for name in (
                "raw.jsonl",
                "annotations.json",
                "graph.json",
                "audit.json",
                "composition.json",
            ):
                (evidence_dir / name).touch()
            marker = completion_path(output_dir, trial)
            write_json_atomic(
                marker,
                {
                    "status": "complete",
                    "task_id": 1,
                    "episode_index": 9,
                    "files": {
                        "csv": csv_path.name,
                        "pickle": pickle_path.name,
                        "video": None,
                    },
                    "evidence_graph": {
                        "directory": "/unavailable/original/path",
                        "directory_relative_to_run": "../evidence/task1--ep9",
                    },
                },
            )

            self.assertEqual(prepare_trial(output_dir, trial, resume=True), "complete")
            (evidence_dir / "audit.json").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_trial(output_dir, trial, resume=True)

    def test_atomic_json_is_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json_atomic(path, {"b": 2, "a": 1})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"a": 1, "b": 2},
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
