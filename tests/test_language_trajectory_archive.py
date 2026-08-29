import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class LanguageTrajectoryArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("NumPy is required") from error
        cls.np = np

    def test_archive_keeps_post_intervention_state_images_and_named_values(self) -> None:
        from embodied_silent_failures.language_trajectory_archive import (
            TrajectoryRecorder,
        )

        runtime = SimpleNamespace(np=self.np)
        recorder = TrajectoryRecorder(runtime)

        def tensor(value):
            return SimpleNamespace(numpy=lambda: self.np.asarray(value))

        for step in (10, 11):
            logits = SimpleNamespace(
                sequence_token_ids=tensor(self.np.arange(20, dtype=self.np.int64)),
                action_token_logits=tensor(
                    self.np.full((7, 256), step, dtype=self.np.float32)
                ),
                top_token_ids=tensor(self.np.zeros((7, 32), dtype=self.np.int64)),
                top_token_logits=tensor(
                    self.np.full((7, 32), step, dtype=self.np.float32)
                ),
                log_normalizer=tensor(
                    self.np.full((7,), step, dtype=self.np.float32)
                ),
                entropy=tensor(self.np.full((7,), step, dtype=self.np.float32)),
            )
            decision = SimpleNamespace(
                raw_action=self.np.full((7,), step, dtype=self.np.float32),
                action_tokens=tuple(range(7)),
                generation_logits=logits,
            )
            recorder.append(
                policy_step=step,
                stage="before_action",
                simulator_state=self.np.asarray([step, step + 1], dtype=self.np.float64),
                observation={
                    "agentview_image": self.np.full((2, 2, 3), step, dtype=self.np.uint8),
                    "robot0_eef_pos": self.np.asarray([step], dtype=self.np.float32),
                },
            )
            recorder.append_decision(
                policy_step=step,
                decision=decision,
                executed_command=self.np.full((7,), step, dtype=self.np.float32),
            )
        recorder.append(
            policy_step=12,
            stage="after_final_action",
            simulator_state=self.np.asarray([12, 13], dtype=self.np.float64),
            observation={
                "agentview_image": self.np.full((2, 2, 3), 12, dtype=self.np.uint8),
                "robot0_eef_pos": self.np.asarray([12], dtype=self.np.float32),
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            record = recorder.write(path)
            with self.np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["simulator_state"].shape, (3, 2))
                self.assertEqual(archive["snapshot_stage"].tolist(), [0, 0, 1])
                self.assertEqual(archive["decision_policy_step"].tolist(), [10, 11])
                self.assertEqual(archive["sequence_token_ids"].shape, (2, 20))
                self.assertEqual(archive["action_token_logits"].shape, (2, 7, 256))
                image_record = next(
                    value
                    for value in record["observations"]
                    if value["name"] == "agentview_image"
                )
                self.assertEqual(archive[image_record["archive_key"]].shape, (3, 2, 2, 3))
            self.assertEqual(record["snapshot_count"], 3)
            self.assertEqual(record["decision_count"], 2)


if __name__ == "__main__":
    unittest.main()
