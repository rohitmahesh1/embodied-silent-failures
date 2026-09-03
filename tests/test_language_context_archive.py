import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_silent_failures.language_context import (
    CapturedContext,
    replay_context,
    write_captured_context_archive,
)
from embodied_silent_failures.openvla_runtime import array_sha256


class CapturedContextArchiveTests(unittest.TestCase):
    def test_preserves_raw_state_and_named_observations_without_pickle(self) -> None:
        runtime = SimpleNamespace(np=np)
        state = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        observation = {
            "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "robot0_joint_pos": np.asarray([0.25, -0.5], dtype=np.float32),
        }
        captured = CapturedContext(
            observation=observation,
            simulator_state=state,
            simulator_state_sha256=array_sha256(runtime, state),
            prefix_commands=(),
            prefix_hidden_states=(),
            prefix_rows=(),
            source_trace=None,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captured_context.npz"
            manifest = write_captured_context_archive(path, runtime, captured)
            archive = np.load(path, allow_pickle=False)

            np.testing.assert_array_equal(archive["simulator_state"], state)
            by_name = {item["name"]: item for item in manifest["observations"]}
            for name, expected in observation.items():
                record = by_name[name]
                np.testing.assert_array_equal(archive[record["archive_key"]], expected)
                self.assertEqual(record["sha256"], array_sha256(runtime, expected))
            self.assertEqual(
                manifest["simulator_state"]["sha256"],
                array_sha256(runtime, state),
            )
            self.assertEqual(manifest["artifact"]["name"], path.name)

    def test_prefix_replay_reapplies_the_original_trial_seed(self) -> None:
        class Runtime:
            np = np

            def __init__(self) -> None:
                self.seeds = []

            def set_seed_everywhere(self, seed: int) -> None:
                self.seeds.append(seed)

            @staticmethod
            def get_libero_dummy_action(_model_family):
                return np.asarray([0.0])

        class Environment:
            def __init__(self) -> None:
                self.state = np.asarray([0.0])

            def reset(self) -> None:
                self.state = np.asarray([-1.0])

            def set_init_state(self, state):
                self.state = np.asarray(state).copy()
                return {"state": self.state.copy()}

            def step(self, command):
                self.state = self.state + np.asarray(command)
                return {"state": self.state.copy()}, 0.0, False, {}

            def get_sim_state(self):
                return self.state.copy()

        runtime = Runtime()
        environment = Environment()
        captured = CapturedContext(
            observation={"state": np.asarray([3.0])},
            simulator_state=np.asarray([3.0]),
            simulator_state_sha256="unused",
            prefix_commands=(np.asarray([1.0]), np.asarray([2.0])),
            prefix_hidden_states=(),
            prefix_rows=(),
            source_trace=None,
        )

        _observation, record = replay_context(
            runtime,
            environment,
            np.asarray([0.0]),
            captured,
            wait_steps=2,
            trial_seed=1234,
        )

        self.assertEqual(runtime.seeds, [1234])
        self.assertTrue(record["simulator_state_exact_equal"])
        self.assertEqual(record["trial_seed_reapplied"], 1234)


if __name__ == "__main__":
    unittest.main()
