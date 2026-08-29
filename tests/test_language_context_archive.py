import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_silent_failures.language_context import (
    CapturedContext,
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


if __name__ == "__main__":
    unittest.main()
