import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from embodied_silent_failures.run_openvla import (
    Arguments,
    _array_sha256,
    _prepare_run,
)


class RunTests(unittest.TestCase):
    def arguments(self, output_dir: Path, resume: bool = False) -> Arguments:
        return Arguments(
            checkpoint=Path("/checkpoint"),
            openvla_root=Path("/openvla"),
            libero_root=Path("/libero"),
            output_dir=output_dir,
            task_suite="libero_10",
            task_ids="0",
            episode_start=0,
            episode_stop=1,
            episode_stride=1,
            seed=7,
            wait_steps=10,
            save_video=True,
            resume=resume,
        )

    def test_prepare_run_supports_only_matching_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            metadata = {
                "configuration": {"task": 0},
                "trial_plan": [{"task_id": 0, "episode_index": 0}],
            }
            args = self.arguments(output_dir)
            _prepare_run(args, metadata)

            run_path = output_dir / "run.json"
            self.assertEqual(json.loads(run_path.read_text()), metadata)
            _prepare_run(replace(args, resume=True), metadata)

            changed = {**metadata, "trial_plan": []}
            with self.assertRaises(ValueError):
                _prepare_run(replace(args, resume=True), changed)
            with self.assertRaises(FileExistsError):
                _prepare_run(args, metadata)

    def test_initial_state_hash_includes_values_shape_and_dtype(self) -> None:
        class Array:
            def __init__(self, values: bytes, shape: tuple[int, ...], dtype: str):
                self.values = values
                self.shape = shape
                self.dtype = SimpleNamespace(hasobject=False, str=dtype)

            def tobytes(self) -> bytes:
                return self.values

        arrays = SimpleNamespace(
            asarray=lambda value: value,
            ascontiguousarray=lambda value: value,
        )
        runtime = SimpleNamespace(np=arrays)
        state = Array(b"values", (1, 2), "<f4")
        digest = _array_sha256(runtime, state)

        self.assertEqual(digest, _array_sha256(runtime, Array(b"values", (1, 2), "<f4")))
        self.assertNotEqual(digest, _array_sha256(runtime, Array(b"values", (2, 1), "<f4")))
        self.assertNotEqual(digest, _array_sha256(runtime, Array(b"values", (1, 2), "<f8")))
        self.assertNotEqual(digest, _array_sha256(runtime, Array(b"changed", (1, 2), "<f4")))


if __name__ == "__main__":
    unittest.main()
