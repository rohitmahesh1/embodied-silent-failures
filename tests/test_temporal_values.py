import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from embodied_silent_failures.temporal_values import (
    TemporalValueCollector,
    write_temporal_value_archive,
)


class _NoTensorTorch:
    class Tensor:
        pass


def _site():
    return {
        "site_id": "camera",
        "identity": {
            "kind": "declared_runtime_boundary",
            "event_name": "policy.selected_image",
            "event_call_index": 0,
            "output_port": "value",
        },
        "intervention": {"value_slice": "full"},
    }


class TemporalValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("NumPy is required") from error
        cls.np = np

    def test_declared_boundary_is_collected_without_a_handwritten_hook(self) -> None:
        collector = TemporalValueCollector(_NoTensorTorch, self.np, [_site()])
        collector.begin_capture()
        value = self.np.arange(12, dtype=self.np.uint8).reshape(2, 2, 3)

        observed = collector.boundary(
            "policy.selected_image", value, policy_step=4
        )

        self.assertIs(observed, value)
        self.assertTrue(self.np.array_equal(collector.values["camera"], value))
        self.assertEqual(collector.missing_site_ids(), [])

    def test_archive_preserves_raw_bytes_and_declared_dtype(self) -> None:
        runtime = SimpleNamespace(torch=_NoTensorTorch, np=self.np)
        source = self.np.asarray([1, 2, 3], dtype=self.np.uint16)
        current = self.np.asarray([4, 5, 6], dtype=self.np.uint16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "values.npz"

            record = write_temporal_value_archive(
                path,
                runtime,
                [_site()],
                {"camera": source},
                {"camera": current},
            )

            archive = self.np.load(path, allow_pickle=False)
            entry = record["entries"][0]
            self.assertEqual(entry["source"]["dtype"], source.dtype.str)
            self.assertEqual(
                archive[entry["source"]["archive_key"]].tobytes(),
                source.tobytes(),
            )
            self.assertEqual(
                archive[entry["current"]["archive_key"]].tobytes(),
                current.tobytes(),
            )


if __name__ == "__main__":
    unittest.main()
