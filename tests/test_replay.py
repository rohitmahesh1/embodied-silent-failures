import csv
import pickle
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.replay import (
    ACTION_COLUMNS,
    load_clean_trace,
    observation_error,
    replay_action,
)


class ReplayTests(unittest.TestCase):
    class Array:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)
            self.size = len(self.values)

        def astype(self, _dtype):
            return self

        def __sub__(self, other):
            return ReplayTests.Array(
                left - right for left, right in zip(self.values, other.values)
            )

    class Arrays:
        float64 = float

        @staticmethod
        def asarray(values, dtype=None):
            del dtype
            return ReplayTests.Array(values)

        @staticmethod
        def abs(value):
            return ReplayTests.Array(abs(item) for item in value.values)

        @staticmethod
        def max(value):
            return max(value.values)

    def make_trace(self, directory: Path) -> dict:
        rows = [
            {column: str(index + offset) for offset, column in enumerate(ACTION_COLUMNS)}
            for index in (0, 10)
        ]
        with (directory / "trace.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(ACTION_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        with (directory / "trace.pkl").open("wb") as file:
            pickle.dump(
                {
                    "hidden_states": [[[0.0] * 4] * 7 for _ in range(2)],
                    "observations": {"robot": [[1.0], [2.0]]},
                },
                file,
            )
        return {
            "condition": "clean",
            "success": True,
            "policy_steps": 2,
            "_source_dir": str(directory),
            "files": {"csv": "trace.csv", "pickle": "trace.pkl"},
        }

    def test_loads_actions_features_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.make_trace(Path(directory))
            trace = load_clean_trace(result)

            self.assertEqual(
                replay_action(self.Arrays, trace, 1).values,
                list(range(10, 17)),
            )
            self.assertEqual(
                observation_error(self.Arrays, trace, {"robot": [2.25]}, 1),
                0.25,
            )

    def test_rejects_incomplete_clean_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.make_trace(Path(directory))
            result["policy_steps"] = 3
            with self.assertRaises(ValueError):
                load_clean_trace(result)


if __name__ == "__main__":
    unittest.main()
