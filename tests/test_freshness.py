import unittest

from embodied_silent_failures.freshness import (
    hold_action,
    observe_freshness,
    summarize_freshness,
)


class DType:
    str = "|u1"


class Array:
    def __init__(self, values: list[int], shape: tuple[int, ...]):
        self.values = values
        self.shape = shape
        self.dtype = DType()

    def copy(self):
        return Array(list(self.values), self.shape)

    def tobytes(self) -> bytes:
        return bytes(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]

    def __setitem__(self, index: int, value: int) -> None:
        self.values[index] = value


class Arrays:
    @staticmethod
    def asarray(value):
        return value

    @staticmethod
    def ascontiguousarray(value):
        return value

    @staticmethod
    def array_equal(left: Array, right: Array) -> bool:
        return (
            left.shape == right.shape
            and left.dtype.str == right.dtype.str
            and left.values == right.values
        )

    @staticmethod
    def zeros(size: int, dtype: DType) -> Array:
        result = Array([0] * size, (size,))
        result.dtype = dtype
        return result


class FreshnessTests(unittest.TestCase):
    def test_exact_replay_alarms_without_trusting_metadata(self) -> None:
        previous = Array(list(range(12)), (2, 2, 3))
        signals = observe_freshness(
            Arrays,
            policy_step=8,
            policy_image=previous.copy(),
            previous_policy_image=previous,
            image_source_policy_step=7,
        )

        self.assertTrue(signals.source_metadata_alarm)
        self.assertFalse(signals.relabelled_metadata_alarm)
        self.assertTrue(signals.exact_duplicate_alarm)
        self.assertTrue(signals.selected_alarm("source_metadata"))
        self.assertTrue(signals.selected_alarm("exact_duplicate"))
        self.assertEqual(signals.source_metadata_age_steps, 1)
        self.assertEqual(signals.input_sha256, signals.previous_input_sha256)

    def test_clean_changed_image_does_not_alarm(self) -> None:
        previous = Array([0] * 12, (2, 2, 3))
        current = previous.copy()
        current[0] = 1
        signals = observe_freshness(
            Arrays,
            policy_step=8,
            policy_image=current,
            previous_policy_image=previous,
            image_source_policy_step=8,
        )

        self.assertFalse(signals.source_metadata_alarm)
        self.assertFalse(signals.exact_duplicate_alarm)
        self.assertFalse(signals.selected_alarm("either"))

    def test_first_image_has_no_duplicate_predecessor(self) -> None:
        image = Array([0] * 12, (2, 2, 3))
        signals = observe_freshness(
            Arrays,
            policy_step=0,
            policy_image=image,
            previous_policy_image=None,
            image_source_policy_step=0,
        )

        self.assertFalse(signals.exact_duplicate_alarm)
        self.assertIsNone(signals.previous_input_sha256)

    def test_hold_suppresses_motion_and_preserves_gripper(self) -> None:
        previous = Array([1, 2, 3, 4, 5, 6, 255], (7,))

        held = hold_action(Arrays, previous)

        self.assertEqual(held.values, [0, 0, 0, 0, 0, 0, 255])

    def test_summary_keeps_detection_and_response_counts_separate(self) -> None:
        rows = [
            {
                "freshness/source_metadata_alarm": False,
                "freshness/relabelled_metadata_alarm": False,
                "freshness/exact_duplicate_alarm": False,
                "freshness/selected_gate_alarm": False,
                "freshness/response_applied": False,
            },
            {
                "freshness/source_metadata_alarm": True,
                "freshness/relabelled_metadata_alarm": False,
                "freshness/exact_duplicate_alarm": True,
                "freshness/selected_gate_alarm": True,
                "freshness/response_applied": True,
            },
        ]

        summary = summarize_freshness(
            rows, gate="exact_duplicate", response="hold"
        )

        self.assertEqual(summary["evaluated_policy_steps"], 2)
        self.assertEqual(summary["source_metadata_alarms"], 1)
        self.assertEqual(summary["relabelled_metadata_alarms"], 0)
        self.assertEqual(summary["exact_duplicate_alarms"], 1)
        self.assertEqual(summary["responses_applied"], 1)


if __name__ == "__main__":
    unittest.main()
