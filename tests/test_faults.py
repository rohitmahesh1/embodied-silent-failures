import unittest

from embodied_silent_failures.faults import (
    FaultSpec,
    _active_token_flat_index,
    _event_seed,
    _indices_for_flat_index,
)


class FaultTests(unittest.TestCase):
    def test_fault_spec_distinguishes_shared_and_action_only_sites(self) -> None:
        shared = FaultSpec("decoder_layer", 15, 50, 0, None, 3)
        exact = FaultSpec("final_hidden", None, 50, 6, 7, 3, 1024)
        action_only = FaultSpec("action_logits", None, 50, 0, 7, 3)

        self.assertEqual(shared.layer, 15)
        self.assertIsNone(action_only.layer)
        self.assertEqual(
            shared.to_dict()["evidence_relation"], "upstream_of_monitor_tap"
        )
        self.assertEqual(exact.feature_index, 1024)
        self.assertEqual(
            exact.to_dict()["evidence_relation"],
            "exact_monitor_input_when_targeting_monitored_token",
        )
        self.assertEqual(
            action_only.to_dict()["evidence_relation"],
            "post_tap_with_autoregressive_feedback",
        )
        with self.assertRaises(ValueError):
            FaultSpec("decoder_layer", None, 50, 0, None, 3)
        with self.assertRaises(ValueError):
            FaultSpec("action_logits", 15, 50, 0, None, 3)
        with self.assertRaises(ValueError):
            FaultSpec("final_hidden", 15, 50, 6, None, 3)

    def test_fault_spec_rejects_invalid_time_and_bit_values(self) -> None:
        invalid = [
            ("decoder_layer", 0, -1, 0, None, 0),
            ("decoder_layer", 0, 0, -1, None, 0),
            ("decoder_layer", 0, 0, 7, None, 0),
            ("decoder_layer", 0, 0, 0, -1, 0),
            ("decoder_layer", 0, 0, 0, None, -1),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                FaultSpec(*values)
        with self.assertRaises(ValueError):
            FaultSpec("final_hidden", None, 0, 6, 7, 0, -1)

    def test_fault_location_seed_is_stable_and_trial_specific(self) -> None:
        spec = FaultSpec("decoder_layer", 15, 50, 0, None, 3)
        self.assertEqual(_event_seed(spec, 7), _event_seed(spec, 7))
        self.assertNotEqual(_event_seed(spec, 7), _event_seed(spec, 8))
        self.assertNotEqual(
            _event_seed(spec, 7),
            _event_seed(FaultSpec("decoder_layer", 16, 50, 0, None, 3), 7),
        )

    def test_fault_spec_round_trips_recorded_metadata(self) -> None:
        spec = FaultSpec("final_hidden", None, 50, 6, 15, 3, 2048)
        self.assertEqual(FaultSpec.from_dict(spec.to_dict()), spec)

    def test_flat_indices_are_recorded_in_tensor_coordinates(self) -> None:
        self.assertEqual(_indices_for_flat_index((2, 3, 4), 0), [0, 0, 0])
        self.assertEqual(_indices_for_flat_index((2, 3, 4), 23), [1, 2, 3])

    def test_fault_targets_a_feature_in_the_active_sequence_token(self) -> None:
        flat_index = _active_token_flat_index((1, 5, 4), feature_index=2)

        self.assertEqual(flat_index, 18)
        self.assertEqual(_indices_for_flat_index((1, 5, 4), flat_index), [0, 4, 2])
        with self.assertRaises(ValueError):
            _active_token_flat_index((2, 5, 4), feature_index=2)
        with self.assertRaises(IndexError):
            _active_token_flat_index((1, 5, 4), feature_index=4)


if __name__ == "__main__":
    unittest.main()
