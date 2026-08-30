import unittest

from embodied_silent_failures.analyze_language_interfaces import (
    replay_row,
    replay_summary,
)


class LanguageInterfaceAnalysisTests(unittest.TestCase):
    def test_nontrivial_exact_replay_is_counted_as_closure(self) -> None:
        context = {
            "context_id": "c000",
            "analysis_split": "development",
            "task_id": 0,
            "episode_index": 0,
            "phase": "early",
            "action_token_position": 2,
        }
        intervention = {
            "layer_index": 10,
            "injection": {"exact_equal": False},
            "propagation": [{"layer_index": 11, "exact_equal": False}],
            "action_tokens_exact_equal": False,
            "executed_command": {"exact_equal": False},
            "cache_precondition": {
                "key": {"all_coordinates_exact": True},
                "value": {"all_coordinates_exact": True},
            },
        }
        replay = {
            "status": "complete",
            "injection_layer": 10,
            "boundary_kind": "immediate",
            "boundary_layer": 11,
            "cache_cut": {
                "keys": {"all_coordinates_exact": True},
                "values": {"all_coordinates_exact": True},
            },
            "residual_path": {"all_coordinates_exact": True},
            "attention_cache_keys": {"all_coordinates_exact": True},
            "attention_cache_values": {"all_coordinates_exact": True},
            "action_logits_exact_equal": True,
            "action_tokens_exact_equal": True,
            "raw_action": {"exact_equal": True},
            "executed_command": {"exact_equal": True},
        }

        row = replay_row(context=context, intervention=intervention, replay=replay)
        summary = replay_summary([row])

        self.assertTrue(row["closure_exact"])
        self.assertTrue(row["boundary_residual_nontrivial"])
        self.assertEqual(
            summary["exact_closure_given_nontrivial_boundary"],
            {"numerator": 1, "denominator": 1},
        )

    def test_replay_mismatch_is_retained_as_a_result(self) -> None:
        context = {
            "context_id": "c000",
            "analysis_split": "development",
            "task_id": 0,
            "episode_index": 0,
            "phase": "early",
            "action_token_position": 2,
        }
        intervention = {
            "layer_index": 10,
            "injection": {"exact_equal": False},
            "propagation": [{"layer_index": 11, "exact_equal": False}],
            "action_tokens_exact_equal": False,
            "executed_command": {"exact_equal": False},
            "cache_precondition": {
                "key": {"all_coordinates_exact": True},
                "value": {"all_coordinates_exact": True},
            },
        }
        replay = {
            "status": "complete",
            "injection_layer": 10,
            "boundary_kind": "immediate",
            "boundary_layer": 11,
            "cache_cut": {
                "keys": {"all_coordinates_exact": True},
                "values": {"all_coordinates_exact": True},
            },
            "residual_path": {"all_coordinates_exact": False},
            "attention_cache_keys": {"all_coordinates_exact": True},
            "attention_cache_values": {"all_coordinates_exact": True},
            "action_logits_exact_equal": False,
            "action_tokens_exact_equal": False,
            "raw_action": {"exact_equal": False},
            "executed_command": {"exact_equal": False},
        }

        row = replay_row(context=context, intervention=intervention, replay=replay)

        self.assertFalse(row["closure_exact"])
        self.assertEqual(replay_summary([row])["exact_closure_records"], 0)


if __name__ == "__main__":
    unittest.main()
