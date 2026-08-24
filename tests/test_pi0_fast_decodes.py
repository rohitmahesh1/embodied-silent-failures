import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.analyze_pi0_fast_decodes import (
    analyze_campaign,
    parse_attempt_log,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


class Pi0FastDecodeAnalysisTests(unittest.TestCase):
    def test_duplicate_trace_text_is_one_shape_signature(self):
        message = "ValueError: cannot reshape array of size 68 into shape (7)"
        result = parse_attempt_log(f"{message}\n{message}\n")
        self.assertEqual(result["family"], "fast_dct_shape")
        self.assertEqual(
            result["reshape_signatures"],
            [{"coefficient_count": 68, "action_dimension": 7}],
        )

    def test_campaign_accounting_and_selection_are_mechanical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "run.json",
                {
                    "trial_plan": [
                        {"task_id": 0, "episode_index": 0},
                        {"task_id": 0, "episode_index": 1},
                    ],
                    "repository_states": {
                        "experiment_code": {"revision": "baseline"}
                    },
                },
            )
            _write(
                root / "task0--ep0.complete.json",
                {"task_id": 0, "episode_index": 0},
            )
            _write(
                root / "task0--ep1.unresolved.json",
                {"task_id": 0, "episode_index": 1},
            )
            attempts = []
            for attempt in (1, 2):
                relative = f"logs/task0--ep1--attempt{attempt}.log"
                _write(
                    root / relative,
                    "ValueError: cannot reshape array of size 68 into shape (7)\n",
                )
                attempts.append(
                    {"attempt": attempt, "return_code": 1, "log": relative}
                )
            _write(
                root / "attempts/task0--ep1.json",
                {
                    "task_id": 0,
                    "episode_index": 1,
                    "attempts": attempts,
                },
            )
            _write(
                root / "heartbeats/task0--ep1.json",
                {"environment_step": 5, "model_decisions": 2},
            )

            result = analyze_campaign(root)

        self.assertEqual(
            result["terminal_accounting"],
            {
                "planned": 2,
                "complete": 1,
                "unresolved": 1,
                "all_planned_trials_have_one_terminal_state": True,
            },
        )
        self.assertTrue(
            result["summary"]["all_unresolved_are_fast_dct_shape_failures"]
        )
        self.assertEqual(
            result["audit_selection"]["trials"],
            [
                {
                    "task_id": 0,
                    "episode_index": 1,
                    "baseline_coefficient_count": 68,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
