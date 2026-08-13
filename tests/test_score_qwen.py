import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.evidence_graph.qwen import (
    build_messages,
    parse_response,
    prompt_sha256,
    query_steps,
    selected_frame_steps,
    trajectory_prediction,
)
from embodied_silent_failures.qwen_artifacts import load_trial_manifest
from embodied_silent_failures.prepare_qwen_campaign import (
    select_causal_pairs,
    select_native_trials,
)


class QwenScoringTests(unittest.TestCase):
    def test_native_selection_is_balanced_and_repeatable(self) -> None:
        records = {}
        for task_id in range(10):
            for episode_index in range(45):
                success = episode_index >= 20
                records[(task_id, episode_index)] = {
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "success": success,
                    "policy_steps": 420 if episode_index in (0, 20) else 250,
                }

        first = select_native_trials(records, seed=13)
        second = select_native_trials(records, seed=13)
        strata = [(item["task_id"], item["success"]) for item in first]

        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(len(set(strata)), 20)
        self.assertTrue(all(strata.count(stratum) == 2 for stratum in set(strata)))

    def test_causal_selection_requires_stale_failure_and_control_success(self) -> None:
        stale = {}
        control = {}
        for episode_index in range(27):
            key = (episode_index % 10, episode_index)
            common = {
                "task_id": key[0],
                "episode_index": key[1],
                "trial_seed": episode_index,
                "initial_state_sha256": str(episode_index),
                "fault": {"policy_step": 10},
            }
            stale[key] = {**common, "success": episode_index == 26}
            control[key] = {**common, "success": True}

        pairs = select_causal_pairs(stale, control)

        self.assertEqual(len(pairs), 26)
        self.assertTrue(all(left["success"] is False for left, _right in pairs))

    def test_frame_selection_is_chronological_and_ends_at_query(self) -> None:
        self.assertEqual(selected_frame_steps(10, 4, 3), (1, 4, 7, 10))
        self.assertEqual(selected_frame_steps(2, 8, 2), (0, 2))
        self.assertEqual(query_steps(11, 4), (0, 4, 8))

    def test_prompt_and_media_order_are_fixed(self) -> None:
        frames = [object(), object()]
        messages = build_messages(frames, "move the bowl", 4)
        content = messages[1]["content"]

        self.assertIs(content[0]["image"], frames[0])
        self.assertIs(content[1]["image"], frames[1])
        self.assertEqual(content[-1]["type"], "text")
        self.assertEqual(
            prompt_sha256("move the bowl", 4),
            prompt_sha256("move the bowl", 4),
        )
        self.assertNotEqual(
            prompt_sha256("move the bowl", 4),
            prompt_sha256("move the bowl", 5),
        )

    def test_response_parser_does_not_repair_or_coerce(self) -> None:
        decision = parse_response('{"failure_now":1,"reason":"object dropped"}')
        self.assertTrue(decision.alarm)
        self.assertEqual(decision.reason, "object dropped")

        invalid = (
            '```json\n{"failure_now":1,"reason":"drop"}\n```',
            '{"failure_now":true,"reason":"drop"}',
            '{"failure_now":"1","reason":"drop"}',
            '{"failure_now":1,"reason":"drop","confidence":0.9}',
            'answer: {"failure_now":1,"reason":"drop"}',
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_response(value)

    def test_invalid_reply_does_not_become_a_no_alarm_trajectory(self) -> None:
        self.assertTrue(trajectory_prediction([False, True, None]))
        self.assertIsNone(trajectory_prediction([False, None, False]))
        self.assertFalse(trajectory_prediction([False, False]))

    def test_manifest_fixes_existing_video_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "stale"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"schema_version": 1, "condition": "stale_image"}),
                encoding="utf-8",
            )
            video = run_dir / "task0--ep2--succ0.mp4"
            video.write_bytes(b"video")
            completion = {
                "schema_version": 1,
                "status": "complete",
                "condition": "stale_image",
                "task_id": 0,
                "episode_index": 2,
                "task_description": "put the cup away",
                "success": False,
                "policy_steps": 20,
                "fault": {"policy_step": 10},
                "files": {"video": video.name},
            }
            (run_dir / "task0--ep2.complete.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            manifest_path = root / "qwen-trials.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selection_basis": "protocol:test:matched-trial",
                        "trials": [
                            {
                                "source": "stale",
                                "run_dir": "stale",
                                "task_id": 0,
                                "episode_index": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest, trials = load_trial_manifest(manifest_path)

        self.assertEqual(manifest["selection_basis"], "protocol:test:matched-trial")
        self.assertEqual(trials[0].key, "stale--task0--ep2")
        self.assertEqual(trials[0].run_path.name, "run.json")
        self.assertEqual(trials[0].video_sha256, trials[0].video_sha256.lower())


if __name__ == "__main__":
    unittest.main()
