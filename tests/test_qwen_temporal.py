import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.analyze_qwen_temporal import (
    _exact_binomial_two_sided,
    _exact_sign_flip,
    analyze,
)
from embodied_silent_failures.qwen_artifacts import file_sha256


class QwenTemporalAnalysisTests(unittest.TestCase):
    def _trial(
        self,
        root: Path,
        source: str,
        *,
        task_id: int,
        stale_alarms: list[bool | None],
        control_alarms: list[bool | None],
    ) -> None:
        fault_step = 5
        condition = "stale_image" if source == "stale" else "current_image_control"
        success = source == "control"
        completion = {
            "status": "complete",
            "condition": condition,
            "task_id": task_id,
            "episode_index": 2,
            "initial_state_sha256": f"state-{task_id}",
            "trial_seed": 10 + task_id,
            "success": success,
            "counterfactual_replay": {
                "enabled": True,
                "replayed_policy_steps": fault_step,
                "maximum_numeric_observation_error": 0.0,
                "observation_tolerance": 1e-6,
            },
        }
        source_root = root / "sources"
        source_root.mkdir(exist_ok=True)
        completion_path = source_root / f"{source}-{task_id}.complete.json"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        alarms = stale_alarms if source == "stale" else control_alarms
        timeline = [
            {
                "policy_step": step,
                "alarm": alarm,
                "frame_sha256": [f"{source}-{task_id}-{step}"],
            }
            for step, alarm in zip((0, 5, 10, 15), alarms, strict=True)
        ]
        value = {
            "schema_version": 1,
            "status": "complete",
            "configuration_sha256": "configuration",
            "source": source,
            "task_id": task_id,
            "episode_index": 2,
            "condition": condition,
            "success": success,
            "fault": {"policy_step": fault_step, "trial_seed": 10 + task_id},
            "completion_path": str(completion_path),
            "completion_sha256": file_sha256(completion_path),
            "video_path": str(root / f"{source}-{task_id}.mp4"),
            "expected_query_steps": [0, 5, 10, 15],
            "timeline": timeline,
        }
        (root / f"{source}--task{task_id}--ep2.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    def test_primary_result_uses_pairs_and_pre_post_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ("stale", "control"):
                self._trial(
                    root,
                    source,
                    task_id=0,
                    stale_alarms=[False, False, True, True],
                    control_alarms=[False, False, False, False],
                )
                self._trial(
                    root,
                    source,
                    task_id=1,
                    stale_alarms=[False, True, False, False],
                    control_alarms=[False, False, False, False],
                )
            result = analyze(root)

        self.assertEqual(result["source"]["pair_count"], 2)
        self.assertEqual(result["primary_result"]["estimate"], 0.25)
        self.assertEqual(result["primary_result"]["pair_directions"], {"negative": 1, "positive": 1})
        self.assertEqual(result["first_exposed_query"]["stale_only_alarm"], 1)
        self.assertEqual(result["relative_queries"][0]["supported_pairs"], 2)
        self.assertEqual(result["relative_queries"][0]["policy_step_offset_range"], [1, 5])
        self.assertEqual(
            result["first_alarm_timing_on_common_support"]["stale"][
                "before_or_at_intervention"
            ],
            1,
        )
        self.assertEqual(
            result["first_alarm_timing_on_common_support"][
                "pairs_with_no_pre_intervention_alarm"
            ]["stale_only"],
            1,
        )

    def test_indeterminate_queries_are_excluded_without_becoming_no_alarm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in ("stale", "control"):
                self._trial(
                    root,
                    source,
                    task_id=0,
                    stale_alarms=[False, False, None, True],
                    control_alarms=[False, False, False, False],
                )
            result = analyze(root)

        pair = result["pairs"][0]
        self.assertEqual(pair["post_determinate_queries"], 1)
        self.assertEqual(pair["post_indeterminate_queries"], 1)
        self.assertEqual(pair["post_stale_minus_control_alarm_fraction"], 1.0)
        self.assertEqual(result["first_exposed_query"]["indeterminate"], 1)

    def test_exact_paired_references_handle_ties_and_discordance(self) -> None:
        reference = _exact_sign_flip([1.0, -1.0, 0.0])
        self.assertEqual(reference["assignments"], 4)
        self.assertEqual(reference["two_sided_p"], 1.0)
        self.assertEqual(_exact_binomial_two_sided(3, 0), 0.25)
        self.assertIsNone(_exact_binomial_two_sided(0, 0))


if __name__ == "__main__":
    unittest.main()
