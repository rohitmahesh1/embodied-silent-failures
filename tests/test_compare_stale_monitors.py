import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.compare_stale_monitors import compare
from embodied_silent_failures.stale_monitor_inputs import FROZEN_SAFE_MLP


class CompareStaleMonitorsTests(unittest.TestCase):
    def _completion(
        self,
        *,
        condition: str,
        episode: int,
        success: bool,
        freshness: bool,
        alarm: bool = False,
    ) -> dict:
        common = {
            "schema_version": 1,
            "status": "complete",
            "condition": condition,
            "task_id": 0,
            "episode_index": episode,
            "task_suite_name": "libero_10",
            "task_description": "put away the cup",
            "initial_state_sha256": f"state-{episode}",
            "trial_seed": 100 + episode,
            "maximum_policy_steps": 520,
            "policy_steps": 20,
            "success": success,
        }
        if condition == "stale_image":
            fault = {
                "kind": "stale_image",
                "policy_step": 4,
                "source_policy_step": 3,
                "image_lag": 1,
                "trial_seed": 100 + episode,
            }
        else:
            fault = {
                "kind": "current_image_control",
                "policy_step": 4,
                "input_policy_step": 4,
                "matched_stale_source_policy_step": 3,
                "matched_stale_image_lag": 1,
                "trial_seed": 100 + episode,
            }
        if freshness:
            fault["freshness_at_intervention"] = {
                "source_metadata_alarm": alarm,
                "relabelled_metadata_alarm": False,
                "exact_duplicate_alarm": alarm,
                "source_metadata_age_steps": int(alarm),
                "input_sha256": "previous" if alarm else "current",
                "previous_input_sha256": "previous",
            }
            common["freshness"] = {
                "evaluated_policy_steps": 20,
                "source_metadata_alarms": int(alarm),
                "relabelled_metadata_alarms": 0,
                "exact_duplicate_alarms": int(alarm),
            }
        common["fault"] = fault
        common["counterfactual_replay"] = {
            "enabled": True,
            "maximum_numeric_observation_error": 0.0,
            "observation_tolerance": 1e-6,
            "replayed_policy_steps": fault["policy_step"],
        }
        return common

    def _write_completion(self, directory: Path, result: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"task{result['task_id']}--ep{result['episode_index']}.complete.json"
        )
        path.write_text(json.dumps(result), encoding="utf-8")

    def _write_safe(
        self, path: Path, condition: str, results: list[dict], alarm: bool = False
    ) -> None:
        records = []
        for result in results:
            windows = {
                name: {"triggered": alarm, "first_step": 4 if alarm else None}
                for name in (
                    "within_5_steps",
                    "within_10_steps",
                    "within_25_steps",
                    "post_fault_any",
                )
            }
            records.append(
                {
                    "condition": condition,
                    "task_id": result["task_id"],
                    "episode_index": result["episode_index"],
                    "success": result["success"],
                    "fault": result["fault"],
                    "alarms": {"0.1": windows},
                }
            )
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "experiment_code_revision": "test-revision",
                    "safe_revision": "test-safe-revision",
                    "alarm_rule": "score >= frozen time-varying upper band",
                    "monitor": {**FROZEN_SAFE_MLP, "primary_alpha": 0.1},
                    "records": records,
                }
            ),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> dict[str, Path]:
        paths = {
            name: root / name
            for name in ("stale", "current", "fresh-stale", "fresh-current")
        }
        stale = self._completion(
            condition="stale_image",
            episode=0,
            success=False,
            freshness=False,
        )
        current = self._completion(
            condition="current_image_control",
            episode=0,
            success=True,
            freshness=False,
        )
        fresh_stale = self._completion(
            condition="stale_image",
            episode=0,
            success=True,
            freshness=True,
            alarm=True,
        )
        fresh_current = self._completion(
            condition="current_image_control",
            episode=0,
            success=True,
            freshness=True,
            alarm=False,
        )
        for name, result in (
            ("stale", stale),
            ("current", current),
            ("fresh-stale", fresh_stale),
            ("fresh-current", fresh_current),
        ):
            self._write_completion(paths[name], result)
        paths["safe-stale"] = root / "safe-stale.json"
        paths["safe-current"] = root / "safe-current.json"
        self._write_safe(paths["safe-stale"], "stale_image", [stale])
        self._write_safe(
            paths["safe-current"], "current_image_control", [current]
        )
        return paths

    def _compare(self, paths: dict[str, Path]) -> dict:
        return compare(
            paths["stale"],
            paths["current"],
            paths["safe-stale"],
            paths["safe-current"],
            paths["fresh-stale"],
            paths["fresh-current"],
        )

    def test_uses_no_response_outcome_and_shadow_freshness_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._compare(self._fixture(Path(directory)))

        self.assertEqual(result["outcome_population"]["stale_only_failure"], 1)
        detection = result["causal_failure_detection"]
        self.assertEqual(
            detection["freshness_at_intervention"]["exact_duplicate"]["count"], 1
        )
        self.assertEqual(
            detection["safe_mlp_alpha_0_1"]["within_25_steps"]["count"], 0
        )
        self.assertIn("no-response", result["interpretation_boundary"]["outcomes"])

    def test_rejects_missing_freshness_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            next(paths["fresh-stale"].glob("*.complete.json")).unlink()

            with self.assertRaisesRegex(ValueError, "no stale_image completion"):
                self._compare(paths)

    def test_rejects_a_different_safe_monitor_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            value = json.loads(paths["safe-stale"].read_text(encoding="utf-8"))
            value["monitor"]["checkpoint_sha256"] = "lstm-or-other-checkpoint"
            paths["safe-stale"].write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                self._compare(paths)

    def test_rejects_cross_campaign_step_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            marker = next(paths["fresh-stale"].glob("*.complete.json"))
            value = json.loads(marker.read_text(encoding="utf-8"))
            value["fault"]["policy_step"] = 5
            value["counterfactual_replay"]["replayed_policy_steps"] = 5
            marker.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "policy_step"):
                self._compare(paths)


if __name__ == "__main__":
    unittest.main()
