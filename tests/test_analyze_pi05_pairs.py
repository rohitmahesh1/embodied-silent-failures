import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.analyze_pi05_pairs import analyze


def _alarm(detected: bool, before: bool = False):
    return {
        "alarm_before_intervention": before,
        "first_alarm_decision": 2 if detected else None,
        "windows": {
            name: {
                "triggered": detected and not before,
                "first_decision": 2 if detected and not before else None,
            }
            for name in (
                "intervention_decision",
                "within_5_decisions",
                "within_10_decisions",
                "through_terminal_outcome",
            )
        },
    }


class Pi05PairAnalysisTests(unittest.TestCase):
    def _pair(
        self,
        root: Path,
        episode: int,
        *,
        current_success: bool,
        stale_success: bool,
    ) -> None:
        directory = root / "pairs" / f"task0--ep{episode}"
        directory.mkdir(parents=True)
        freshness = {
            "source_metadata_alarm": True,
            "relabelled_metadata_alarm": False,
            "exact_duplicate_alarm": True,
            "selected_gate_alarm": True,
        }
        clean_freshness = {**freshness, "source_metadata_alarm": False}
        clean_freshness["exact_duplicate_alarm"] = False
        clean_freshness["selected_gate_alarm"] = False
        value = {
            "status": "complete",
            "pair_condition": "stale_main_camera",
            "task_id": 0,
            "episode_index": episode,
            "branches": {
                "current": {
                    "success": current_success,
                    "intervention": {"freshness": clean_freshness},
                },
                "stale": {
                    "success": stale_success,
                    "intervention": {"freshness": freshness},
                },
            },
        }
        (directory / "pair.complete.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    def test_analysis_separates_policy_effect_freshness_and_safe_misses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._pair(root, 0, current_success=True, stale_success=False)
            self._pair(root, 1, current_success=True, stale_success=True)
            score_path = root / "scores.json"
            score_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "task_id": 0,
                                "episode_index": episode,
                                "label": label,
                                "alarm": _alarm(False),
                            }
                            for episode in range(2)
                            for label in ("current", "stale")
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = analyze(root, score_path)

        stale = result["stale"]
        self.assertEqual(stale["outcomes"]["stale_only_failure"], 1)
        self.assertEqual(
            stale["freshness_at_intervention"][
                "stale_exact_duplicate_detection_rate"
            ]["estimate"],
            1.0,
        )
        self.assertEqual(
            stale["safe_mlp"]["causal_stale_failures"][
                "silent_cofailure_rate"
            ]["estimate"],
            1.0,
        )
        self.assertFalse(
            result["interpretation_boundary"]["freshness"].startswith("response")
        )


if __name__ == "__main__":
    unittest.main()
