import csv
import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.analyze_freshness import (
    _exact_binomial_two_sided,
    analyze,
)


class AnalyzeFreshnessTests(unittest.TestCase):
    def _write_trial(
        self,
        directory: Path,
        *,
        episode: int,
        condition: str,
        success: bool,
        alarm: bool,
        response: bool,
    ) -> None:
        csv_name = f"task0--ep{episode}.csv"
        with (directory / csv_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "action/timestep",
                    "freshness/source_metadata_alarm",
                    "freshness/relabelled_metadata_alarm",
                    "freshness/exact_duplicate_alarm",
                    "freshness/selected_gate_alarm",
                    "freshness/response_applied",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "action/timestep": 4,
                    "freshness/source_metadata_alarm": alarm,
                    "freshness/relabelled_metadata_alarm": False,
                    "freshness/exact_duplicate_alarm": alarm,
                    "freshness/selected_gate_alarm": alarm,
                    "freshness/response_applied": response,
                }
            )
        marker = {
            "status": "complete",
            "condition": condition,
            "task_id": 0,
            "episode_index": episode,
            "success": success,
            "fault": {"policy_step": 4},
            "files": {"csv": csv_name},
        }
        (directory / f"task0--ep{episode}.complete.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )

    def test_summarizes_paired_outcomes_detection_and_clean_alarms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "stale"
            current = root / "current"
            stale.mkdir()
            current.mkdir()
            self._write_trial(
                stale,
                episode=0,
                condition="stale_image",
                success=False,
                alarm=True,
                response=True,
            )
            self._write_trial(
                current,
                episode=0,
                condition="current_image_control",
                success=True,
                alarm=False,
                response=False,
            )
            self._write_trial(
                stale,
                episode=1,
                condition="stale_image",
                success=True,
                alarm=True,
                response=True,
            )
            self._write_trial(
                current,
                episode=1,
                condition="current_image_control",
                success=True,
                alarm=False,
                response=False,
            )

            result = analyze(stale, current)

            self.assertEqual(result["paired_trials"], 2)
            self.assertEqual(
                result["paired_task_outcomes"]["stale_only_failure"], 1
            )
            self.assertEqual(
                result["stale_intervention_detection"]["exact_duplicate"], 2
            )
            self.assertEqual(
                result["current_control_false_alarms"]["exact_duplicate_alarms"],
                0,
            )

    def test_exact_two_sided_binomial_matches_mcnemar_small_case(self) -> None:
        self.assertEqual(_exact_binomial_two_sided(3, 0), 0.25)
        self.assertIsNone(_exact_binomial_two_sided(0, 0))


if __name__ == "__main__":
    unittest.main()
