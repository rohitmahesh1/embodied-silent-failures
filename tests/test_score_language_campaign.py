import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from embodied_silent_failures.language_scoring import (
    composition_check,
    intervention_sources,
)
from embodied_silent_failures.score_language_campaign import _same_monitor


def local_record(layer_index: int, *, exact: bool) -> dict:
    return {
        "status": "complete",
        "layer_index": layer_index,
        "executed_command": {"exact_equal": exact},
    }


class LanguageCampaignScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = {
            "branch": "control",
            "result": {"status": "complete", "success": True},
        }
        self.group = {
            "command_id": "changed",
            "representative_layer_index": 1,
            "member_layer_indices": [1, 2],
        }

    def test_maps_equal_commands_to_control_and_group_members_to_one_branch(self) -> None:
        command = {
            "branch": "command-changed",
            "command_group": self.group,
            "result": {"status": "complete", "success": False},
        }
        summary = {
            "branches": [self.control, command],
            "command_groups": [self.group],
        }

        plans = intervention_sources(
            summary,
            [
                local_record(0, exact=True),
                local_record(1, exact=False),
                local_record(2, exact=False),
            ],
        )

        self.assertEqual(plans[0]["physical_branch"], self.control)
        self.assertEqual(plans[0]["terminal_evidence"], "inherited_from_exact_command_control")
        self.assertEqual(plans[1]["physical_branch"], command)
        self.assertEqual(plans[2]["physical_branch"], command)

    def test_changed_command_without_branch_is_scored_only_through_fault(self) -> None:
        summary = {
            "branches": [
                {
                    "branch": "control",
                    "result": {"status": "complete", "success": False},
                }
            ],
            "command_groups": [self.group],
        }

        plan = intervention_sources(summary, [local_record(1, exact=False)])[0]

        self.assertIsNone(plan["terminal_result"])
        self.assertEqual(plan["monitor_horizon"], "through_fault_step_only")

    def test_composition_requires_score_and_alarm_equivalence(self) -> None:
        band = np.asarray([0.5, 0.5], dtype=np.float32)
        physical = np.asarray([0.1, 0.6], dtype=np.float32)

        valid = composition_check(physical.copy(), physical, band, np)
        changed_alarm = composition_check(
            np.asarray([0.1, 0.49], dtype=np.float32), physical, band, np
        )

        self.assertTrue(valid["valid"])
        self.assertTrue(valid["score_exact_equal"])
        self.assertFalse(changed_alarm["valid"])
        self.assertFalse(changed_alarm["alarm_timeline_exact_equal"])

    def test_physical_scores_must_use_the_frozen_score_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "clean_scores.npz"
            archive.write_bytes(b"frozen scores")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            monitor = {
                "checkpoint": {"sha256": "checkpoint"},
                "configuration": {"sha256": "configuration"},
                "split_manifest": {"sha256": "split"},
            }
            physical = {
                "checkpoint_sha256": "checkpoint",
                "configuration_sha256": "configuration",
                "split_manifest_sha256": "split",
                "clean_score_archive_sha256": digest,
            }

            self.assertTrue(_same_monitor(physical, monitor, archive))
            physical["clean_score_archive_sha256"] = "different"
            self.assertFalse(_same_monitor(physical, monitor, archive))


if __name__ == "__main__":
    unittest.main()
