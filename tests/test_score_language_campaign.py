import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from embodied_silent_failures.language_scoring import (
    composition_check,
    composition_verified,
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

    def test_composition_allows_float32_scale_relative_roundoff(self) -> None:
        band = np.asarray([200.0], dtype=np.float32)
        physical = np.asarray([100.0], dtype=np.float32)
        reconstructed = np.asarray([100.00005], dtype=np.float32)

        check = composition_check(reconstructed, physical, band, np)

        self.assertTrue(check["valid"])
        self.assertFalse(check["score_exact_equal"])
        self.assertTrue(check["alarm_timeline_exact_equal"])
        self.assertEqual(check["absolute_tolerance"], 1e-6)
        self.assertEqual(check["relative_tolerance"], 1e-6)

    def test_unexecuted_command_group_is_not_composition_verified(self) -> None:
        record = {
            "context_id": "c000",
            "command_id": "changed",
            "terminal_evidence": "unavailable_without_successful_control",
        }

        self.assertFalse(
            composition_verified(record, {}, control_feature_exact=True)
        )

    def test_executed_command_group_requires_a_valid_representative(self) -> None:
        record = {
            "context_id": "c002",
            "command_id": "changed",
            "terminal_evidence": "observed_exact_command_branch",
        }
        key = ("c002", "changed")

        self.assertFalse(
            composition_verified(record, {}, control_feature_exact=True)
        )
        self.assertTrue(
            composition_verified(
                record, {key: {"valid": True}}, control_feature_exact=True
            )
        )

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
