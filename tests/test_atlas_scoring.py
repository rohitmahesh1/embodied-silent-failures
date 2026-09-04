import unittest

import numpy as np

from embodied_silent_failures.atlas_scoring import (
    intervention_sources,
    reconstruct_cumulative_scores,
    replay_is_exact,
)


class AtlasScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = {
            "branch": "control",
            "result": {"status": "complete", "success": True},
        }
        self.group = {
            "command_id": "changed",
            "representative_site_id": "site-a",
            "member_site_ids": ["site-a", "site-b"],
        }

    @staticmethod
    def local(site_id: str, *, exact: bool) -> dict:
        return {
            "status": "complete",
            "site_id": site_id,
            "executed_command": {"exact_equal": exact},
        }

    def test_maps_equal_and_grouped_commands_to_physical_branches(self) -> None:
        command = {
            "branch": "command-changed",
            "command_group": self.group,
            "result": {"status": "complete", "success": False},
        }
        plans = intervention_sources(
            {
                "branches": [self.control, command],
                "command_groups": [self.group],
            },
            [
                self.local("site-control", exact=True),
                self.local("site-a", exact=False),
                self.local("site-b", exact=False),
            ],
        )

        by_site = {value["site_id"]: value for value in plans}
        self.assertEqual(by_site["site-control"]["physical_branch"], self.control)
        self.assertEqual(by_site["site-a"]["physical_branch"], command)
        self.assertEqual(by_site["site-b"]["physical_branch"], command)

    def test_keeps_changed_site_when_control_failure_prevented_branch(self) -> None:
        plans = intervention_sources(
            {
                "branches": [self.control],
                "command_groups": [self.group],
            },
            [self.local("site-a", exact=False)],
        )

        self.assertEqual(plans[0]["status"], "local_only")
        self.assertEqual(plans[0]["monitor_horizon"], "through_fault_step_only")

    def test_replaces_one_contribution_for_the_remaining_cumulative_trace(self) -> None:
        physical = np.asarray([0.2, 0.5, 0.9], dtype=np.float32)

        result = reconstruct_cumulative_scores(
            physical,
            fault_step=1,
            replacement_contribution=0.6,
            physical_contribution=0.3,
            np=np,
        )

        np.testing.assert_allclose(result, [0.2, 0.8, 1.2])

    def test_exact_replay_requires_state_numeric_and_image_equality(self) -> None:
        result = {
            "context_replay": {
                "simulator_state_exact_equal": True,
                "observation": {
                    "maximum_numeric_error": 0.0,
                    "maximum_image_channel_error": 0.0,
                    "changed_image_channels": 0,
                },
            }
        }

        self.assertTrue(replay_is_exact(result))
        result["context_replay"]["observation"]["changed_image_channels"] = 1
        self.assertFalse(replay_is_exact(result))


if __name__ == "__main__":
    unittest.main()
