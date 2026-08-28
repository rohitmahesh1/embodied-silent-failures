import unittest

from embodied_silent_failures.language_worker import _select_terminal_groups


class LanguageWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = [{"command_id": "a"}, {"command_id": "b"}]

    def test_failed_control_keeps_groups_but_runs_no_fault_branches(self) -> None:
        selected, reason = _select_terminal_groups(
            self.groups, {"status": "complete", "success": False}, None
        )

        self.assertEqual(selected, [])
        self.assertEqual(reason, "control_failed")

    def test_unresolved_control_runs_no_fault_branches(self) -> None:
        selected, reason = _select_terminal_groups(
            self.groups, {"status": "unresolved"}, None
        )

        self.assertEqual(selected, [])
        self.assertEqual(reason, "control_unresolved")

    def test_branch_limit_applies_to_unique_commands(self) -> None:
        selected, reason = _select_terminal_groups(
            self.groups, {"status": "complete", "success": True}, 1
        )

        self.assertEqual(selected, [self.groups[0]])
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
