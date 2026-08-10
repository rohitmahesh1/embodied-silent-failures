import unittest

from embodied_silent_failures.plan import (
    Trial,
    build_trial_plan,
    parse_task_ids,
    seed_for_trial,
)


class PlanTests(unittest.TestCase):
    def test_parse_task_ids_accepts_values_and_ranges(self) -> None:
        self.assertEqual(parse_task_ids("0,2-4,7"), [0, 2, 3, 4, 7])

    def test_parse_task_ids_rejects_ambiguous_values(self) -> None:
        for value in ("", "0,,1", "3-1", "1,1", "-1", "a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_task_ids(value)

    def test_build_trial_plan_supports_episode_shards(self) -> None:
        self.assertEqual(
            build_trial_plan([0, 1], episode_start=1, episode_stop=6, episode_stride=2),
            [
                Trial(0, 1),
                Trial(0, 3),
                Trial(0, 5),
                Trial(1, 1),
                Trial(1, 3),
                Trial(1, 5),
            ],
        )

    def test_build_trial_plan_validates_bounds(self) -> None:
        invalid = [
            ([], 0, 1, 1),
            ([0], -1, 1, 1),
            ([0], 1, 1, 1),
            ([0], 0, 1, 0),
            ([-1], 0, 1, 1),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_trial_plan(*arguments)

    def test_trial_seeds_are_stable_and_distinguish_trials(self) -> None:
        trial = Trial(task_id=3, episode_index=17)
        self.assertEqual(seed_for_trial(7, trial), 3759677755)
        self.assertNotEqual(seed_for_trial(8, trial), seed_for_trial(7, trial))
        self.assertNotEqual(
            seed_for_trial(7, Trial(task_id=3, episode_index=18)),
            seed_for_trial(7, trial),
        )

        with self.assertRaises(ValueError):
            seed_for_trial(-1, trial)


if __name__ == "__main__":
    unittest.main()
