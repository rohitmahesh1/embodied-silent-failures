import unittest

from embodied_silent_failures.language_state_temporal import (
    fit_distance_scales,
    nearest_context_probabilities,
    temporal_margin_features,
)


def branch(
    name: str,
    *,
    state: list[float],
    command: list[float],
    failure: bool,
    episode: int,
    context: str,
) -> dict:
    return {
        "physical_run": name,
        "task_id": 0,
        "episode_index": episode,
        "context_id": context,
        "state": state,
        "delta_command": command,
        "executed_command": command,
        "task_failure": failure,
    }


class LanguageStateTemporalTests(unittest.TestCase):
    def test_temporal_features_use_existing_half_open_alarm_windows(self) -> None:
        result = temporal_margin_features(
            {
                "scores": [0.0, 2.0, 8.0, 4.0, 9.0, 1.0] + [1.0] * 30,
                "band": [10.0] * 36,
                "fault_step": 1,
            }
        )

        self.assertAlmostEqual(result["monitor_margin_0"], 0.8)
        self.assertAlmostEqual(result["monitor_margin_5"], 0.1)
        self.assertAlmostEqual(result["monitor_margin_10"], 0.1)
        self.assertAlmostEqual(result["monitor_margin_25"], 0.1)

    def test_state_scales_count_each_restored_context_once(self) -> None:
        rows = [
            branch(
                "a", state=[0.0], command=[0.0], failure=False, episode=0, context="c0"
            ),
            branch(
                "b", state=[0.0], command=[1.0], failure=False, episode=0, context="c0"
            ),
            branch(
                "c", state=[2.0], command=[2.0], failure=True, episode=1, context="c1"
            ),
        ]

        scales = fit_distance_scales(rows)

        self.assertEqual(scales[0]["state"], [1.0])

    def test_nearest_neighbors_give_one_vote_per_context(self) -> None:
        training = [
            branch(
                "a", state=[0.0], command=[0.0], failure=False, episode=0, context="c0"
            ),
            branch(
                "b", state=[0.0], command=[0.1], failure=False, episode=0, context="c0"
            ),
            branch(
                "c", state=[10.0], command=[0.0], failure=True, episode=1, context="c1"
            ),
        ]
        target = branch(
            "target",
            state=[9.0],
            command=[0.0],
            failure=True,
            episode=2,
            context="target",
        )

        prediction = nearest_context_probabilities(
            training,
            [target],
            mode="state",
            neighbors=1,
        )

        self.assertEqual(prediction, [1.0])

    def test_leave_trajectory_out_excludes_every_phase_of_target_episode(self) -> None:
        rows = [
            branch(
                "target", state=[0.0], command=[0.0], failure=True, episode=0, context="c0"
            ),
            branch(
                "same-trajectory",
                state=[0.0],
                command=[0.0],
                failure=True,
                episode=0,
                context="c1",
            ),
            branch(
                "other", state=[1.0], command=[1.0], failure=False, episode=1, context="c2"
            ),
        ]

        prediction = nearest_context_probabilities(
            rows,
            [rows[0]],
            mode="state_and_executed_command",
            neighbors=1,
            leave_target_trajectory_out=True,
        )

        self.assertEqual(prediction, [0.0])


if __name__ == "__main__":
    unittest.main()
