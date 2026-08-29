import unittest

from embodied_silent_failures.language_campaign import (
    CACHE_AWARE_BOUNDARY_STATE,
    CACHE_REPLAY_STORAGE,
    build_contexts,
    language_block_sites,
    select_clean_trajectories,
    validate_language_campaign_manifest,
)


def _table():
    return {
        "sites": [
            {
                "site_id": f"site-{layer}-{token}",
                "status": "structurally_eligible_pending_canary",
                "identity": {
                    "kind": "module_output",
                    "module_path": f"policy.language_model.model.layers.{layer}",
                    "module_call_index": token,
                    "output_port": "value[0]",
                },
                "schemas": [{"shape": [1, 1 if token else 285, 4096]}],
                "topologies": ["shared_action_and_monitor_evidence"],
            }
            for layer in range(32)
            for token in range(7)
        ]
    }


def _clean_frame():
    return [
        {
            "task_id": task,
            "episode_index": episode,
            "policy_steps": 100 + episode,
            "trial_seed": task * 100 + episode,
            "initial_state_sha256": f"state-{task}-{episode}",
            "source": {},
        }
        for task in range(10)
        for episode in range(8)
    ]


class LanguageCampaignTests(unittest.TestCase):
    def _manifest(self, instrumentation):
        trajectories = select_clean_trajectories(_clean_frame(), seed=31)
        contexts = build_contexts(trajectories, seed=31)
        return {
            "schema_version": 1,
            "sites": language_block_sites(_table()),
            "contexts": contexts,
            "clean_trajectories": trajectories,
            "counts": {
                "contexts": len(contexts),
                "trajectories_per_task": 5,
                "development_trajectories_per_task": 3,
            },
            "excluded_prior_trajectories": [],
            "excluded_prior_trajectory_count": 0,
            "instrumentation": instrumentation,
        }

    def test_sites_are_a_complete_block_by_token_census(self) -> None:
        sites = language_block_sites(_table())

        self.assertEqual(len(sites), 224)
        self.assertEqual(sites[0]["layer_index"], 0)
        self.assertEqual(sites[-1]["layer_index"], 31)
        self.assertEqual(sites[-1]["action_token_position"], 6)

    def test_incomplete_site_table_is_rejected(self) -> None:
        table = _table()
        table["sites"].pop()

        with self.assertRaises(ValueError):
            language_block_sites(table)

    def test_contexts_preserve_trajectories_and_balance_tokens(self) -> None:
        trajectories = select_clean_trajectories(_clean_frame(), seed=31)
        contexts = build_contexts(trajectories, seed=31)

        self.assertEqual(len(trajectories), 50)
        self.assertEqual(len(contexts), 150)
        counts = {
            token: sum(
                context["action_token_position"] == token for context in contexts
            )
            for token in range(7)
        }
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for task in range(10):
            task_contexts = [value for value in contexts if value["task_id"] == task]
            self.assertEqual(
                sum(value["analysis_split"] == "development" for value in task_contexts),
                9,
            )
            self.assertEqual(
                sum(value["analysis_split"] == "holdout" for value in task_contexts),
                6,
            )
        by_trajectory = {}
        for context in contexts:
            key = (context["task_id"], context["episode_index"])
            by_trajectory.setdefault(key, set()).add(context["analysis_split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_trajectory.values()))
        worker_by_trajectory = {}
        for context in contexts:
            key = (context["task_id"], context["episode_index"])
            worker_by_trajectory.setdefault(key, set()).add(context["worker_shard"])
        self.assertTrue(
            all(len(workers) == 1 for workers in worker_by_trajectory.values())
        )

    def test_prior_trajectories_can_be_excluded_before_balanced_sampling(self) -> None:
        excluded = {(task, episode) for task in range(10) for episode in range(5)}

        trajectories = select_clean_trajectories(
            _clean_frame(),
            seed=41,
            trajectories_per_task=3,
            development_trajectories_per_task=2,
            excluded_trajectories=excluded,
        )
        contexts = build_contexts(trajectories, seed=41)

        self.assertEqual(len(trajectories), 30)
        self.assertEqual(len(contexts), 90)
        self.assertFalse(
            excluded
            & {
                (int(value["task_id"]), int(value["episode_index"]))
                for value in trajectories
            }
        )
        for task in range(10):
            task_values = [value for value in trajectories if value["task_id"] == task]
            self.assertEqual(
                sum(value["analysis_split"] == "development" for value in task_values),
                2,
            )
            self.assertEqual(
                sum(value["analysis_split"] == "holdout" for value in task_values),
                1,
            )

    def test_full_interface_manifest_requires_exact_cache_replay(self) -> None:
        instrumentation = {
            "full_language_interfaces": True,
            "language_ports": [
                "exact post-rotary current-token key cache entry",
                "exact current-token value cache entry",
            ],
            "boundary_state": CACHE_AWARE_BOUNDARY_STATE,
            "boundary_replay_storage": CACHE_REPLAY_STORAGE,
        }

        validate_language_campaign_manifest(self._manifest(instrumentation))
        incomplete = {**instrumentation, "language_ports": []}
        with self.assertRaises(ValueError):
            validate_language_campaign_manifest(self._manifest(incomplete))


if __name__ == "__main__":
    unittest.main()
