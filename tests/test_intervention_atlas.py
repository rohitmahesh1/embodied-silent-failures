import unittest

from embodied_silent_failures.intervention_atlas import (
    atlas_stratum,
    build_atlas_contexts,
    intervention_rule,
    sample_atlas_sites,
    topology_label,
    validate_intervention_atlas_manifest,
)


def _site(site_id, topology, owner, *, kind="module_output", depth=None):
    identity = {"kind": kind, "output_port": "value"}
    if kind == "module_output":
        identity.update(
            {
                "module_path": f"policy.{owner}.{site_id}",
                "module_call_index": 0,
            }
        )
    else:
        identity.update({"event_name": site_id, "event_call_index": 0})
    return {
        "site_id": site_id,
        "status": "structurally_eligible_pending_canary",
        "identity": identity,
        "topologies": [topology],
        "architecture": {"observed_owners": [owner], "depth": depth},
        "value_families": ["continuous_tensor"],
        "schemas": [],
        "same_value_alias_site_ids": [],
    }


class InterventionAtlasTests(unittest.TestCase):
    def test_sampling_uses_every_graph_derived_stratum(self) -> None:
        shared = "shared_action_and_monitor_evidence"
        action = "action_only"
        monitor = "monitor_evidence_only"
        sites = [
            _site("s0", shared, "language"),
            _site("s1", shared, "language"),
            _site("s2", shared, "language"),
            _site("a0", action, "action"),
            _site("m0", monitor, "monitor", kind="declared_runtime_boundary"),
        ]

        selected, populations = sample_atlas_sites(
            {"sites": sites}, seed=7, sites_per_stratum=2, census_below=1
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(sorted(populations.values()), [1, 1, 3])
        by_topology = {topology_label(site) for site in selected}
        self.assertEqual(by_topology, {shared, action, monitor})
        sampled_shared = [site for site in selected if topology_label(site) == shared]
        self.assertTrue(
            all(site["sampling"]["site_inclusion_probability"] == 2 / 3 for site in sampled_shared)
        )
        self.assertEqual(
            next(site for site in selected if site["site_id"] == "m0")["sampling"]["method"],
            "complete_stratum_census",
        )

    def test_stratum_is_a_literal_composition_of_recorded_fields(self) -> None:
        site = _site(
            "s0",
            "shared_action_and_monitor_evidence",
            "language",
            depth={"normalized": 0.5},
        )
        self.assertEqual(
            atlas_stratum(site),
            "shared_action_and_monitor_evidence:module_output:language:middle:direct",
        )

    def test_language_sequence_rule_selects_only_the_final_position(self) -> None:
        site = _site("s0", "shared_action_and_monitor_evidence", "language")
        site["identity"]["module_path"] = "policy.language_model.model.layers.0"
        site["schemas"] = [{"shape": [1, 97, 4096]}]

        self.assertEqual(
            intervention_rule(site)["value_slice"], "final_sequence_position"
        )

    def test_contexts_keep_trajectory_split_and_worker_together(self) -> None:
        trajectories = [
            {
                "task_id": 0,
                "episode_index": 1,
                "trial_seed": 8,
                "initial_state_sha256": "a",
                "policy_steps": 20,
                "analysis_split": "development",
            },
            {
                "task_id": 1,
                "episode_index": 2,
                "trial_seed": 9,
                "initial_state_sha256": "b",
                "policy_steps": 40,
                "analysis_split": "holdout",
            },
        ]

        contexts = build_atlas_contexts(trajectories, worker_count=2)

        self.assertEqual(len(contexts), 6)
        for trajectory in trajectories:
            matching = [
                context
                for context in contexts
                if context["task_id"] == trajectory["task_id"]
            ]
            self.assertEqual(
                {value["analysis_split"] for value in matching},
                {trajectory["analysis_split"]},
            )
            self.assertEqual(len({value["worker_shard"] for value in matching}), 1)

    def test_validation_keeps_errors_as_outcomes_but_checks_design_identity(self) -> None:
        site = _site("s0", "shared_action_and_monitor_evidence", "language")
        site["sampling"] = {"site_inclusion_probability": 1.0}
        manifest = {
            "schema_version": 1,
            "sites": [site],
            "contexts": [
                {
                    "context_id": "c0000",
                    "task_id": 0,
                    "episode_index": 1,
                    "analysis_split": "holdout",
                    "worker_shard": 0,
                    "policy_step": 2,
                    "source_policy_step": 1,
                },
                {
                    "context_id": "c0001",
                    "task_id": 0,
                    "episode_index": 1,
                    "analysis_split": "holdout",
                    "worker_shard": 0,
                    "policy_step": 3,
                    "source_policy_step": 2,
                },
                {
                    "context_id": "c0002",
                    "task_id": 0,
                    "episode_index": 1,
                    "analysis_split": "holdout",
                    "worker_shard": 0,
                    "policy_step": 4,
                    "source_policy_step": 3,
                },
            ],
            "clean_trajectories": [{"task_id": 0, "episode_index": 1}],
            "counts": {"selected_sites": 1, "contexts": 3},
        }

        validate_intervention_atlas_manifest(manifest)
        manifest["contexts"][1]["worker_shard"] = 1
        with self.assertRaisesRegex(ValueError, "split across atlas workers"):
            validate_intervention_atlas_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
