import unittest

from embodied_silent_failures.temporal_campaign import (
    depth_band,
    output_family,
    sample_sites,
    validate_campaign_manifest,
)


def _site(site_id, topology, owner, port="value", depth=None, kind="module_output"):
    identity = {"kind": kind, "output_port": port}
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
        "architecture": {
            "observed_owners": [owner],
            "literal_module_role": site_id,
            "depth": depth,
        },
        "value_families": ["continuous_tensor"],
        "schemas": [],
        "same_value_alias_site_ids": [],
    }


class TemporalCampaignTests(unittest.TestCase):
    def test_shared_sampling_is_uniform_within_mechanical_strata(self) -> None:
        shared = "shared_action_and_monitor_evidence"
        action = "action_only"
        table = {
            "sites": [
                _site("a", shared, "language"),
                _site("b", shared, "language"),
                _site("c", shared, "vision"),
                _site("d", shared, "vision"),
                _site("e", action, "command", kind="declared_runtime_boundary"),
            ]
        }

        selected, populations = sample_sites(
            table, seed=11, shared_per_stratum=1
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(sorted(populations.values()), [2, 2])
        census = next(item for item in selected if item["site"]["site_id"] == "e")
        self.assertEqual(census["selection"], "action_only_census")
        self.assertEqual(census["site_inclusion_probability"], 1.0)
        sampled = [item for item in selected if item is not census]
        self.assertTrue(
            all(item["site_inclusion_probability"] == 0.5 for item in sampled)
        )

    def test_depth_and_port_families_are_literal_transformations(self) -> None:
        site = _site(
            "hidden",
            "shared_action_and_monitor_evidence",
            "language",
            port="value.hidden_states[4]",
            depth={"normalized": 0.5},
        )
        self.assertEqual(depth_band(site), "middle")
        self.assertEqual(output_family(site), "returned_hidden_states")

    def test_manifest_validation_preserves_t_minus_one(self) -> None:
        manifest = {
            "schema_version": 1,
            "sites": [{"site_id": "site"}],
            "attempts": [
                {
                    "attempt_id": "a0000",
                    "site_id": "site",
                    "policy_step": 10,
                    "source_policy_step": 9,
                }
            ],
        }
        validate_campaign_manifest(manifest)
        manifest["attempts"][0]["source_policy_step"] = 8
        with self.assertRaises(ValueError):
            validate_campaign_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
