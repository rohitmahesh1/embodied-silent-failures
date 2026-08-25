import unittest

from embodied_silent_failures.evidence_graph.census import build_site_census


def _region(region_id, name, interface=None, semantic_key=None, disposition=None):
    return {
        "region_id": region_id,
        "name": name,
        "semantic_key": semantic_key or name,
        "event_ids": [f"event-{region_id}"],
        "event_count": 1,
        "fault_interface": interface,
        "reachable_sinks": ["monitor", "outcome"],
        "basis": ["observed:test"],
        "disposition": disposition,
    }


def _event(region_id, shape=(2, 3)):
    return {
        "event_id": f"event-{region_id}",
        "kind": "boundary",
        "name": region_id,
        "outputs": [
            {
                "port": "value",
                "type": "torch.Tensor",
                "shape": list(shape),
                "dtype": "torch.float32",
                "device": "cuda:0",
            }
        ],
    }


class SiteCensusTests(unittest.TestCase):
    def test_topology_uses_non_temporal_reachability(self) -> None:
        graph = {
            "regions": [
                _region(
                    "shared",
                    "backbone",
                    "registered_model_state",
                    "backbone/state/policy.layers.2.self_attn.q_proj",
                ),
                _region("action-path", "decode", "action_tokens"),
                _region("action", "decode", "raw_action"),
                _region("monitor-path", "feature", "final_layer_action_features"),
                _region("monitor-evidence", "feature", "safe_feature"),
                _region("future", "environment", "environment_observation"),
            ],
            "edges": [
                {"source": "shared", "target": "action-path", "kind": "dataflow"},
                {"source": "action-path", "target": "action", "kind": "dataflow"},
                {"source": "shared", "target": "monitor-path", "kind": "dataflow"},
                {
                    "source": "monitor-path",
                    "target": "monitor-evidence",
                    "kind": "dataflow",
                },
                {
                    "source": "future",
                    "target": "monitor-evidence",
                    "kind": "temporal_world_feedback",
                },
            ],
            "sinks": [
                {"event_id": "monitor", "name": "rollout.monitor_timeline"},
                {"event_id": "outcome", "name": "rollout.outcome"},
            ],
        }
        events = [_event(region["region_id"]) for region in graph["regions"]]

        census = build_site_census(graph, events)
        sites = {site["site_id"]: site for site in census["sites"]}

        self.assertEqual(
            sites["shared"]["topology"], "shared_action_and_monitor_evidence"
        )
        self.assertEqual(sites["action-path"]["topology"], "action_only")
        self.assertEqual(
            sites["monitor-path"]["topology"], "monitor_evidence_only"
        )
        self.assertEqual(
            sites["future"]["topology"], "neither_same_decision_path"
        )
        self.assertEqual(
            sites["shared"]["architecture"]["depth"],
            {
                "family": "policy.layers",
                "index": 2,
                "observed_maximum_index": 2,
                "normalized": 1.0,
            },
        )
        self.assertEqual(
            sites["shared"]["observed_value_schema"]["outputs"][0]["shape"],
            [2, 3],
        )

    def test_unknown_anchor_is_rejected(self) -> None:
        graph = {
            "regions": [_region("action", "decode", "raw_action")],
            "edges": [],
            "sinks": [],
        }
        with self.assertRaisesRegex(ValueError, "safe_feature"):
            build_site_census(graph, [_event("action")])


if __name__ == "__main__":
    unittest.main()
