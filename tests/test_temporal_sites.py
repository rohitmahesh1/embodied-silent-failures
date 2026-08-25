import unittest

from embodied_silent_failures.evidence_graph.temporal_sites import (
    build_temporal_site_table,
    csv_rows,
)


def _tensor(value_id, shape=(1, 4), port="value"):
    return {
        "port": port,
        "value_id": value_id,
        "storage_id": f"storage-{value_id}",
        "type": "torch.Tensor",
        "shape": list(shape),
        "dtype": "torch.float32",
        "device": "cuda:0",
    }


def _module(event_id, step, path, outputs, *, call_index=0, inputs=None):
    event = {
        "event_id": event_id,
        "kind": "module",
        "name": f"module.{path}",
        "context": {"policy_step": step, "phase": "policy"},
        "outputs": outputs,
        "details": {
            "module_path": path,
            "module_call_index": call_index,
            "module_calls": [{"path": path, "call_index": call_index}],
        },
    }
    if inputs:
        event["inputs"] = inputs
    return event


def _boundary(event_id, step, name, *, inputs=None, outputs=None):
    event = {
        "event_id": event_id,
        "kind": "boundary",
        "name": name,
        "context": {"policy_step": step},
    }
    if inputs:
        event["inputs"] = inputs
    if outputs:
        event["outputs"] = outputs
    return event


def _source():
    events = [
        {
            "event_id": "start",
            "kind": "trace_start",
            "name": "trace",
            "details": {
                "condition": "clean",
                "episode_index": 2,
                "task_id": 1,
                "traced_steps": [0, 1],
            },
        }
    ]
    candidate_events = []
    action_events = []
    monitor_events = []
    camera_events = []
    for step in (0, 1):
        used = _tensor(f"used-{step}", port="value.used")
        unused = _tensor(f"unused-{step}", port="value.unused")
        shared = _module(
            f"shared-{step}",
            step,
            "policy.language_model.model.layers.3.mlp",
            [used, unused],
        )
        action_only_value = _tensor(f"action-only-{step}")
        action_only = _module(
            f"action-only-{step}",
            step,
            "policy.language_model.lm_head",
            [action_only_value],
            call_index=2,
        )
        mismatch_value = _tensor(
            f"mismatch-{step}", shape=(1, 4 + step)
        )
        mismatch = _module(
            f"mismatch-{step}",
            step,
            "policy.projector.linear",
            [mismatch_value],
        )
        camera_value = {
            "port": "value",
            "value_id": f"camera-{step}",
            "storage_id": f"storage-camera-{step}",
            "type": "numpy.ndarray",
            "shape": [224, 224, 3],
            "dtype": "uint8",
        }
        camera = _boundary(
            f"camera-{step}",
            step,
            "policy.selected_image",
            outputs=[camera_value],
        )
        command = _boundary(
            f"command-{step}",
            step,
            "libero.environment_step",
            inputs=[used, action_only_value, mismatch_value, camera_value],
            outputs=[_tensor(f"command-output-{step}")],
        )
        monitor = _boundary(
            f"monitor-{step}",
            step,
            "safe.monitor_input",
            inputs=[used],
            outputs=[_tensor(f"monitor-output-{step}")],
        )
        events.extend([shared, action_only, mismatch, camera, command, monitor])
        candidate_events.extend([shared, action_only, mismatch])
        camera_events.append(camera)
        action_events.append(command)
        monitor_events.append(monitor)

    sparse_value = _tensor("sparse-0")
    sparse = _module(
        "sparse-0",
        0,
        "policy.vision_backbone.blocks.1.attn",
        [sparse_value],
    )
    events.insert(5, sparse)
    events[6]["inputs"].append(sparse_value)
    candidate_events.append(sparse)
    events.append(
        {
            "event_id": "end",
            "kind": "trace_end",
            "name": "trace",
            "details": {"completed": True},
        }
    )

    regions = [
        {
            "region_id": "language",
            "name": "language_backbone",
            "event_ids": [
                event["event_id"]
                for event in candidate_events
                if "language_model" in event["details"]["module_path"]
            ],
            "basis": ["paper:openvla:test"],
        },
        {
            "region_id": "projector",
            "name": "multimodal_projector",
            "event_ids": ["mismatch-0", "mismatch-1"],
            "basis": ["paper:openvla:test"],
        },
        {
            "region_id": "vision",
            "name": "vision_encoder",
            "event_ids": ["sparse-0"],
            "basis": ["paper:openvla:test"],
        },
        {
            "region_id": "camera",
            "name": "policy_input_buffer",
            "event_ids": [event["event_id"] for event in camera_events],
            "fault_interface": "policy_image_buffer",
            "basis": ["protocol:test:camera"],
        },
        {
            "region_id": "command",
            "name": "environment",
            "event_ids": [event["event_id"] for event in action_events],
            "fault_interface": "simulator_command",
            "basis": ["code:test:command"],
        },
        {
            "region_id": "monitor",
            "name": "safe_feature",
            "event_ids": [event["event_id"] for event in monitor_events],
            "fault_interface": "safe_feature",
            "basis": ["paper:safe:test"],
        },
    ]
    raw_to_region = {
        event_id: region["region_id"]
        for region in regions
        for event_id in region["event_ids"]
    }
    return {
        "source_id": "test-source",
        "events": events,
        "graph": {
            "regions": regions,
            "raw_to_region": raw_to_region,
        },
    }


class TemporalSiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = build_temporal_site_table([_source()])
        self.by_path_port = {
            (
                site["identity"].get("module_path"),
                site["identity"]["output_port"],
            ): site
            for site in self.table["sites"]
            if site["identity"]["kind"] == "module_output"
        }

    def test_consecutive_shared_module_output_is_eligible(self) -> None:
        site = self.by_path_port[
            ("policy.language_model.model.layers.3.mlp", "value.used")
        ]
        self.assertEqual(site["status"], "structurally_eligible_pending_canary")
        self.assertEqual(site["eligible_opportunity_count"], 1)
        self.assertEqual(
            site["topologies"], ["shared_action_and_monitor_evidence"]
        )
        self.assertEqual(site["architecture"]["depth"]["index"], 3)

    def test_output_port_reachability_is_not_inferred_from_a_sibling(self) -> None:
        site = self.by_path_port[
            ("policy.language_model.model.layers.3.mlp", "value.unused")
        ]
        self.assertEqual(site["status"], "structurally_ineligible")
        self.assertIn(
            "output_port_does_not_reach_the_executed_command",
            site["eligibility_reasons"],
        )

    def test_action_only_topology_is_retained(self) -> None:
        site = self.by_path_port[("policy.language_model.lm_head", "value")]
        self.assertEqual(site["topologies"], ["action_only"])
        self.assertEqual(site["identity"]["module_call_index"], 2)
        self.assertEqual(site["eligible_opportunity_count"], 1)

    def test_schema_mismatch_is_ineligible(self) -> None:
        site = self.by_path_port[("policy.projector.linear", "value")]
        self.assertEqual(site["status"], "structurally_ineligible")
        self.assertEqual(site["consecutive_pair_count"], 1)
        self.assertEqual(site["matching_schema_pair_count"], 0)
        self.assertIn(
            "all_consecutive_observations_have_different_schemas",
            site["eligibility_reasons"],
        )

    def test_sparse_module_trace_remains_unresolved(self) -> None:
        site = self.by_path_port[
            ("policy.vision_backbone.blocks.1.attn", "value")
        ]
        self.assertEqual(site["status"], "unresolved_without_consecutive_trace")
        self.assertEqual(site["canary_status"], "not_run")

    def test_declared_camera_boundary_is_discovered(self) -> None:
        site = next(
            site
            for site in self.table["sites"]
            if site["identity"].get("event_name") == "policy.selected_image"
        )
        self.assertEqual(site["value_families"], ["image_array"])
        self.assertEqual(site["fault_interfaces"], ["policy_image_buffer"])
        self.assertEqual(site["eligible_opportunity_count"], 1)

    def test_action_anchor_outputs_are_not_action_sites(self) -> None:
        event_names = {
            site["identity"].get("event_name") for site in self.table["sites"]
        }
        self.assertNotIn("libero.environment_step", event_names)
        self.assertEqual(
            self.table["counts"]["excluded_events"][
                "action_anchor_is_an_input_interface"
            ],
            2,
        )

    def test_csv_has_one_flat_row_per_site(self) -> None:
        rows = csv_rows(self.table)
        self.assertEqual(len(rows), self.table["counts"]["sites"])
        self.assertTrue(all("eligibility_reasons" in row for row in rows))

    def test_duplicate_source_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate temporal-site source ID"):
            build_temporal_site_table([_source(), _source()])


if __name__ == "__main__":
    unittest.main()
