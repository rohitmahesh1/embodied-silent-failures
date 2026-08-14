import unittest

from embodied_silent_failures.evidence_graph.audit import audit_graph
from embodied_silent_failures.evidence_graph.qwen_rollout import (
    QWEN_EVIDENCE_ENDPOINTS,
    ROLLOUT_ENDPOINTS,
    compose_qwen_rollout,
)
from embodied_silent_failures.evidence_graph.reduce import reduce_graph


class QwenRolloutGraphTests(unittest.TestCase):
    def test_qwen_replaces_safe_on_the_same_rollout_and_outcome(self) -> None:
        events = [
            {"event_id": "e00000000", "kind": "trace_start", "name": "trace"},
            {
                "event_id": "e00000001",
                "kind": "source",
                "name": "libero.current_observation",
                "context": {"policy_step": 0},
                "outputs": [
                    {
                        "port": "value.agentview_image",
                        "value_id": "v00000000",
                        "type": "numpy.ndarray",
                        "shape": [256, 256, 3],
                        "dtype": "uint8",
                    }
                ],
            },
            {
                "event_id": "e00000002",
                "kind": "boundary",
                "name": "policy.output",
                "context": {"policy_step": 0},
                "inputs": [
                    {
                        "port": "value",
                        "value_id": "v00000000",
                        "type": "numpy.ndarray",
                    }
                ],
                "outputs": [
                    {
                        "port": "value",
                        "value_id": "v00000001",
                        "type": "builtins.float",
                        "value": 0.5,
                    }
                ],
            },
            {
                "event_id": "e00000003",
                "kind": "boundary",
                "name": "safe.monitor_input",
                "inputs": [
                    {
                        "port": "value",
                        "value_id": "v00000001",
                        "type": "builtins.float",
                    }
                ],
            },
            {
                "event_id": "e00000004",
                "kind": "boundary",
                "name": "rollout.monitor_timeline",
                "inputs": [
                    {
                        "port": "value",
                        "value_id": "v00000001",
                        "type": "builtins.float",
                    }
                ],
            },
            {
                "event_id": "e00000005",
                "kind": "boundary",
                "name": "rollout.fault",
                "outputs": [
                    {
                        "port": "value",
                        "value_id": "v00000002",
                        "type": "builtins.bool",
                        "value": False,
                    }
                ],
            },
            {
                "event_id": "e00000006",
                "kind": "boundary",
                "name": "rollout.outcome",
                "inputs": [
                    {
                        "port": "value",
                        "value_id": "v00000001",
                        "type": "builtins.float",
                    }
                ],
            },
            {
                "event_id": "e00000007",
                "kind": "trace_end",
                "name": "trace",
                "details": {"completed": True},
            },
        ]
        annotations = [
            {
                "event_id": "e00000001",
                "region": "environment",
                "basis": ["observed:test-camera"],
                "lifetime": "step",
                "fault_interface": "environment_observation",
            },
            {
                "event_id": "e00000002",
                "region": "policy",
                "basis": ["observed:test-policy"],
                "lifetime": "step",
            },
            {
                "event_id": "e00000003",
                "region": "safe_feature",
                "basis": ["observed:test-safe"],
                "lifetime": "step",
            },
            {
                "event_id": "e00000004",
                "region": "monitor_timeline",
                "basis": ["observed:test-safe"],
                "lifetime": "temporal",
                "role": "sink",
            },
            {
                "event_id": "e00000005",
                "region": "fault",
                "basis": ["observed:test-fault"],
                "lifetime": "temporal",
                "role": "fault",
                "disposition": "not_applicable_clean_rollout",
            },
            {
                "event_id": "e00000006",
                "region": "task_outcome",
                "basis": ["observed:test-outcome"],
                "lifetime": "temporal",
                "role": "sink",
            },
        ]
        trial = {
            "task_description": "put the cup away",
            "video_sha256": "a" * 64,
            "video_metadata": {"height": 224, "width": 224},
            "timeline": [
                {
                    "policy_step": 0,
                    "frame_steps": [0],
                    "frame_sha256": ["b" * 64],
                    "prompt_sha256": "c" * 64,
                    "raw_response": '{"failure_now":0,"reason":"on track"}',
                    "parsed_response": {"failure_now": 0, "reason": "on track"},
                    "parse_error": None,
                    "alarm": False,
                }
            ],
        }
        qwen_run = {
            "configuration": {
                "model": {"revision": "d" * 40},
                "protocol": {"history_frames": 8},
            },
            "runtime": {
                "model_implementation": {
                    "class": "transformers.QwenForConditionalGeneration",
                    "sha256": "e" * 64,
                }
            },
        }

        composed, composed_annotations = compose_qwen_rollout(
            events,
            annotations,
            trial=trial,
            qwen_run=qwen_run,
            source_revision="f" * 40,
        )
        graph = reduce_graph(composed, composed_annotations)
        audit = audit_graph(
            composed,
            composed_annotations,
            graph,
            required_endpoints=ROLLOUT_ENDPOINTS,
            repeated_endpoints=QWEN_EVIDENCE_ENDPOINTS,
        )

        self.assertTrue(audit["passed"], audit)
        self.assertFalse(any(event["name"].startswith("safe.") for event in composed))
        self.assertEqual(
            {sink["name"] for sink in graph["sinks"]},
            {"rollout.monitor_timeline", "rollout.outcome"},
        )
        sink_names = {sink["event_id"]: sink["name"] for sink in graph["sinks"]}
        by_name = {region["name"]: region for region in graph["regions"]}
        self.assertEqual(
            {sink_names[item] for item in by_name["environment"]["reachable_sinks"]},
            {"rollout.monitor_timeline", "rollout.outcome"},
        )
        self.assertEqual(
            {sink_names[item] for item in by_name["policy"]["reachable_sinks"]},
            {"rollout.outcome"},
        )
        self.assertEqual(
            {sink_names[item] for item in by_name["qwen_private_compute"]["reachable_sinks"]},
            {"rollout.monitor_timeline"},
        )


if __name__ == "__main__":
    unittest.main()
