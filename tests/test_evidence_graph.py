import gc
import tempfile
import unittest
from collections import UserDict
from pathlib import Path

from embodied_silent_failures.evidence_graph.audit import audit_graph
from embodied_silent_failures.evidence_graph.openvla import (
    contract_issues as openvla_contract_issues,
    operator_annotations as openvla_annotations,
    record_policy_image,
)
from embodied_silent_failures.evidence_graph.record import Recorder, read_events
from embodied_silent_failures.evidence_graph.reduce import reduce_graph
from embodied_silent_failures.evidence_graph.rollout import (
    RolloutEvidence,
    attach_monitor_timeline,
    prepare_evidence_output,
    structural_signature,
    summarize_saturation,
)
from embodied_silent_failures.evidence_graph.safe import (
    contract_issues as safe_contract_issues,
    operator_annotations as safe_annotations,
)
from embodied_silent_failures.evidence_graph.torch_trace import (
    contract_issues as torch_trace_contract_issues,
)


class Value:
    def __init__(self, name: str):
        self.name = name


class ArrayValue:
    def __init__(self, address: int):
        self.shape = (1,)
        self.__array_interface__ = {"data": (address, False)}


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2:
            return FakeTensor((self.shape[0], 7))
        if isinstance(index, tuple):
            return FakeTensor((self.shape[-1],))
        return FakeTensor(self.shape[1:])


class FakeArray:
    shape = (7,)

    def copy(self):
        return FakeArray()

    def tolist(self):
        return [0.0] * 7


class FakeTorch:
    @staticmethod
    def stack(values, dim=0):
        assert dim == 0
        return FakeTensor((len(values), *values[0].shape))


class FakeProcessorOutput:
    def to(self, *_args, **_kwargs):
        return {"input_ids": FakeTensor((1, 4))}


class FakeProcessor:
    def __call__(self, _prompt, _image):
        return FakeProcessorOutput()


class FakeModel:
    @staticmethod
    def get_action_dim(_unnorm_key):
        return 7


class FakeRuntime:
    @staticmethod
    def normalize_gripper_action(action, binarize=True):
        assert binarize
        return action

    @staticmethod
    def invert_gripper_action(action):
        return action


class FakeEnvironment:
    def __init__(self):
        self.next_observation = {"agentview_image": Value("next-image")}

    def step(self, command):
        assert len(command) == 7
        return self.next_observation, 1.0, True, {}


class EvidenceGraphTests(unittest.TestCase):
    def record_branching_graph(self, directory: Path):
        source = Value("source")
        shared = Value("shared")
        action = Value("action")
        score = Value("score")
        path = directory / "raw.jsonl"
        with Recorder(path, {"scope": "test"}) as recorder:
            recorder.source(
                "source",
                source,
                basis="observed:test-source",
                region="input",
            )
            recorder.mark(
                "shared",
                inputs=source,
                outputs=shared,
                basis="paper:test:shared",
                region="shared_compute",
            )
            recorder.mark(
                "action",
                inputs=shared,
                outputs=action,
                basis="code:test@abc:action:return-action",
                region="shared_compute",
                role="sink",
            )
            recorder.mark(
                "monitor",
                inputs=shared,
                outputs=score,
                basis="protocol:test:monitor",
                region="monitor",
                role="sink",
            )
        return recorder, read_events(path)

    def test_raw_trace_and_semantic_overlay_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, events = self.record_branching_graph(Path(directory))

        self.assertNotIn("semantic", events[1])
        self.assertEqual(recorder.annotations[0]["event_id"], events[1]["event_id"])
        self.assertEqual(recorder.annotations[0]["basis"], ["observed:test-source"])
        self.assertEqual(recorder.annotations[0]["lifetime"], "step")
        self.assertEqual(events[0]["kind"], "trace_start")
        self.assertEqual(events[-1]["kind"], "trace_end")

    def test_unlabeled_operator_waits_for_adapter_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            source = Value("source")
            output = Value("output")
            with Recorder(path, {"scope": "test"}) as recorder:
                recorder.source(
                    "source",
                    source,
                    basis="observed:test-source",
                    region="input",
                )
                event = recorder.mark(
                    "operator",
                    kind="operator",
                    inputs=source,
                    outputs=output,
                )

        self.assertNotIn(event["event_id"], {
            annotation["event_id"] for annotation in recorder.annotations
        })

    def test_reduction_rejects_duplicate_operator_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            with Recorder(path, {"scope": "test"}) as recorder:
                event = recorder.mark(
                    "operator",
                    kind="operator",
                    inputs=Value("input"),
                    outputs=Value("output"),
                )
        annotation = {
            "event_id": event["event_id"],
            "region": "compute",
            "basis": ["observed:test-operator"],
            "lifetime": "step",
        }

        with self.assertRaisesRegex(ValueError, "multiple annotations"):
            reduce_graph(recorder.events, [annotation, annotation])

    def test_recorder_traverses_mapping_containers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            left = Value("left")
            right = Value("right")
            with Recorder(path, {"scope": "test"}) as recorder:
                event = recorder.mark(
                    "mapping",
                    outputs=UserDict({"left": left, "nested": {"right": right}}),
                    basis="observed:test-mapping",
                    region="input",
                )

        self.assertEqual(
            [output["port"] for output in event["outputs"]],
            ["value.left", "value.nested.right"],
        )

    def test_storage_identity_distinguishes_reused_addresses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            with Recorder(path, {"scope": "test"}) as recorder:
                first = ArrayValue(1234)
                alias = ArrayValue(1234)
                first_event = recorder.source(
                    "first",
                    first,
                    basis="observed:test-first",
                    region="input",
                )
                alias_event = recorder.source(
                    "alias",
                    alias,
                    basis="observed:test-alias",
                    region="input",
                )
                first_storage = first_event["outputs"][0]["storage_id"]
                self.assertEqual(
                    alias_event["outputs"][0]["storage_id"], first_storage
                )
                del first
                del alias
                gc.collect()
                recycled = recorder.source(
                    "recycled",
                    ArrayValue(1234),
                    basis="observed:test-recycled",
                    region="input",
                )

        self.assertNotEqual(recycled["outputs"][0]["storage_id"], first_storage)

    def test_reduction_preserves_branch_reachability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, events = self.record_branching_graph(Path(directory))
        graph = reduce_graph(events, recorder.annotations)

        event_ids = {event["name"]: event["event_id"] for event in events}
        shared_reachability = graph["raw_reachability"][event_ids["shared"]]
        self.assertEqual(
            shared_reachability,
            sorted([event_ids["action"], event_ids["monitor"]]),
        )
        shared_regions = [
            region for region in graph["regions"] if region["name"] == "shared_compute"
        ]
        self.assertEqual(len(shared_regions), 2)
        self.assertNotEqual(
            graph["raw_to_region"][event_ids["shared"]],
            graph["raw_to_region"][event_ids["action"]],
        )

    def test_reduction_projects_paths_through_audit_only_operators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            source = Value("source")
            intermediate = Value("intermediate")
            output = Value("output")
            with Recorder(path, {"scope": "test"}) as recorder:
                source_event = recorder.source(
                    "source",
                    source,
                    basis="observed:test-source",
                    region="input",
                )
                recorder.mark(
                    "operator",
                    kind="operator",
                    inputs=source,
                    outputs=intermediate,
                )
                module_event = recorder.mark(
                    "module.consumer",
                    kind="module",
                    inputs=intermediate,
                    outputs=output,
                    basis="observed:test-module",
                    region="compute",
                    role="sink",
                )
        graph = reduce_graph(recorder.events, recorder.annotations)

        self.assertIn(
            {
                "source": source_event["event_id"],
                "target": module_event["event_id"],
                "kind": "dataflow",
            },
            graph["raw_edges"],
        )
        self.assertEqual(
            graph["raw_reachability"][source_event["event_id"]],
            [module_event["event_id"]],
        )

    def test_direct_dataflow_is_not_duplicated_as_storage_aliasing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            shared = ArrayValue(1234)
            output = ArrayValue(5678)
            with Recorder(path, {"scope": "test"}) as recorder:
                source = recorder.source(
                    "source",
                    shared,
                    basis="observed:test-source",
                    region="input",
                )
                sink = recorder.mark(
                    "sink",
                    inputs=shared,
                    outputs=output,
                    basis="observed:test-sink",
                    region="compute",
                    role="sink",
                )
        graph = reduce_graph(recorder.events, recorder.annotations)

        matching = [
            edge
            for edge in graph["raw_edges"]
            if edge["source"] == source["event_id"]
            and edge["target"] == sink["event_id"]
        ]
        self.assertEqual(matching, [
            {
                "source": source["event_id"],
                "target": sink["event_id"],
                "kind": "dataflow",
            }
        ])

    def test_complete_graph_passes_four_part_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, events = self.record_branching_graph(Path(directory))
        graph = reduce_graph(events, recorder.annotations)
        audit = audit_graph(
            events,
            recorder.annotations,
            graph,
            required_endpoints=("source", "action", "monitor"),
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(
            set(audit["sections"]),
            {
                "trace_integrity",
                "annotation_coverage",
                "reduction_integrity",
                "model_contracts",
                "provenance_format",
            },
        )
        self.assertIn("not_established", audit["trust_boundary"])

    def test_audit_localizes_missing_scope_annotation_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder, events = self.record_branching_graph(Path(directory))
        annotations = recorder.annotations[:-1]
        graph = reduce_graph(events, recorder.annotations)
        missing_event = next(event for event in events if event["name"] == "shared")
        del graph["raw_to_region"][missing_event["event_id"]]
        audit = audit_graph(
            events,
            annotations,
            graph,
            required_endpoints=("source", "missing"),
        )

        self.assertFalse(audit["passed"])
        self.assertFalse(audit["sections"]["trace_integrity"]["passed"])
        self.assertFalse(audit["sections"]["annotation_coverage"]["passed"])
        self.assertFalse(audit["sections"]["reduction_integrity"]["passed"])

    def test_audit_rejects_a_tensor_with_no_observed_origin(self) -> None:
        Tensor = type("Tensor", (), {"__module__": "torch"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            missing = Tensor()
            output = Value("output")
            with Recorder(path, {"scope": "test"}) as recorder:
                recorder.mark(
                    "operator",
                    kind="operator",
                    inputs=missing,
                    outputs=output,
                    basis="observed:test-operator",
                    region="compute",
                    role="sink",
                )
        graph = reduce_graph(recorder.events, recorder.annotations)
        audit = audit_graph(
            recorder.events,
            recorder.annotations,
            graph,
            required_endpoints=("operator",),
        )

        self.assertFalse(audit["passed"])
        unresolved = audit["sections"]["annotation_coverage"][
            "unproduced_tensor_inputs"
        ]
        self.assertEqual(unresolved[0]["name"], "operator")

    def test_audit_reports_pytorch_constructor_tensor_as_an_origin(self) -> None:
        Tensor = type("Tensor", (), {"__module__": "torch"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            missing = Tensor()
            output = Value("output")
            consumed = Value("consumed")
            with Recorder(path, {"scope": "test"}) as recorder:
                event = recorder.mark(
                    "aten.lift_fresh.default",
                    kind="operator",
                    inputs=missing,
                    outputs=output,
                    details={
                        "operator_semantics": {
                            "schema_status": "available",
                            "declared_aliases": [],
                            "mutated_input_ports": [],
                        }
                    },
                )
                module = recorder.mark(
                    "module.constructor",
                    kind="module",
                    inputs=output,
                    outputs=consumed,
                )
        annotations = [
            {
                "event_id": event["event_id"],
                "region": "tensor_constructor",
                "basis": ["observed:test-constructor"],
                "lifetime": "step",
            },
            {
                "event_id": module["event_id"],
                "region": "tensor_consumer",
                "basis": ["observed:test-consumer"],
                "lifetime": "step",
                "role": "sink",
            }
        ]
        graph = reduce_graph(recorder.events, annotations)
        audit = audit_graph(
            recorder.events,
            annotations,
            graph,
            required_endpoints=("module.constructor",),
        )

        self.assertTrue(audit["passed"])
        unresolved = audit["sections"]["annotation_coverage"]
        self.assertEqual(unresolved["unproduced_tensor_inputs"], [])
        self.assertEqual(unresolved["tensor_origins"][0]["name"], event["name"])

    def test_adapter_annotations_follow_runtime_scope(self) -> None:
        events = [
            {
                "event_id": "e1",
                "kind": "operator",
                "name": "aten.mm.default",
                "context": {"phase": "policy"},
                "details": {
                    "module_scope": ["policy", "policy.language_model.model.layers.3"]
                },
            },
            {
                "event_id": "e2",
                "kind": "operator",
                "name": "aten.linear.default",
                "context": {"phase": "monitor"},
                "details": {"module_scope": ["safe_monitor.projector.0"]},
            },
        ]

        policy = openvla_annotations(events)
        monitor = safe_annotations(events)
        self.assertEqual(policy[0]["region"], "language_backbone")
        self.assertTrue(any(item.startswith("paper:openvla") for item in policy[0]["basis"]))
        self.assertEqual(monitor[0]["region"], "safe_monitor")
        self.assertTrue(any(item.startswith("code:safe") for item in monitor[0]["basis"]))

    def test_model_contracts_are_explicit_audit_inputs(self) -> None:
        complete = [
            {
                "name": "openvla.action_tokens",
                "details": {"generated_positions": list(range(7))},
            },
            {
                "name": "safe.final_layer_action_features",
                "details": {
                    "action_token_positions": list(range(7)),
                    "feature_dimension": 4096,
                },
            },
        ]
        incomplete = [
            {
                "name": "openvla.action_tokens",
                "details": {"generated_positions": list(range(6))},
            },
            {
                "name": "safe.final_layer_action_features",
                "details": {
                    "action_token_positions": list(range(6)),
                    "feature_dimension": 2048,
                },
            },
        ]

        self.assertEqual(openvla_contract_issues(complete), [])
        self.assertEqual(safe_contract_issues(complete), [])
        self.assertEqual(len(openvla_contract_issues(incomplete)), 1)
        self.assertEqual(len(safe_contract_issues(incomplete)), 2)

    def test_runtime_contract_requires_modules_and_operators_per_phase(self) -> None:
        complete = [
            {"kind": kind, "context": {"phase": phase}}
            for phase in ("policy", "monitor")
            for kind in ("module", "operator")
        ]

        self.assertEqual(
            torch_trace_contract_issues(complete, ("policy", "monitor")), []
        )
        self.assertEqual(
            torch_trace_contract_issues(complete[:-1], ("policy", "monitor")),
            ["monitor trace has no observed operator events"],
        )

    def test_explicit_scalar_lineage_connects_distinct_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Recorder(Path(directory) / "raw.jsonl", {"scope": "test"}) as recorder:
                source = recorder.mark(
                    "prompt",
                    outputs=recorder.lineage("prompt:0", "hello"),
                    basis="protocol:test:prompt",
                    region="input",
                )
                sink = recorder.mark(
                    "processor",
                    inputs=recorder.lineage("prompt:0", "hello"),
                    outputs=Value("output"),
                    basis="protocol:test:processor",
                    region="compute",
                    role="sink",
                )
        graph = reduce_graph(recorder.events, recorder.annotations)
        self.assertIn(
            {
                "source": source["event_id"],
                "target": sink["event_id"],
                "kind": "dataflow",
            },
            graph["raw_edges"],
        )

    def test_storage_overlap_is_metadata_not_causal_reachability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Recorder(Path(directory) / "raw.jsonl", {"scope": "test"}) as recorder:
                original = ArrayValue(1234)
                view = ArrayValue(1234)
                source = recorder.source(
                    "source",
                    original,
                    basis="observed:test-source",
                    region="input",
                )
                alias = recorder.mark(
                    "view",
                    kind="operator",
                    inputs={"args": [original]},
                    outputs=view,
                    details={
                        "operator_semantics": {
                            "schema_status": "available",
                            "declared_aliases": [
                                {"input": "value.args[0]", "output": "value"}
                            ],
                            "mutated_input_ports": [],
                        }
                    },
                )
                sink = recorder.mark(
                    "sink",
                    inputs=view,
                    outputs=Value("output"),
                    basis="observed:test-sink",
                    region="compute",
                    role="sink",
                )
        annotations = [
            *recorder.annotations,
            {
                "event_id": alias["event_id"],
                "region": "view",
                "basis": ["observed:test-view"],
                "lifetime": "step",
            },
        ]
        graph = reduce_graph(recorder.events, annotations)
        self.assertEqual(
            graph["raw_reachability"][source["event_id"]],
            [sink["event_id"]],
        )
        self.assertEqual(
            [edge["kind"] for edge in graph["raw_edges"]],
            ["dataflow"],
        )
        self.assertEqual(len(graph["raw_storage_overlaps"]), 1)

    def test_confirmed_mutation_adds_a_causal_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Recorder(Path(directory) / "raw.jsonl", {"scope": "test"}) as recorder:
                value = ArrayValue(1234)
                recorder.source(
                    "source",
                    value,
                    basis="observed:test-source",
                    region="input",
                )
                mutation = recorder.mark(
                    "in_place",
                    kind="operator",
                    inputs={"args": [value]},
                    outputs=value,
                    details={
                        "operator_semantics": {
                            "schema_status": "available",
                            "declared_aliases": [],
                            "mutated_input_ports": ["value.args[0]"],
                        }
                    },
                )
                sink = recorder.mark(
                    "sink",
                    inputs=value,
                    outputs=Value("output"),
                    basis="observed:test-sink",
                    region="compute",
                    role="sink",
                )
        annotations = [
            *recorder.annotations,
            {
                "event_id": mutation["event_id"],
                "region": "mutation",
                "basis": ["observed:test-mutation"],
                "lifetime": "step",
            },
        ]
        graph = reduce_graph(recorder.events, annotations)
        self.assertIn(
            {
                "source": next(
                    event["event_id"]
                    for event in recorder.events
                    if event["name"] == "source"
                ),
                "target": sink["event_id"],
                "kind": "mutation",
            },
            graph["raw_edges"],
        )

    def test_temporal_labels_distinguish_world_feedback_from_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Recorder(Path(directory) / "raw.jsonl", {"scope": "test"}) as recorder:
                command = Value("command")
                observation = Value("observation")
                reward = Value("reward")
                recorder.source(
                    "command",
                    command,
                    basis="observed:test-command",
                    region="command",
                )
                environment = recorder.mark(
                    "environment",
                    inputs=command,
                    outputs={"observation": observation, "reward": reward},
                    basis="observed:test-environment",
                    region="environment",
                    details={"temporal_relation": "world_feedback"},
                )
                next_observation = recorder.mark(
                    "libero.next_observation",
                    inputs=observation,
                    outputs=observation,
                    basis="observed:test-observation",
                    region="environment",
                    role="sink",
                )
                outcome = recorder.mark(
                    "outcome",
                    inputs=reward,
                    outputs=True,
                    basis="observed:test-outcome",
                    region="outcome",
                    role="sink",
                )
        graph = reduce_graph(recorder.events, recorder.annotations)
        self.assertIn(
            {
                "source": environment["event_id"],
                "target": next_observation["event_id"],
                "kind": "temporal_world_feedback",
            },
            graph["raw_edges"],
        )
        self.assertIn(
            {
                "source": environment["event_id"],
                "target": outcome["event_id"],
                "kind": "dataflow",
            },
            graph["raw_edges"],
        )

    def test_stale_image_lineage_survives_a_copied_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Recorder(Path(directory) / "raw.jsonl", {"scope": "test"}) as recorder:
                old_image = Value("old-image")
                copied_image = Value("copied-image")
                with recorder.scope(policy_step=2):
                    recorder.source(
                        "old_camera",
                        recorder.lineage("policy_step:0:current_image", old_image),
                        basis="observed:test-camera",
                        region="camera",
                    )
                    record_policy_image(
                        recorder,
                        Value("current-image"),
                        copied_image,
                        policy_step=2,
                        source_step=0,
                    )
                    sink = recorder.mark(
                        "policy_input",
                        inputs=recorder.lineage(
                            "policy_step:2:selected_image", copied_image
                        ),
                        outputs=Value("result"),
                        basis="observed:test-policy",
                        region="policy",
                        role="sink",
                    )
        graph = reduce_graph(recorder.events, recorder.annotations)
        prior = next(
            event for event in recorder.events if event["name"] == "policy.prior_image"
        )
        selected = next(
            event for event in recorder.events if event["name"] == "policy.selected_image"
        )
        self.assertIn(
            {
                "source": prior["event_id"],
                "target": selected["event_id"],
                "kind": "temporal_buffer_history",
            },
            graph["raw_edges"],
        )
        self.assertEqual(
            graph["raw_reachability"][prior["event_id"]], [sink["event_id"]]
        )
    def test_registered_state_groups_by_mechanical_module_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Recorder(Path(directory) / "raw.jsonl", {"scope": "test"}) as recorder:
                first = recorder.mark("weight", kind="state", outputs=Value("weight"))
                second = recorder.mark("bias", kind="state", outputs=Value("bias"))
        annotations = [
            {
                "event_id": event["event_id"],
                "region": "layer_state",
                "semantic_key": "layer_state/state/policy.layer",
                "basis": ["observed:test-state-registration"],
                "lifetime": "step",
                "fault_interface": "registered_model_state",
            }
            for event in (first, second)
        ]
        graph = reduce_graph(recorder.events, annotations)
        state_regions = [
            region for region in graph["regions"] if region["name"] == "layer_state"
        ]
        self.assertEqual(len(state_regions), 1)
        self.assertEqual(state_regions[0]["event_count"], 2)
        self.assertEqual(
            state_regions[0]["aggregation"],
            "registered_state_with_same_mechanical_module_key",
        )

    def test_structural_signature_ignores_policy_step_and_values(self) -> None:
        first = [
            {
                "event_id": "e1",
                "kind": "boundary",
                "name": "input",
                "context": {"policy_step": 2},
                "outputs": [
                    {
                        "port": "value",
                        "value_id": "v1",
                        "type": "builtins.float",
                        "value": 1.0,
                    }
                ],
            }
        ]
        second = [
            {
                "event_id": "e9",
                "kind": "boundary",
                "name": "input",
                "context": {"policy_step": 50},
                "outputs": [
                    {
                        "port": "value",
                        "value_id": "v7",
                        "type": "builtins.float",
                        "value": 8.0,
                    }
                ],
            }
        ]
        self.assertEqual(
            structural_signature(first)["sha256"],
            structural_signature(second)["sha256"],
        )
        self.assertNotIn("structure", structural_signature(first))
        self.assertEqual(structural_signature(first)["event_kind_counts"], {"boundary": 1})

    def test_monitor_attachment_preserves_composition_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            composition = {
                "schema_version": 1,
                "policy_steps": 2,
                "monitor_timeline_recorded_inline": False,
            }
            (root / "composition.json").write_text(
                __import__("json").dumps(composition), encoding="utf-8"
            )
            attachment = attach_monitor_timeline(
                root,
                {"kind": "test"},
                [
                    {"policy_step": 0, "score": 0.1},
                    {"policy_step": 1, "score": 0.2},
                ],
            )
            attachment_exists = (root / "monitor-timeline.json").is_file()
            updated = __import__("json").loads(
                (root / "composition.json").read_text(encoding="utf-8")
            )
        self.assertEqual(attachment["monitor"]["kind"], "test")
        self.assertFalse(updated["monitor_timeline_recorded_inline"])
        self.assertTrue(attachment_exists)

    def test_boundary_only_rollout_builds_an_audited_composition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            evidence = RolloutEvidence(root, {"scope": "test"}, set())
            observation = {"agentview_image": Value("image")}
            image = Value("processed-image")
            evidence.begin_step(0, observation, image, image, 0, None)
            processor = evidence.processor(FakeProcessor(), image, "task", 0)
            processor("prompt", image).to("cuda")
            generated = {
                "sequences": FakeTensor((1, 20)),
                "hidden_states": [
                    [FakeTensor((1, 5, 4096))] for _ in range(7)
                ],
            }
            action = FakeArray()
            evidence.policy_outputs(
                FakeModel(), generated, action, "libero_10", 0, FakeTorch()
            )
            command = evidence.command(FakeRuntime(), action, 0)
            _observation, reward, done, _info = evidence.environment_step(
                FakeEnvironment(), command, 0
            )
            evidence.finish_step(
                0, fault_applied=False, reward=reward, done=done
            )
            result = evidence.close(success=True, policy_steps=1, fault=None)
            audit = __import__("json").loads(
                (root / "audit.json").read_text(encoding="utf-8")
            )
            events = read_events(root / "raw.jsonl")
        self.assertTrue(result["audit_passed"])
        self.assertTrue(audit["passed"])
        self.assertIn("openvla.policy_call", [event["name"] for event in events])

    def test_saturation_summary_is_descriptive_not_a_coverage_claim(self) -> None:
        summary = summarize_saturation(
            [
                {
                    "timeline": [
                        {"policy_step": 1, "template_sha256": "a"},
                        {"policy_step": 2, "template_sha256": "a"},
                    ]
                },
                {"timeline": [{"policy_step": 3, "template_sha256": "b"}]},
            ]
        )
        self.assertEqual(summary["traced_execution_count"], 3)
        self.assertEqual(summary["unique_template_count"], 2)
        self.assertIn("does not prove", summary["interpretation"])

    def test_prepare_evidence_output_removes_a_partial_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evidence"
            root.mkdir()
            (root / "raw.jsonl").touch()
            prepare_evidence_output(root)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
