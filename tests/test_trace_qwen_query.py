import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.evidence_graph.qwen import internal_annotations
from embodied_silent_failures.qwen_artifacts import select_trace_query


class QwenTraceSelectionTests(unittest.TestCase):
    def _write_trial(
        self, directory: Path, name: str, queries: list[dict], *, status: str = "complete"
    ) -> None:
        value = {
            "status": status,
            "configuration_sha256": "a" * 64,
            "timeline": queries,
        }
        (directory / name).write_text(json.dumps(value), encoding="utf-8")

    def test_selects_longest_valid_full_history_without_alarm_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_trial(
                root,
                "b.json",
                [
                    {
                        "policy_step": 10,
                        "frame_steps": list(range(8)),
                        "generated_token_ids": [1, 2, 3, 4],
                        "parse_error": None,
                        "parsed_response": {"failure_now": 1, "reason": "x"},
                        "alarm": True,
                    }
                ],
            )
            self._write_trial(
                root,
                "a.json",
                [
                    {
                        "policy_step": 20,
                        "frame_steps": list(range(8)),
                        "generated_token_ids": [1, 2, 3, 4],
                        "parse_error": None,
                        "parsed_response": {"failure_now": 0, "reason": "x"},
                        "alarm": False,
                    },
                    {
                        "policy_step": 25,
                        "frame_steps": list(range(8)),
                        "generated_token_ids": list(range(20)),
                        "parse_error": "invalid response",
                        "parsed_response": None,
                        "alarm": None,
                    },
                ],
            )

            selected = select_trace_query(
                root, configuration_sha256="a" * 64, history_frames=8
            )

            self.assertEqual(selected["trial_path"].name, "a.json")
            self.assertEqual(selected["policy_step"], 20)
            self.assertEqual(selected["generated_tokens"], 4)
            self.assertFalse(selected["selection"]["alarm_used_for_selection"])
            self.assertEqual(selected["selection"]["eligible_query_count"], 2)

    def test_rejects_an_incomplete_selection_census(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_trial(root, "trial.json", [], status="running")
            with self.assertRaisesRegex(ValueError, "incomplete trial"):
                select_trace_query(
                    root, configuration_sha256="a" * 64, history_frames=8
                )


class QwenInternalAnnotationTests(unittest.TestCase):
    def test_uses_observed_module_ownership_and_exact_paths(self) -> None:
        events = [
            {
                "event_id": "e1",
                "kind": "module",
                "name": "module.qwen_model.model.visual.blocks.0",
                "context": {"phase": "qwen_model"},
                "details": {"module_path": "qwen_model.model.visual.blocks.0"},
            },
            {
                "event_id": "e2",
                "kind": "state",
                "name": "visual.weight.registered_state",
                "context": {"phase": "qwen_model"},
                "details": {
                    "root": "qwen_model",
                    "registrations": [
                        {
                            "module_path": "qwen_model.model.visual.blocks.0",
                            "name": "weight",
                        }
                    ],
                },
            },
            {
                "event_id": "e3",
                "kind": "state",
                "name": "shared.weight.registered_state",
                "context": {"phase": "qwen_model"},
                "details": {
                    "root": "qwen_model",
                    "registrations": [
                        {"module_path": "qwen_model.model.visual", "name": "weight"},
                        {
                            "module_path": "qwen_model.model.language_model",
                            "name": "weight",
                        },
                    ],
                },
            },
        ]

        annotations = internal_annotations(events, model_basis="code:qwen:test")
        by_id = {item["event_id"]: item for item in annotations}

        self.assertEqual(by_id["e1"]["region"], "qwen_model_visual")
        self.assertEqual(
            by_id["e1"]["semantic_key"],
            "qwen_model/module/qwen_model.model.visual.blocks.0",
        )
        self.assertEqual(by_id["e2"]["region"], "qwen_model_visual")
        self.assertEqual(by_id["e2"]["fault_interface"], "registered_qwen_model_state")
        self.assertEqual(by_id["e3"]["region"], "qwen_model_shared_state")


if __name__ == "__main__":
    unittest.main()
