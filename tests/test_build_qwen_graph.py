import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.build_qwen_graph import _record_graph
from embodied_silent_failures.evidence_graph.qwen import prompt_sha256


class QwenGraphBuilderTests(unittest.TestCase):
    def test_frozen_query_builds_audited_graph_without_qwen_internals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instruction = "put the cup away"
            response = '{"failure_now":0,"reason":"on track"}'
            query = {
                "policy_step": 10,
                "frame_steps": [5, 10],
                "frame_sha256": ["a" * 64, "b" * 64],
                "prompt_sha256": prompt_sha256(instruction, 2),
                "raw_response": response,
                "parsed_response": {"failure_now": 0, "reason": "on track"},
                "alarm": False,
            }
            trial = {
                "configuration_sha256": "configuration",
                "source": "stale",
                "task_id": 2,
                "episode_index": 4,
                "task_description": instruction,
                "run_sha256": "c" * 64,
                "video_sha256": "d" * 64,
            }
            run = {
                "configuration_sha256": "configuration",
                "configuration": {
                    "model": {
                        "id": "Qwen/Qwen3-VL-8B-Instruct",
                        "revision": "e" * 40,
                        "snapshot_sha256": "f" * 64,
                    },
                    "protocol": {"history_frames": 2},
                },
                "runtime": {
                    "model_implementation": {"sha256": "1" * 64},
                    "processor_implementation": {"sha256": "2" * 64},
                },
                "repository_state": {"revision": "3" * 40},
            }
            result = _record_graph(
                trial=trial,
                run=run,
                query=query,
                frames=[object(), object()],
                output_dir=root / "evidence",
                trial_sha256="4" * 64,
                run_sha256="5" * 64,
            )
            graph = json.loads(
                (root / "evidence" / "graph.json").read_text(encoding="utf-8")
            )
            audit = json.loads(
                (root / "evidence" / "audit.json").read_text(encoding="utf-8")
            )

        self.assertTrue(result["audit_passed"])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["construction"]["kind"], "post_hoc_lineage_reconstruction_from_frozen_qwen_scoring_artifact")
        self.assertEqual(
            {region["name"] for region in graph["regions"]},
            {
                "qwen_observation_evidence",
                "qwen_monitor_input",
                "qwen_private_compute",
                "qwen_response_parser",
                "qwen_alarm",
            },
        )
        self.assertEqual(result["sinks"], ["qwen.alarm"])


if __name__ == "__main__":
    unittest.main()
