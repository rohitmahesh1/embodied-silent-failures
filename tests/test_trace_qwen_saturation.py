import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from embodied_silent_failures.artifacts import write_json_atomic
from embodied_silent_failures.qwen_artifacts import file_sha256, load_json
from embodied_silent_failures.qwen_saturation import select_saturation_queries
from embodied_silent_failures.trace_qwen_saturation import run_campaign


class TraceQwenSaturationCampaignTests(unittest.TestCase):
    def _run(self, path: Path, configuration: str) -> dict:
        value = {
            "configuration_sha256": configuration,
            "configuration": {"protocol": {"history_frames": 8}},
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    def _trial(
        self, path: Path, condition: str, configuration: str, index: int
    ) -> None:
        timeline = []
        for alarm in (False, True):
            timeline.append(
                {
                    "policy_step": 10 + int(alarm),
                    "frame_steps": list(range(8)),
                    "generated_token_ids": list(
                        range(index + 4 + (10 if alarm else 0))
                    ),
                    "parse_error": None,
                    "parsed_response": {
                        "failure_now": int(alarm),
                        "reason": "recorded",
                    },
                    "alarm": alarm,
                }
            )
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "configuration_sha256": configuration,
                    "condition": condition,
                    "task_id": index,
                    "episode_index": index,
                    "timeline": timeline,
                }
            ),
            encoding="utf-8",
        )

    def test_checkpoints_all_queries_and_resumes_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_trials = root / "native-trials"
            causal_trials = root / "causal-trials"
            native_trials.mkdir()
            causal_trials.mkdir()
            native_run_path = root / "native-run.json"
            causal_run_path = root / "causal-run.json"
            native_run = self._run(native_run_path, "native-config")
            causal_run = self._run(causal_run_path, "causal-config")
            for index in range(6):
                self._trial(
                    native_trials / f"native-{index}.json",
                    "clean",
                    "native-config",
                    index,
                )
                self._trial(
                    causal_trials / f"control-{index}.json",
                    "current_image_control",
                    "causal-config",
                    index,
                )
                self._trial(
                    causal_trials / f"stale-{index}.json",
                    "stale_image",
                    "causal-config",
                    index,
                )
            sources = {
                "ordinary": {
                    "label": "ordinary",
                    "run": native_run,
                    "run_path": native_run_path,
                    "run_sha256": file_sha256(native_run_path),
                    "trials_dir": native_trials,
                },
                "control": {
                    "label": "control",
                    "run": causal_run,
                    "run_path": causal_run_path,
                    "run_sha256": file_sha256(causal_run_path),
                    "trials_dir": causal_trials,
                },
                "stale": {
                    "label": "stale",
                    "run": causal_run,
                    "run_path": causal_run_path,
                    "run_sha256": file_sha256(causal_run_path),
                    "trials_dir": causal_trials,
                },
            }
            selected = select_saturation_queries(
                sources, seed=20260813, holdouts_per_stratum=1
            )["selections"][0]
            coverage = {
                "schema_version": 1,
                "regions": [{"signature": "r"}],
                "edges": [{"signature": "e"}],
                "operators": [{"signature": "o"}],
                "processor_shapes": [{"signature": "p"}],
            }
            seed_trace = root / "seed-trace"
            seed_trace.mkdir()
            write_json_atomic(seed_trace / "audit.json", {"passed": True})
            write_json_atomic(
                seed_trace / "composition.json",
                {
                    "equivalent_to_frozen_query": True,
                    "selection": {
                        "trial": selected["trial_path"].name,
                        "policy_step": selected["policy_step"],
                    },
                    "trial_sha256": selected["trial_sha256"],
                    "trace_revision": "trace-commit",
                    "processed_inputs": [],
                },
            )
            write_json_atomic(seed_trace / "graph.json", {})
            write_json_atomic(seed_trace / "coverage.json", coverage)
            (seed_trace / "raw.jsonl").write_text("", encoding="utf-8")
            output = root / "campaign"
            args = argparse.Namespace(
                native_run=native_run_path,
                native_trials=native_trials,
                causal_run=causal_run_path,
                causal_trials=causal_trials,
                seed_trace=seed_trace,
                output_dir=output,
                cache_dir=None,
                seed=20260813,
                holdouts_per_stratum=1,
            )
            runtime = SimpleNamespace(
                torch=SimpleNamespace(
                    cuda=SimpleNamespace(empty_cache=lambda: None)
                )
            )
            calls = []

            def fake_trace(_runtime, item, attempt):
                calls.append(item["index"])
                attempt.mkdir(parents=True)
                write_json_atomic(attempt / "audit.json", {"passed": True})
                write_json_atomic(
                    attempt / "composition.json",
                    {
                        "equivalent_to_frozen_query": True,
                        "selection": {
                            "trial": item["trial_path"].name,
                            "policy_step": item["policy_step"],
                        },
                        "trial_sha256": item["trial_sha256"],
                    },
                )
                write_json_atomic(attempt / "coverage.json", coverage)
                return {"audit_passed": True}

            with patch(
                "embodied_silent_failures.trace_qwen_saturation.load_trace_runtime",
                return_value=runtime,
            ), patch(
                "embodied_silent_failures.trace_qwen_saturation.trace_query",
                side_effect=fake_trace,
            ):
                with redirect_stdout(io.StringIO()):
                    result = run_campaign(args)

            self.assertEqual(calls, list(range(1, 12)))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_queries"], 12)
            self.assertEqual(result["completed_holdouts"], 6)
            self.assertEqual(result["novel_holdouts"], 0)
            self.assertTrue(result["saturated"])

            with patch(
                "embodied_silent_failures.trace_qwen_saturation.load_trace_runtime",
                side_effect=AssertionError("completed resume loaded Qwen"),
            ):
                resumed = run_campaign(args)
            self.assertEqual(resumed, load_json(output / "campaign.json"))


if __name__ == "__main__":
    unittest.main()
