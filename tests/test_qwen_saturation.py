import json
import math
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.qwen_artifacts import file_sha256
from embodied_silent_failures.qwen_saturation import (
    STRATA,
    coverage_novelty,
    coverage_record,
    empty_coverage_union,
    select_saturation_queries,
    update_coverage_union,
    zero_discovery_upper_bound,
)


class QwenSaturationSelectionTests(unittest.TestCase):
    def _trial(self, path: Path, condition: str, index: int) -> None:
        configuration = (
            "native-configuration" if condition == "clean" else "causal-configuration"
        )
        timeline = []
        for alarm in (False, True):
            timeline.append(
                {
                    "policy_step": 10 + int(alarm),
                    "frame_steps": list(range(8)),
                    "generated_token_ids": list(
                        range(index + 3 + (20 if alarm else 0))
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

    def test_balances_six_strata_and_uses_distinct_holdout_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "native"
            causal = root / "causal"
            native.mkdir()
            causal.mkdir()
            for index in range(14):
                self._trial(native / f"native-{index}.json", "clean", index)
                self._trial(
                    causal / f"control-{index}.json",
                    "current_image_control",
                    index,
                )
                self._trial(
                    causal / f"stale-{index}.json", "stale_image", index
                )

            def source(label: str, configuration: str, path: Path) -> dict:
                run = {
                    "configuration_sha256": configuration,
                    "configuration": {"protocol": {"history_frames": 8}},
                }
                run_path = root / f"{label}-run.json"
                run_path.write_text(json.dumps(run), encoding="utf-8")
                return {
                    "label": label,
                    "run": run,
                    "run_path": run_path,
                    "run_sha256": file_sha256(run_path),
                    "trials_dir": path,
                }

            sources = {
                "ordinary": source("ordinary", "native-configuration", native),
                "control": source(
                    "control", "causal-configuration", causal
                ),
                "stale": source("stale", "causal-configuration", causal),
            }
            result = select_saturation_queries(sources, seed=20260813)

            self.assertEqual(result["total_queries"], 36)
            self.assertEqual(result["discovery_queries"], 6)
            self.assertEqual(result["holdout_queries"], 30)
            self.assertEqual(
                [item["stratum"] for item in result["selections"][:6]],
                [f"{label}--alarm-{int(alarm)}" for label, _condition, alarm in STRATA],
            )
            for discovery in result["selections"][:6]:
                matching = [
                    item
                    for item in result["selections"][6:]
                    if item["stratum"] == discovery["stratum"]
                ]
                self.assertEqual(len(matching), 5)
                names = {item["trial_path"].name for item in matching}
                self.assertEqual(len(names), 5)
                self.assertNotIn(discovery["trial_path"].name, names)
            for label in ("ordinary", "control", "stale"):
                condition_items = [
                    item
                    for item in result["selections"]
                    if item["condition_label"] == label
                ]
                self.assertEqual(
                    len({item["trial_path"].name for item in condition_items}),
                    len(condition_items),
                )
            json.dumps(result["public_selections"])

            repeated = select_saturation_queries(sources, seed=20260813)
            self.assertEqual(
                result["selection_sha256"], repeated["selection_sha256"]
            )


class QwenCoverageTests(unittest.TestCase):
    def test_reports_each_predeclared_novelty_kind(self) -> None:
        events = [
            {
                "kind": "operator",
                "details": {
                    "module_calls": [{"path": "qwen_model.model.visual"}],
                    "operator_semantics": {"schema": "aten::linear(Tensor) -> Tensor"},
                },
            }
        ]
        graph = {
            "regions": [
                {
                    "region_id": "r0",
                    "name": "visual",
                    "semantic_key": "visual/block0",
                    "lifetime": "step",
                    "fault_interface": "qwen_internal_compute",
                    "disposition": None,
                    "basis": ["observed:torch-module:qwen_model.model.visual"],
                },
                {
                    "region_id": "r1",
                    "name": "alarm",
                    "semantic_key": "alarm",
                    "lifetime": "step",
                    "fault_interface": None,
                    "disposition": None,
                    "basis": ["protocol:test"],
                },
            ],
            "edges": [{"source": "r0", "target": "r1", "kind": "dataflow"}],
        }
        coverage = coverage_record(
            events,
            graph,
            [{"name": "input_ids", "shape": [1, 20], "dtype": "torch.int64"}],
        )
        novelty = coverage_novelty(coverage, empty_coverage_union())
        self.assertTrue(novelty["novel"])
        self.assertEqual(
            novelty["new"],
            {"regions": 2, "edges": 1, "operators": 1, "processor_shapes": 1},
        )
        union = empty_coverage_union()
        update_coverage_union(union, coverage)
        self.assertFalse(coverage_novelty(coverage, union)["novel"])

    def test_zero_discovery_bound_matches_predeclared_30_query_claim(self) -> None:
        self.assertTrue(
            math.isclose(zero_discovery_upper_bound(30), 0.0950338528553041)
        )


if __name__ == "__main__":
    unittest.main()
