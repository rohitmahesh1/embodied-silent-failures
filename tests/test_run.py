import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from embodied_silent_failures.openvla_rollout import (
    REPLAY_OBSERVATION_TOLERANCE,
    CounterfactualReplayDivergence,
    CounterfactualReplayTerminated,
    build_image_intervention_record,
    image_fault_applied,
)
from embodied_silent_failures.openvla_runtime import array_sha256, model_config
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.replay import paired_clean_results
from embodied_silent_failures.run_openvla import (
    Arguments,
    _execution_record,
    _prepare_run,
)


class RunTests(unittest.TestCase):
    def arguments(self, output_dir: Path, resume: bool = False) -> Arguments:
        return Arguments(
            checkpoint=Path("/checkpoint"),
            openvla_root=Path("/openvla"),
            libero_root=Path("/libero"),
            output_dir=output_dir,
            task_suite="libero_10",
            task_ids="0",
            trial_manifest=None,
            episode_start=0,
            episode_stop=1,
            episode_stride=1,
            seed=7,
            wait_steps=10,
            save_video=True,
            resume=resume,
            fault_site=None,
            fault_manifest=None,
            stale_image_manifest=None,
            image_input_mode="stale",
            fault_layer=None,
            fault_policy_step=None,
            fault_generation_step=0,
            fault_bit_index=None,
            fault_feature_index=None,
            fault_seed=0,
            replay_clean_prefix=False,
            paired_clean_dirs=[],
        )

    def test_prepare_run_supports_only_matching_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            metadata = {
                "configuration": {"task": 0},
                "trial_plan": [{"task_id": 0, "episode_index": 0}],
                "created_at": "initial",
                "upstream_revisions": {
                    "experiment_code": "first-revision",
                    "experiment_code_dirty": False,
                },
            }
            args = self.arguments(output_dir)
            _prepare_run(args, metadata)

            run_path = output_dir / "run.json"
            self.assertEqual(json.loads(run_path.read_text()), metadata)
            _prepare_run(replace(args, resume=True), metadata)

            resumed = {
                **metadata,
                "created_at": "later",
                "upstream_revisions": {
                    "experiment_code": "second-revision",
                    "experiment_code_dirty": False,
                },
            }
            (output_dir / "task0--ep0.complete.json").touch()
            _prepare_run(replace(args, resume=True), resumed)
            stored = json.loads(run_path.read_text())
            self.assertEqual(
                stored["resume_code_revisions"],
                [
                    {
                        "resumed_at": "later",
                        "experiment_code": "second-revision",
                        "experiment_code_dirty": False,
                        "existing_completion_count": 1,
                        "existing_exclusion_count": 0,
                    }
                ],
            )
            _prepare_run(replace(args, resume=True), resumed)
            self.assertEqual(
                len(json.loads(run_path.read_text())["resume_code_revisions"]), 1
            )

            changed = {**metadata, "trial_plan": []}
            with self.assertRaises(ValueError):
                _prepare_run(replace(args, resume=True), changed)
            with self.assertRaises(FileExistsError):
                _prepare_run(args, metadata)

    def test_replay_divergence_preserves_step_error_and_tolerance(self) -> None:
        error = CounterfactualReplayDivergence(180, 0.0163)

        self.assertEqual(error.policy_step, 180)
        self.assertEqual(error.error, 0.0163)
        self.assertIn(f"exceeds {REPLAY_OBSERVATION_TOLERANCE:.3g}", str(error))

    def test_early_replay_termination_preserves_both_steps(self) -> None:
        error = CounterfactualReplayTerminated(52, 198)

        self.assertEqual(error.policy_step, 52)
        self.assertEqual(error.intervention_step, 198)
        self.assertEqual(
            error.reason, "counterfactual_replay_terminated_before_intervention"
        )

    def test_current_image_control_records_matched_stale_intervention(self) -> None:
        from embodied_silent_failures.stale_image_manifest import StaleImageSpec

        record = build_image_intervention_record(
            StaleImageSpec(policy_step=80, image_lag=1, source_policy_step=79),
            "current_control",
            trial_seed=17,
        )

        self.assertEqual(
            record,
            {
                "kind": "current_image_control",
                "policy_step": 80,
                "input_policy_step": 80,
                "matched_stale_image_lag": 1,
                "matched_stale_source_policy_step": 79,
                "trial_seed": 17,
            },
        )

    def test_only_stale_image_mode_records_an_applied_fault(self) -> None:
        from embodied_silent_failures.stale_image_manifest import StaleImageSpec

        spec = StaleImageSpec(policy_step=80, image_lag=1, source_policy_step=79)

        self.assertTrue(image_fault_applied(spec, "stale", 80))
        self.assertFalse(image_fault_applied(spec, "stale", 79))
        self.assertFalse(image_fault_applied(spec, "current_control", 80))

    def test_execution_record_identifies_the_launch_that_wrote_a_trial(self) -> None:
        metadata = {
            "created_at": "2026-08-14T12:00:00+00:00",
            "repository_states": {
                "experiment_code": {
                    "revision": "revision-two",
                    "dirty": False,
                    "worktree_sha256": "clean-tree",
                }
            },
            "evidence_graph_code_sha256": {
                "embodied_silent_failures/run_openvla.py": "runner-hash",
                "embodied_silent_failures/openvla_rollout.py": "rollout-hash",
            },
        }

        self.assertEqual(
            _execution_record(metadata),
            {
                "run_started_at": "2026-08-14T12:00:00+00:00",
                "experiment_code": metadata["repository_states"]["experiment_code"],
                "run_openvla_sha256": "runner-hash",
                "openvla_rollout_sha256": "rollout-hash",
            },
        )

    def test_initial_state_hash_includes_values_shape_and_dtype(self) -> None:
        class Array:
            def __init__(self, values: bytes, shape: tuple[int, ...], dtype: str):
                self.values = values
                self.shape = shape
                self.dtype = SimpleNamespace(hasobject=False, str=dtype)

            def tobytes(self) -> bytes:
                return self.values

        arrays = SimpleNamespace(
            asarray=lambda value: value,
            ascontiguousarray=lambda value: value,
        )
        runtime = SimpleNamespace(np=arrays)
        state = Array(b"values", (1, 2), "<f4")
        digest = array_sha256(runtime, state)

        self.assertEqual(
            digest, array_sha256(runtime, Array(b"values", (1, 2), "<f4"))
        )
        self.assertNotEqual(
            digest, array_sha256(runtime, Array(b"values", (2, 1), "<f4"))
        )
        self.assertNotEqual(
            digest, array_sha256(runtime, Array(b"values", (1, 2), "<f8"))
        )
        self.assertNotEqual(
            digest, array_sha256(runtime, Array(b"changed", (1, 2), "<f4"))
        )

    def test_model_config_uses_the_declared_checkpoint_and_suite(self) -> None:
        config = model_config(Path("/checkpoint"), "libero_10")

        self.assertEqual(config.pretrained_checkpoint, "/checkpoint")
        self.assertEqual(config.task_suite_name, "libero_10")
        self.assertEqual(config.unnorm_key, "libero_10")
        self.assertTrue(config.output_hidden_states)

    def test_paired_clean_results_select_only_successful_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clean_dir = Path(directory)
            for episode, success in ((0, True), (1, False)):
                value = {
                    "status": "complete",
                    "condition": "clean",
                    "task_id": 0,
                    "episode_index": episode,
                    "initial_state_sha256": f"state-{episode}",
                    "trial_seed": 10 + episode,
                    "success": success,
                    "policy_steps": 100,
                }
                path = clean_dir / f"task0--ep{episode}.complete.json"
                path.write_text(json.dumps(value), encoding="utf-8")

            plan = [Trial(0, 0), Trial(0, 1)]
            eligible, indexed = paired_clean_results([clean_dir], plan)

            self.assertEqual(eligible, [Trial(0, 0)])
            self.assertEqual(set(indexed), set(plan))
            self.assertEqual(indexed[Trial(0, 0)]["_source_dir"], str(clean_dir))

    def test_paired_clean_results_reject_missing_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                paired_clean_results([Path(directory)], [Trial(0, 0)])


if __name__ == "__main__":
    unittest.main()
