import csv
import json
import tempfile
import unittest
from pathlib import Path


class LanguageProductStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("NumPy is required") from error
        cls.np = np

    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        from embodied_silent_failures.artifacts import artifact_record

        np = self.np
        campaign = root / "campaign"
        attempt = campaign / "attempts" / "c000-command-example"
        scoring = campaign / "scoring"
        attempt.mkdir(parents=True)
        scoring.mkdir()
        (campaign / "run.json").write_text(
            json.dumps(
                {
                    "worker_shard": 0,
                    "execution": {
                        "experiment_code": {"revision": "fixture-revision"}
                    },
                }
            )
        )

        trajectory = attempt / "task0--ep1--succ0.trajectory.npz"
        np.savez_compressed(
            trajectory,
            policy_step=np.asarray([10, 11, 12], dtype=np.int32),
            snapshot_stage=np.asarray([0, 0, 1], dtype=np.uint8),
            simulator_state=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float64),
            decision_policy_step=np.asarray([10, 11], dtype=np.int32),
            raw_action=np.asarray([[1] * 7, [2] * 7], dtype=np.float64),
            executed_command=np.asarray([[3] * 7, [4] * 7], dtype=np.float64),
            action_tokens=np.asarray([[5] * 7, [6] * 7], dtype=np.int32),
            sequence_token_ids=np.asarray([[7] * 4, [8] * 4], dtype=np.int64),
            action_token_logits=np.full((2, 7, 256), 9, dtype=np.float32),
            global_top_token_ids=np.full((2, 7, 32), 10, dtype=np.int64),
            global_top_token_logits=np.full((2, 7, 32), 11, dtype=np.float32),
            action_log_normalizer=np.full((2, 7), 12, dtype=np.float32),
            action_entropy=np.full((2, 7), 13, dtype=np.float32),
            observation_0000=np.full((3, 2, 2, 3), 14, dtype=np.uint8),
            observation_0001=np.asarray([[15], [16], [17]], dtype=np.float32),
        )
        trajectory_artifact = artifact_record(trajectory)
        marker = {
            "trajectory_archive": {
                "artifact": trajectory_artifact,
                "simulator_state": {"sha256": "simulator-series"},
                "observations": [
                    {
                        "name": "agentview_image",
                        "archive_key": "observation_0000",
                        "dtype": "|u1",
                        "shape": [3, 2, 2, 3],
                        "sha256": "image-series",
                        "kind": "image",
                    },
                    {
                        "name": "robot0_eef_pos",
                        "archive_key": "observation_0001",
                        "dtype": "<f4",
                        "shape": [3, 1],
                        "sha256": "eef-series",
                        "kind": "numeric",
                    },
                ],
            }
        }
        marker_path = attempt / "task0--ep1.complete.json"
        marker_path.write_text(json.dumps(marker))

        safe_archive = scoring / "physical-safe.npz"
        np.savez_compressed(
            safe_archive,
            runs=np.asarray(["c000-command-example", "c001-control"]),
            task_ids=np.asarray([0, 0], dtype=np.int16),
            episode_indices=np.asarray([1, 2], dtype=np.int16),
            successes=np.asarray([False, True]),
            lengths=np.asarray([12, 12], dtype=np.int16),
            scores=np.asarray([[0] * 11 + [20], [0] * 12], dtype=np.float32),
            alphas=np.asarray([0.1], dtype=np.float32),
            bands=np.asarray([[10] * 12], dtype=np.float32),
        )
        physical_scores = scoring / "physical-safe.json"
        physical_scores.write_text(
            json.dumps(
                {
                    "monitor": {"primary_alpha": 0.1},
                    "score_archive": {
                        "path": str(safe_archive),
                        "sha256": artifact_record(safe_archive)["sha256"],
                    },
                    "records": [
                        {
                            "run": "c000-command-example",
                            "condition": "activation_fault",
                            "task_id": 0,
                            "episode_index": 1,
                            "length": 12,
                            "success": False,
                            "fault": {
                                "kind": "language_block_temporal_replacement",
                                "policy_step": 10,
                                "source_policy_step": 9,
                                "action_token_position": 2,
                                "layer_index": 4,
                                "command_group": {"command_id": "command-example"},
                            },
                        },
                        {
                            "run": "c001-control",
                            "condition": "activation_control",
                            "task_id": 0,
                            "episode_index": 2,
                            "length": 12,
                            "success": True,
                            "fault": {
                                "kind": "language_block_current_control",
                                "policy_step": 10,
                                "source_policy_step": 9,
                                "action_token_position": 2,
                            },
                        },
                    ],
                }
            )
        )

        context = {
            "context_id": "c000",
            "analysis_split": "development",
            "task_id": 0,
            "episode_index": 1,
            "phase": "early",
            "worker_shard": 0,
            "policy_step": 10,
            "action_token_position": 2,
        }
        language_scores = scoring / "language-safe.json"
        language_scores.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "record_id": "c000:layer04",
                            "status": "scored",
                            "context_id": "c000",
                            "context": context,
                            "layer_index": 4,
                            "composition_verified": True,
                            "control_success": True,
                            "terminal_success": False,
                            "monitor_horizon": "complete_physical_trace",
                            "physical_run": "c000-command-example",
                            "command_id": "command-example",
                            "local_measurements": {
                                "site_id": "site-4",
                                "propagation": [],
                                "executed_command": {"exact_equal": False},
                            },
                            "alarms": {
                                "0.1": {
                                    "post_fault_any": {"triggered": True},
                                    "within_10_steps": {"triggered": True},
                                    "within_25_steps": {"triggered": True},
                                }
                            },
                            "alarm_at_fault": False,
                            "alarm_before_fault": False,
                            "control_alarm_at_fault": False,
                        }
                    ]
                }
            )
        )
        return campaign, language_scores, physical_scores

    def test_extracts_aligned_policy_state_monitor_and_outcome_data(self) -> None:
        from embodied_silent_failures.language_product_state import extract_campaign

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, language_scores, physical_scores = self._write_fixture(root)
            output = root / "output"
            result = extract_campaign(
                np=self.np,
                campaign_dir=campaign,
                language_scores_path=language_scores,
                physical_scores_path=physical_scores,
                output_dir=output,
                verify_source_hashes=True,
            )

            self.assertEqual(result["status"], "complete_with_errors")
            self.assertEqual(result["coverage"]["extracted_branches"], 1)
            self.assertEqual(result["coverage"]["failed_branches"], 1)
            self.assertEqual(result["coverage"]["interventions_with_product_state"], 1)
            self.assertEqual(result["errors"][0]["run"], "c001-control")
            with (output / "branches.csv").open(newline="") as file:
                branch = next(csv.DictReader(file))
            self.assertEqual(branch["run"], "c000-command-example")
            self.assertEqual(branch["safe_score_at_fault"], "0.0")
            self.assertEqual(branch["safe_alarm_post_fault_any"], "True")
            self.assertEqual(branch["operational_silent_failure"], "False")

            with self.np.load(output / "product-state.npz", allow_pickle=False) as data:
                self.assertEqual(data["raw_action_values"].tolist(), [1.0] * 7)
                self.assertEqual(data["raw_action_shapes"].tolist(), [[7]])
                self.assertEqual(
                    data["numeric_state_before_values"].tolist(), [1.0, 2.0, 15.0]
                )
                self.assertEqual(
                    data["numeric_state_after_values"].tolist(), [3.0, 4.0, 16.0]
                )
            self.assertEqual(len(result["omitted_image_series"]), 1)


if __name__ == "__main__":
    unittest.main()
