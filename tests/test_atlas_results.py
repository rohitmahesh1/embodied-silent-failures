import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.atlas_results import consolidate_intervention_atlas
from embodied_silent_failures.provenance import file_sha256


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class AtlasResultTests(unittest.TestCase):
    def test_consolidation_maps_exact_commands_to_one_physical_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            context = {
                "context_id": "c0000",
                "task_id": 0,
                "episode_index": 1,
                "analysis_split": "holdout",
            }
            manifest = {
                "campaign": "openvla_graph_derived_temporal_intervention_atlas",
                "analysis_contract": {"primary_unit": "intervention"},
                "contexts": [context],
            }
            _write(manifest_path, manifest)
            worker = root / "worker0"
            _write(
                worker / "run.json",
                {
                    "campaign": manifest["campaign"],
                    "worker_shard": 0,
                    "execution": {"manifest_file_sha256": file_sha256(manifest_path)},
                },
            )
            complete_result = {"status": "complete", "success": True}
            failed_result = {"status": "complete", "success": False}
            _write(
                worker / "contexts" / "c0000" / "context.complete.json",
                {
                    "local_complete": 2,
                    "local_unresolved": 0,
                    "unique_faulted_commands": 1,
                    "terminal_unresolved": 0,
                    "faulted_terminal_skip_reason": None,
                    "command_groups": [
                        {
                            "command_id": "changed",
                            "member_site_ids": ["changed-site"],
                        }
                    ],
                    "branches": [
                        {"branch": "control", "result": complete_result},
                        {
                            "branch": "command-changed",
                            "command_group": {"command_id": "changed"},
                            "result": failed_result,
                        },
                    ],
                },
            )
            common = {
                "status": "complete",
                "topologies": ["shared_action_and_monitor_evidence"],
                "sampling": {"site_inclusion_probability": 0.5},
                "fault": {},
                "raw_action": {},
                "action_tokens": {},
                "action_logits": {},
                "safe_input": {},
                "inference_seconds": 1.0,
            }
            _write(
                worker / "contexts" / "c0000" / "local.json",
                {
                    "source_collection": {},
                    "current_collection": {},
                    "interventions": [
                        {
                            **common,
                            "site_id": "same-site",
                            "executed_command": {"exact_equal": True},
                        },
                        {
                            **common,
                            "site_id": "changed-site",
                            "executed_command": {"exact_equal": False},
                        },
                    ],
                },
            )

            result = consolidate_intervention_atlas(manifest_path, [worker])

            records = {value["site_id"]: value for value in result["interventions"]}
            self.assertEqual(records["same-site"]["physical_branch"], "control")
            self.assertFalse(records["same-site"]["policy_failure"])
            self.assertEqual(
                records["changed-site"]["physical_branch"], "command-changed"
            )
            self.assertTrue(records["changed-site"]["policy_failure"])
            self.assertEqual(result["coverage"]["policy_failures"], 1)


if __name__ == "__main__":
    unittest.main()
