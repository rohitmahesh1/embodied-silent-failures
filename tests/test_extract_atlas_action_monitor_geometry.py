from __future__ import annotations

import unittest

from embodied_silent_failures.extract_atlas_action_monitor_geometry import (
    _fault_provenance,
)


class ExtractAtlasActionMonitorGeometryTests(unittest.TestCase):
    def test_policy_logit_replacement_uses_recorded_source_step(self) -> None:
        completion = {
            "fault": {
                "source_policy_step": 9,
                "representative_local_measurements": {
                    "identity": {
                        "kind": "module_output",
                        "module_call_index": 6,
                        "module_path": "policy",
                        "output_port": "value.logits",
                    },
                    "topologies": ["action_only"],
                },
            }
        }

        result = _fault_provenance(completion)

        self.assertEqual(result["stale_logit_source_step"], 9)
        self.assertEqual(result["stale_logit_token_index"], 6)
        self.assertFalse(result["same_feature_comparable"])

    def test_shared_site_is_same_feature_comparable(self) -> None:
        completion = {
            "fault": {
                "source_policy_step": 9,
                "representative_local_measurements": {
                    "identity": {
                        "kind": "module_output",
                        "module_path": "policy.language_model.model.layers.12",
                        "output_port": "value",
                    },
                    "topologies": ["shared_action_and_monitor_evidence"],
                },
            }
        }

        result = _fault_provenance(completion)

        self.assertIsNone(result["stale_logit_source_step"])
        self.assertIsNone(result["stale_logit_token_index"])
        self.assertTrue(result["same_feature_comparable"])


if __name__ == "__main__":
    unittest.main()
