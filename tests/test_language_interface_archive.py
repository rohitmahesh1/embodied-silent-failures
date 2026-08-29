import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class LanguageInterfaceArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as error:
            raise unittest.SkipTest("NumPy and PyTorch are required") from error
        cls.np = np
        cls.torch = torch

    def _decision(self, offset: float, token_position: int = 2):
        from embodied_silent_failures.language_fault import LanguageInferenceTrace
        from embodied_silent_failures.language_policy import (
            GenerationLogitTrace,
            PolicyDecision,
        )

        torch = self.torch
        by_call = {
            layer: {
                token: torch.full((1, 1, 4), offset + layer + token)
                for token in range(7)
            }
            for layer in range(32)
        }
        trace = LanguageInferenceTrace(
            action_token_position=token_position,
            block_values={layer: by_call[layer][token_position] for layer in range(32)},
            block_values_by_call=by_call,
            attention_values_by_call={
                layer: {
                    "key": {token: value.clone() for token, value in calls.items()},
                    "value": {token: value.clone() for token, value in calls.items()},
                }
                for layer, calls in by_call.items()
            },
            sequence_lengths_by_call={
                layer: {token: 300 if token == 0 else 1 for token in range(7)}
                for layer in range(32)
            },
            call_counts={layer: 7 for layer in range(32)},
            anomalies=(),
        )
        logits = GenerationLogitTrace(
            sequence_token_ids=torch.arange(307, dtype=torch.int64),
            action_token_logits=torch.full((7, 256), offset),
            top_token_ids=torch.zeros((7, 32), dtype=torch.int64),
            top_token_logits=torch.full((7, 32), offset),
            log_normalizer=torch.full((7,), offset),
            entropy=torch.full((7,), offset),
            vocabulary_size=32000,
            action_token_start=31744,
        )
        command = self.np.full((7,), offset)
        return PolicyDecision(
            raw_action=command.copy(),
            command=command.copy(),
            action_tokens=(1, 2, 3, 4, 5, 6, 7),
            hidden_states=torch.full((7, 4), offset),
            generation_logits=logits,
            trace=trace,
            inference_seconds=0.1,
        )

    def test_archive_retains_full_clean_trace_and_indexed_fault_rows(self) -> None:
        from embodied_silent_failures.language_interface_archive import (
            InterfaceArchiveBuilder,
        )

        runtime = SimpleNamespace(torch=self.torch, np=self.np)
        source = self._decision(1.0)
        clean = self._decision(2.0)
        fault = self._decision(3.0)
        replay = self._decision(4.0)
        builder = InterfaceArchiveBuilder(runtime, source, clean)
        builder.add_fault(30, fault)
        builder.add_replay(
            injection_layer=30,
            boundary_layer=31,
            boundary_kind="immediate",
            decision=replay,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interfaces.npz"
            record = builder.write(path)
            with self.np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["clean_residuals"].shape, (32, 7, 4))
                self.assertEqual(
                    archive["clean_attention_key_projections"].shape, (32, 7, 4)
                )
                self.assertEqual(
                    archive["clean_attention_value_projections"].shape, (32, 7, 4)
                )
                self.assertEqual(archive["fault_layer"].tolist(), [30])
                self.assertEqual(archive["replay_boundary_layer"].tolist(), [31])
                self.assertEqual(archive["fault_action_logits"].shape, (1, 7, 256))
                self.assertEqual(archive["fault_sequence_token_ids"].shape, (1, 307))
                self.assertEqual(
                    archive["clean_block_sequence_lengths"].shape, (32, 7)
                )
                self.assertEqual(archive["fault_residuals"].shape[0], 130)
                self.assertEqual(
                    archive["fault_attention_key_projections"].shape[0], 129
                )
                self.assertEqual(
                    archive["fault_attention_key_projections_row_layer"][0], 31
                )
                self.assertEqual(
                    archive["replay_attention_key_projections_row_layer"][0], 31
                )
            self.assertEqual(record["fault_records"], 1)
            self.assertEqual(record["boundary_replay_records"], 1)

    def test_boundary_target_selection_does_not_duplicate_final_layer(self) -> None:
        from embodied_silent_failures.language_interface import (
            boundary_replay_targets,
        )

        self.assertEqual(
            boundary_replay_targets(0, ["immediate", "final"]),
            [("immediate", 1), ("final", 31)],
        )
        self.assertEqual(
            boundary_replay_targets(30, ["immediate", "final"]),
            [("immediate", 31)],
        )
        self.assertEqual(boundary_replay_targets(31, ["immediate", "final"]), [])

    def test_replay_distinguishes_exact_output_from_omitted_attention_state(self) -> None:
        from embodied_silent_failures.language_interface import (
            boundary_replay_record,
        )

        runtime = SimpleNamespace(torch=self.torch, np=self.np)
        original = self._decision(3.0)
        replay = self._decision(4.0)
        for token in range(2, 7):
            first_layer = 31 if token == 2 else 0
            for layer in range(first_layer, 32):
                replay.trace.block_values_by_call[layer][token] = (
                    original.trace.block_values_by_call[layer][token].clone()
                )

        record = boundary_replay_record(
            runtime,
            original=original,
            replay=replay,
            injection_layer=30,
            boundary_layer=31,
            boundary_kind="immediate",
        )

        self.assertTrue(record["residual_path"]["all_coordinates_exact"])
        self.assertFalse(
            record["attention_key_projections"]["all_coordinates_exact"]
        )
        self.assertEqual(
            record["attention_key_projections"]["first_difference"],
            {"layer_index": 31, "action_token_position": 2},
        )

    def test_bfloat_archive_encoding_preserves_exact_bits(self) -> None:
        from embodied_silent_failures.language_interface_archive import _exact_array

        value = self.torch.tensor([1.0, -2.5, 0.125], dtype=self.torch.bfloat16)

        array, record = _exact_array(self.torch, value)
        restored = self.torch.from_numpy(array.copy()).view(self.torch.bfloat16)

        self.assertEqual(array.dtype, self.np.dtype(self.np.int16))
        self.assertTrue(self.torch.equal(value, restored))
        self.assertEqual(record["torch_dtype"], "torch.bfloat16")


if __name__ == "__main__":
    unittest.main()
