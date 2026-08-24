import ast
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from embodied_silent_failures.campaign_runner import trial_process_error
from embodied_silent_failures.pi05_supervisor import InfrastructureError, PolicyServer
from embodied_silent_failures.pi0_fast_contract import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    FEATURE_DIMENSION,
    FEATURE_SOURCE_DTYPE,
    FEATURE_TRANSPORT_ENCODING,
    PALIGEMMA_EOS_TOKEN,
    PARITY_ACTION_ATOL,
    PROTOCOL_VERSION,
    validate_replan_steps,
)
from embodied_silent_failures.pi0_fast_policy import (
    INSTRUMENTED_STATIC_ARGUMENTS,
    REQUEST_KEY,
    decode_audit_classification,
    decode_with_diagnostics,
    evidence_metadata,
    parity_record,
)
from embodied_silent_failures.pi0_fast_rollout import (
    EXACT_PARITY_EXIT_CODE,
    FALLBACK_CONTINUATION_CONDITION,
    DecodeAuditComplete,
    ExactParityError,
    RolloutConfig,
    array_sha256,
    bfloat16_bits_to_float32,
    run_trial,
    validate_decode_audit_response,
    validate_policy_response,
)
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.run_pi0_fast import _trial_plan


def _response(decision_id, compare_reference=False, decode_status=None):
    decoded_tokens = 3
    response = {
        "actions": np.arange(
            ACTION_HORIZON * ACTION_DIMENSION, dtype=np.float32
        ).reshape(ACTION_HORIZON, ACTION_DIMENSION),
        "raw_action_tokens": np.asarray([11, 12, PALIGEMMA_EOS_TOKEN]),
        "evidence": {
            "encoded_bfloat16_bits": np.zeros(
                (decoded_tokens, FEATURE_DIMENSION), dtype=np.uint16
            ),
            "pre_logits_bfloat16_bits": np.zeros(
                (decoded_tokens, FEATURE_DIMENSION), dtype=np.uint16
            ),
            "action_token_logits_bfloat16_bits": np.zeros(
                (decoded_tokens, FEATURE_DIMENSION), dtype=np.uint16
            ),
            "action_token_start": 254_976,
            "action_token_stop": 257_024,
            "decoded_tokens": decoded_tokens,
            "decision_id": decision_id,
            "protocol_version": PROTOCOL_VERSION,
            "source_dtypes": {
                "encoded_bfloat16_bits": FEATURE_SOURCE_DTYPE,
                "pre_logits_bfloat16_bits": FEATURE_SOURCE_DTYPE,
                "action_token_logits_bfloat16_bits": FEATURE_SOURCE_DTYPE,
            },
            "transport_encoding": FEATURE_TRANSPORT_ENCODING,
        },
        "policy_timing": {"infer_ms": 1.0},
        "server_timing": {"infer_ms": 1.1},
    }
    if compare_reference:
        response["reference_comparison"] = {
            "decoded_length_exact": True,
            "decoded_tokens_exact": True,
            "unused_padding_exact": True,
            "instrumented_decoded_tokens": 3,
            "reference_decoded_tokens": 3,
            "maximum_absolute_action_error": 0.0,
            "action_atol": PARITY_ACTION_ATOL,
            "passed": True,
        }
    if decode_status is not None:
        captured_stdout = ""
        if decode_status == "tokenizer_fallback":
            captured_stdout = (
                "Error decoding tokens: cannot reshape array of size 68 "
                "into shape (7)\n"
            )
        response["decode_diagnostic"] = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "decision_id": decision_id,
            "status": decode_status,
            "captured_stdout": captured_stdout,
            "action_shape": [ACTION_HORIZON, ACTION_DIMENSION],
            "actions_sha256": array_sha256(response["actions"]),
            "policy_visible_actions_all_zero": False,
        }
    return response


def _decode_audit_response(decision_id):
    tokens = {
        "decode_step": 2,
        "decoded_token_ids": [21, PALIGEMMA_EOS_TOKEN],
        "decoded_tokens_sha256": "decoded",
        "full_buffer_sha256": "full",
    }
    return {
        "decode_audit": {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "decision_id": decision_id,
            "instrumented": {
                "tokens": tokens,
                "decode": {"status": "tokenizer_fallback"},
            },
            "reference": {
                "tokens": tokens,
                "decode": {"status": "tokenizer_fallback"},
            },
            "comparison": {
                "decoded_length_exact": True,
                "decoded_tokens_exact": True,
                "full_buffer_exact": True,
            },
            "classification": "same_tokens_same_tokenizer_fallback",
        }
    }


class FakeClient:
    def __init__(self):
        self.requests = []

    def infer(self, element):
        request = element[REQUEST_KEY]
        self.requests.append(request)
        return _response(request["decision_id"], request["compare_reference"])


class AuditClient(FakeClient):
    def infer(self, element):
        request = element[REQUEST_KEY]
        self.requests.append(request)
        if request["decision_id"] == 1:
            return _decode_audit_response(request["decision_id"])
        return _response(request["decision_id"], request["compare_reference"])


class FallbackClient(FakeClient):
    def infer(self, element):
        request = element[REQUEST_KEY]
        self.requests.append(request)
        return _response(
            request["decision_id"],
            request["compare_reference"],
            decode_status=(
                "tokenizer_fallback"
                if request["decision_id"] == 1
                else "decoded"
            ),
        )


class FakeEnvironment:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def reset(self):
        return None

    def set_init_state(self, initial_state):
        return {"joint": np.asarray(initial_state)}

    def step(self, action):
        self.steps += 1
        observation = {"joint": np.asarray([self.steps], dtype=np.float32)}
        return observation, float(self.steps == 6), self.steps == 6, {}

    def close(self):
        self.closed = True


class Pi0FastTests(unittest.TestCase):
    def test_public_config_can_omit_policy_metadata(self):
        self.assertEqual(
            evidence_metadata(None),
            {
                "evidence_protocol_version": PROTOCOL_VERSION,
                "evidence_names": [
                    "encoded",
                    "pre_logits",
                    "action_token_logits",
                ],
            },
        )

    def test_shape_defining_decode_limit_is_static_under_jit(self):
        self.assertEqual(
            INSTRUMENTED_STATIC_ARGUMENTS,
            ("max_decoding_steps", "temperature", "n_action_samples"),
        )

    def test_client_files_parse_as_python_38(self):
        project_root = Path(__file__).resolve().parents[1]
        for name in (
            "artifacts.py",
            "pi0_fast_contract.py",
            "pi0_fast_rollout.py",
            "plan.py",
            "provenance.py",
            "run_pi0_fast_trial.py",
        ):
            path = project_root / "embodied_silent_failures" / name
            with self.subTest(name=name):
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(path), feature_version=(3, 8))
                if name == "run_pi0_fast_trial.py":
                    self.assertNotIn("BooleanOptionalAction", source)

    def test_campaign_is_interleaved_across_tasks(self):
        args = SimpleNamespace(
            trial_manifest=None,
            task_ids="0-2",
            episode_start=0,
            episode_stop=2,
            episode_stride=1,
        )
        self.assertEqual(
            _trial_plan(args),
            [
                Trial(0, 0),
                Trial(1, 0),
                Trial(2, 0),
                Trial(0, 1),
                Trial(1, 1),
                Trial(2, 1),
            ],
        )

    def test_parity_exit_is_campaign_blocking(self):
        error = trial_process_error(
            EXACT_PARITY_EXIT_CODE,
            (EXACT_PARITY_EXIT_CODE,),
            Path("trial.log"),
        )
        self.assertIsInstance(error, InfrastructureError)
        self.assertIsInstance(
            trial_process_error(1, (), Path("trial.log")), RuntimeError
        )

    def test_policy_server_rejects_an_ambiguous_health_probe(self):
        with self.assertRaisesRegex(ValueError, "health mode"):
            PolicyServer(SimpleNamespace(), health_mode="guess")

    def test_replan_range_matches_action_horizon(self):
        self.assertEqual(validate_replan_steps(ACTION_HORIZON), ACTION_HORIZON)
        for value in (0, ACTION_HORIZON + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_replan_steps(value)

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_parity_ignores_unused_padding_but_not_decoded_tokens(self):
        actions = np.arange(7, dtype=np.float32)
        instrumented = np.asarray([21, PALIGEMMA_EOS_TOKEN, 0, 0])
        reference = np.asarray([21, PALIGEMMA_EOS_TOKEN, 99, 99])
        record = parity_record(
            instrumented,
            2,
            actions,
            reference,
            actions + PARITY_ACTION_ATOL / 2,
        )
        self.assertTrue(record["passed"])
        self.assertFalse(record["unused_padding_exact"])

        reference[0] = 22
        self.assertFalse(
            parity_record(instrumented, 2, actions, reference, actions)["passed"]
        )

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_bfloat16_bits_recover_the_represented_values(self):
        values = np.asarray([0x3F80, 0xC020, 0x0000], dtype=np.uint16)
        np.testing.assert_array_equal(
            bfloat16_bits_to_float32(values),
            np.asarray([1.0, -2.5, 0.0], dtype=np.float32),
        )

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_decode_diagnostics_preserve_the_fast_fallback_signal(self):
        def decode(_state, _tokens):
            print("Error decoding tokens: cannot reshape array of size 68 into shape (7)")
            print("Tokens: [1, 2]")
            return np.ones((10, 7), dtype=np.float32)

        actions, record = decode_with_diagnostics(decode, None, [1, 2])
        self.assertEqual(record["status"], "tokenizer_fallback")
        self.assertIn("size 68", record["captured_stdout"])
        self.assertFalse(record["policy_visible_actions_all_zero"])
        np.testing.assert_array_equal(actions, np.ones((10, 7)))

    def test_decode_audit_classification_requires_matching_tokens(self):
        self.assertEqual(
            decode_audit_classification(
                True, "tokenizer_fallback", "tokenizer_fallback"
            ),
            "same_tokens_same_tokenizer_fallback",
        )
        self.assertEqual(
            decode_audit_classification(
                False, "tokenizer_fallback", "tokenizer_fallback"
            ),
            "sampler_tokens_differ",
        )

    def test_decode_audit_response_is_bound_to_the_requested_decision(self):
        response = _decode_audit_response(4)
        self.assertEqual(
            validate_decode_audit_response(response, decision_index=4)[
                "classification"
            ],
            "same_tokens_same_tokenizer_fallback",
        )
        with self.assertRaisesRegex(ValueError, "decision_id"):
            validate_decode_audit_response(response, decision_index=5)

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_response_validation_raises_only_the_parity_error_for_failed_gate(self):
        response = _response(2, compare_reference=True)
        arrays = validate_policy_response(
            response, decision_index=2, compare_reference=True
        )
        self.assertEqual(arrays["actions"].shape, (10, 7))

        response["reference_comparison"]["passed"] = False
        with self.assertRaises(ExactParityError):
            validate_policy_response(
                response, decision_index=2, compare_reference=True
            )

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_six_step_rollout_records_two_policy_decisions(self):
        trial = Trial(0, 0)
        environment = FakeEnvironment()
        client = FakeClient()

        def policy_input(observation, task_description):
            value = int(np.asarray(observation["joint"]).reshape(-1)[0])
            image = np.full((2, 2, 3), value, dtype=np.uint8)
            element = {
                "observation/image": image,
                "observation/wrist_image": image,
                "observation/state": np.arange(8, dtype=np.float32),
                "prompt": task_description,
            }
            return element, image, image

        with tempfile.TemporaryDirectory() as directory:
            environment_patch = mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._get_libero_env",
                return_value=(environment, "move the object"),
            )
            input_patch = mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._policy_input",
                side_effect=policy_input,
            )
            with environment_patch, input_patch:
                result = run_trial(
                    RolloutConfig(
                        output_dir=Path(directory),
                        task_suite="libero_10",
                        base_seed=7,
                        wait_steps=0,
                        replan_steps=5,
                        save_video=False,
                        compare_reference_first_decision=True,
                    ),
                    client,
                    trial,
                    SimpleNamespace(),
                    np.asarray([1.0], dtype=np.float32),
                    {"server_metadata_sha256": "server"},
                )

        self.assertTrue(environment.closed)
        self.assertTrue(result["success"])
        self.assertEqual(result["environment_steps"], 6)
        self.assertEqual(result["model_decisions"], 2)
        self.assertEqual(len(client.requests), 2)
        self.assertTrue(client.requests[0]["compare_reference"])
        self.assertFalse(client.requests[1]["compare_reference"])
        self.assertFalse(client.requests[0]["audit_malformed_decode"])
        self.assertFalse(client.requests[1]["audit_malformed_decode"])
        self.assertFalse(client.requests[0]["record_decode_fallback"])
        self.assertFalse(client.requests[1]["record_decode_fallback"])

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_fallback_continuation_records_and_executes_the_returned_actions(self):
        trial = Trial(0, 0)
        environment = FakeEnvironment()
        client = FallbackClient()

        def policy_input(observation, task_description):
            value = int(np.asarray(observation["joint"]).reshape(-1)[0])
            image = np.full((2, 2, 3), value, dtype=np.uint8)
            return (
                {
                    "observation/image": image,
                    "observation/wrist_image": image,
                    "observation/state": np.arange(8, dtype=np.float32),
                    "prompt": task_description,
                },
                image,
                image,
            )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            environment_patch = mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._get_libero_env",
                return_value=(environment, "move the object"),
            )
            input_patch = mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._policy_input",
                side_effect=policy_input,
            )
            with environment_patch, input_patch:
                result = run_trial(
                    RolloutConfig(
                        output_dir=output_dir,
                        task_suite="libero_10",
                        base_seed=7,
                        wait_steps=0,
                        replan_steps=5,
                        save_video=False,
                        record_decode_fallbacks=True,
                    ),
                    client,
                    trial,
                    SimpleNamespace(),
                    np.asarray([1.0], dtype=np.float32),
                    {"server_metadata_sha256": "server"},
                )
            with next(output_dir.glob("*.pkl")).open("rb") as file:
                artifact = pickle.load(file)

        self.assertTrue(environment.closed)
        self.assertEqual(environment.steps, 6)
        self.assertEqual(result["condition"], FALLBACK_CONTINUATION_CONDITION)
        self.assertEqual(result["decode_fallbacks"], 1)
        self.assertEqual(result["decode_fallback_decision_indices"], [1])
        self.assertEqual(
            artifact["decisions"]["decode_diagnostics"][1]["status"],
            "tokenizer_fallback",
        )
        self.assertEqual(artifact["decode_fallback_decision_indices"], [1])
        self.assertTrue(
            all(
                request["record_decode_fallback"]
                for request in client.requests
            )
        )

    @unittest.skipIf(np is None, "NumPy is installed in the pi0-FAST runtime")
    def test_decode_audit_stops_before_executing_the_fallback_action(self):
        trial = Trial(0, 0)
        environment = FakeEnvironment()
        client = AuditClient()

        def policy_input(observation, task_description):
            value = int(np.asarray(observation["joint"]).reshape(-1)[0])
            image = np.full((2, 2, 3), value, dtype=np.uint8)
            return (
                {
                    "observation/image": image,
                    "observation/wrist_image": image,
                    "observation/state": np.arange(8, dtype=np.float32),
                    "prompt": task_description,
                },
                image,
                image,
            )

        with tempfile.TemporaryDirectory() as directory:
            environment_patch = mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._get_libero_env",
                return_value=(environment, "move the object"),
            )
            input_patch = mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._policy_input",
                side_effect=policy_input,
            )
            with environment_patch, input_patch:
                with self.assertRaises(DecodeAuditComplete) as raised:
                    run_trial(
                        RolloutConfig(
                            output_dir=Path(directory),
                            task_suite="libero_10",
                            base_seed=7,
                            wait_steps=0,
                            replan_steps=5,
                            save_video=False,
                            audit_malformed_decodes=True,
                        ),
                        client,
                        trial,
                        SimpleNamespace(),
                        np.asarray([1.0], dtype=np.float32),
                        {"server_metadata_sha256": "server"},
                    )

        self.assertTrue(environment.closed)
        self.assertEqual(environment.steps, 5)
        self.assertEqual(
            raised.exception.record["rollout_context"]["environment_step"], 5
        )
        self.assertEqual(
            raised.exception.record["classification"],
            "same_tokens_same_tokenizer_fallback",
        )


if __name__ == "__main__":
    unittest.main()
