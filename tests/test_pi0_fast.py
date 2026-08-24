import ast
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
    REQUEST_KEY,
    evidence_metadata,
    parity_record,
)
from embodied_silent_failures.pi0_fast_rollout import (
    EXACT_PARITY_EXIT_CODE,
    ExactParityError,
    RolloutConfig,
    bfloat16_bits_to_float32,
    run_trial,
    validate_policy_response,
)
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.run_pi0_fast import _trial_plan


def _response(decision_id, compare_reference=False):
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
    return response


class FakeClient:
    def __init__(self):
        self.requests = []

    def infer(self, element):
        request = element[REQUEST_KEY]
        self.requests.append(request)
        return _response(request["decision_id"], request["compare_reference"])


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

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._get_libero_env",
                return_value=(environment, "move the object"),
            ),
            mock.patch(
                "embodied_silent_failures.pi0_fast_rollout._policy_input",
                side_effect=policy_input,
            ),
        ):
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


if __name__ == "__main__":
    unittest.main()
