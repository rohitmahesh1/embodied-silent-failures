import ast
import json
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

from embodied_silent_failures.artifacts import prepare_trial
from embodied_silent_failures.pi05_contract import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    DIFFUSION_STEPS,
    PROTOCOL_VERSION,
    decision_for_step,
    decision_noise_seed,
    validate_replan_steps,
)
from embodied_silent_failures.pi05_policy import REQUEST_KEY
from embodied_silent_failures.pi05_safe_data import reduce_pre_velocity
from embodied_silent_failures.pi05_rollout import (
    RolloutConfig,
    run_trial,
    validate_policy_response,
)
from embodied_silent_failures.run_pi05 import running_status
from embodied_silent_failures.plan import Trial
from embodied_silent_failures.pi05_supervisor import (
    InfrastructureError,
    execute_resilient_plan,
)
from embodied_silent_failures.train_pi05_safe import (
    _functional_band_row,
    published_configuration,
)


def _response(decision_id, noise_seed, compare_reference=False):
    response = {
        "actions": np.arange(ACTION_HORIZON * 7, dtype=np.float32).reshape(
            ACTION_HORIZON, 7
        ),
        "raw_actions": np.zeros((ACTION_HORIZON, ACTION_DIMENSION), dtype=np.float32),
        "evidence": {
            "pre_velocity": np.zeros(
                (DIFFUSION_STEPS, ACTION_HORIZON, 4), dtype=np.float32
            ),
            "sampling_noise": np.zeros(
                (ACTION_HORIZON, ACTION_DIMENSION), dtype=np.float32
            ),
            "completed_diffusion_steps": DIFFUSION_STEPS,
            "noise_seed": noise_seed,
            "decision_id": decision_id,
            "protocol_version": PROTOCOL_VERSION,
        },
        "policy_timing": {"infer_ms": 1.0},
        "server_timing": {"infer_ms": 1.1},
    }
    if compare_reference:
        response["reference_comparison"] = {
            "maximum_absolute_raw_action_error": 0.0,
            "atol": 1e-6,
            "passed": True,
        }
    return response


class FakeClient:
    def __init__(self):
        self.requests = []

    def infer(self, element):
        request = element[REQUEST_KEY]
        self.requests.append(request)
        return _response(
            request["decision_id"],
            request["noise_seed"],
            request["compare_reference"],
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


class Pi05ContractTests(unittest.TestCase):
    def test_rollout_client_modules_parse_as_python_38(self):
        project_root = Path(__file__).resolve().parents[1]
        for name in (
            "artifacts.py",
            "pi05_contract.py",
            "pi05_pair.py",
            "pi05_policy.py",
            "pi05_rollout.py",
            "pi05_safe_data.py",
            "pi05_stale_manifest.py",
            "plan.py",
            "provenance.py",
            "run_pi05_pair_trial.py",
            "run_pi05_pairs.py",
            "run_pi05_trial.py",
            "score_pi05_safe.py",
            "train_pi05_safe.py",
        ):
            path = project_root / "embodied_silent_failures" / name
            with self.subTest(name=name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 8),
                )

    def test_noise_identity_is_stable_and_decision_specific(self):
        trial = Trial(3, 8)
        self.assertEqual(
            decision_noise_seed(7, trial, 2),
            decision_noise_seed(7, trial, 2),
        )
        self.assertNotEqual(
            decision_noise_seed(7, trial, 2),
            decision_noise_seed(7, trial, 3),
        )
        self.assertNotEqual(
            decision_noise_seed(7, trial, 2),
            decision_noise_seed(7, Trial(3, 9), 2),
        )

    def test_native_replanning_maps_steps_to_decisions_and_offsets(self):
        self.assertEqual(
            [decision_for_step(step, 5) for step in range(7)],
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1)],
        )
        for value in (0, 11):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_replan_steps(value)

    @unittest.skipIf(np is None, "NumPy is installed in the pi0.5 runtime")
    def test_safe_reduction_uses_first_horizon_and_final_diffusion(self):
        values = np.arange(3 * 4 * 5 * 2, dtype=np.float32).reshape(3, 4, 5, 2)

        selected = reduce_pre_velocity(values, np)

        np.testing.assert_array_equal(selected, values[:, -1, 0, :])

    def test_safe_configuration_matches_published_pi0_mlp(self):
        config = published_configuration()

        self.assertEqual(config["dataset"]["horizon_idx_rel"], 0.0)
        self.assertEqual(config["dataset"]["diff_idx_rel"], 1.0)
        self.assertIsNone(config["dataset"]["data_path_unseen"])
        self.assertEqual(config["model"]["name"], "indep")
        self.assertEqual(config["model"]["n_layers"], 2)
        self.assertEqual(config["model"]["hidden_dim"], 256)
        self.assertEqual(config["model"]["lr"], 3e-5)
        self.assertEqual(config["model"]["lambda_reg"], 1e-3)

    @unittest.skipIf(np is None, "NumPy is installed in the pi0.5 runtime")
    def test_safe_functional_band_removes_only_the_regression_axis(self):
        values = np.arange(6, dtype=np.float32)[None]

        np.testing.assert_array_equal(
            _functional_band_row(values, 6, np), values[0]
        )
        with self.assertRaisesRegex(ValueError, "unexpected band shape"):
            _functional_band_row(np.zeros((2, 6), dtype=np.float32), 6, np)

    @unittest.skipIf(np is None, "NumPy is installed in the pi0.5 runtime")
    def test_response_validation_requires_exact_evidence_identity(self):
        response = _response(2, 17, compare_reference=True)
        arrays = validate_policy_response(
            response,
            decision_index=2,
            noise_seed=17,
            compare_reference=True,
        )
        self.assertEqual(arrays["pre_velocity"].shape, (10, 10, 4))

        response["evidence"]["decision_id"] = 3
        with self.assertRaises(ValueError):
            validate_policy_response(
                response,
                decision_index=2,
                noise_seed=17,
                compare_reference=True,
            )

    @unittest.skipIf(np is None, "NumPy is installed in the pi0.5 runtime")
    def test_six_step_rollout_records_two_native_policy_decisions(self):
        trial = Trial(0, 0)
        environment = FakeEnvironment()
        client = FakeClient()
        heartbeats = []

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
                "embodied_silent_failures.pi05_rollout._get_libero_env",
                return_value=(environment, "move the object"),
            ),
            mock.patch(
                "embodied_silent_failures.pi05_rollout._policy_input",
                side_effect=policy_input,
            ),
        ):
            output_dir = Path(directory)
            result = run_trial(
                RolloutConfig(
                    output_dir=output_dir,
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
                heartbeats.append,
            )

            self.assertTrue(environment.closed)
            self.assertTrue(result["success"])
            self.assertEqual(result["environment_steps"], 6)
            self.assertEqual(result["model_decisions"], 2)
            self.assertEqual(len(client.requests), 2)
            self.assertTrue(client.requests[0]["compare_reference"])
            self.assertFalse(client.requests[1]["compare_reference"])
            self.assertEqual(prepare_trial(output_dir, trial, True), "complete")
            pickle_path = output_dir / result["files"]["pickle"]
            with pickle_path.open("rb") as file:
                record = pickle.load(file)

        self.assertEqual(record["decisions"]["action_chunks"].shape, (2, 10, 7))
        self.assertEqual(record["decisions"]["pre_velocity"].shape, (2, 10, 10, 4))
        np.testing.assert_array_equal(
            record["environment"]["decision_indices"], [0, 0, 0, 0, 0, 1]
        )
        np.testing.assert_array_equal(
            record["environment"]["chunk_offsets"], [0, 1, 2, 3, 4, 0]
        )
        self.assertEqual(heartbeats[-1]["state"], "complete")


class Pi05SupervisorTests(unittest.TestCase):
    def test_running_status_does_not_confuse_trial_and_campaign_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "task0--ep0.complete.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            status = running_status(
                output_dir,
                planned_trials=20,
                campaign_started="start",
                update={"state": "complete", "trial": Trial(0, 0), "index": 1},
            )

        self.assertEqual(status["state"], "running")
        self.assertEqual(status["trial_progress"], "complete")
        self.assertEqual(status["trial"], {"task_id": 0, "episode_index": 0})
        self.assertEqual(status["completed_trials"], 1)

    def test_one_unresolved_trial_does_not_stop_later_trials(self):
        plan = [Trial(0, 0), Trial(0, 1), Trial(0, 2)]
        calls = []
        unresolved = []

        def attempt(trial, number):
            calls.append((trial, number))
            if trial == plan[0] and number == 1:
                raise RuntimeError("transient")
            if trial == plan[1]:
                raise RuntimeError("persistent")
            return {"status": "complete"}

        result = execute_resilient_plan(
            plan,
            max_attempts=2,
            already_complete=lambda trial: False,
            run_attempt=attempt,
            mark_unresolved=lambda trial, errors: unresolved.append((trial, errors)),
        )

        self.assertEqual(result["complete"], [plan[0], plan[2]])
        self.assertEqual(result["unresolved"], [plan[1]])
        self.assertEqual(unresolved[0][0], plan[1])
        self.assertIn((plan[2], 1), calls)

    def test_infrastructure_failure_stops_before_more_trials_are_launched(self):
        plan = [Trial(0, 0), Trial(0, 1)]
        calls = []

        def attempt(trial, number):
            calls.append((trial, number))
            raise InfrastructureError("policy server unavailable")

        with self.assertRaises(InfrastructureError):
            execute_resilient_plan(
                plan,
                max_attempts=2,
                already_complete=lambda trial: False,
                run_attempt=attempt,
                mark_unresolved=lambda trial, errors: None,
            )
        self.assertEqual(calls, [(plan[0], 1)])


if __name__ == "__main__":
    unittest.main()
