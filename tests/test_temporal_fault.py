import unittest

from embodied_silent_failures.temporal_fault import (
    TemporalReplacementInjector,
    TemporalReplacementSpec,
    replace_at_port,
    value_at_port,
)


class TemporalPortTests(unittest.TestCase):
    def test_nested_port_replacement_preserves_siblings_and_original(self) -> None:
        original = {"hidden_states": ("zero", {"value": "one"}), "logits": "two"}

        changed = replace_at_port(
            original, "value.hidden_states[1].value", "replacement"
        )

        self.assertEqual(
            value_at_port(changed, "value.hidden_states[1].value"), "replacement"
        )
        self.assertEqual(value_at_port(changed, "value.logits"), "two")
        self.assertEqual(
            value_at_port(original, "value.hidden_states[1].value"), "one"
        )

    def test_spec_requires_the_declared_temporal_relation(self) -> None:
        identity = {
            "kind": "module_output",
            "module_path": "policy.layer",
            "module_call_index": 0,
            "output_port": "value",
        }
        with self.assertRaises(ValueError):
            TemporalReplacementSpec("site", identity, 4, 2)
        with self.assertRaises(ValueError):
            TemporalReplacementSpec(
                "site", identity, 4, 3, mode="current_value_canary"
            )


class TemporalInjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as error:
            raise unittest.SkipTest("PyTorch and NumPy are required") from error
        cls.np = np
        cls.torch = torch

    def _model(self):
        class Model(self.torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = TemporalInjectorTests.torch.nn.Identity()

            def forward(self, value):
                return self.layer(value)

        return Model()

    def _identity(self):
        return {
            "kind": "module_output",
            "module_path": "policy.layer",
            "module_call_index": 0,
            "output_port": "value",
        }

    def test_module_output_is_replaced_with_the_preceding_step(self) -> None:
        model = self._model()
        injector = TemporalReplacementInjector(
            self.torch,
            self.np,
            TemporalReplacementSpec("site", self._identity(), 1, 0),
        )
        injector.install(model)
        injector.begin_trial(9)
        with injector.inference(0):
            prior = model(self.torch.tensor([1.0, 2.0]))
        with injector.inference(1):
            replaced = model(self.torch.tensor([4.0, 8.0]))
        record = injector.require_injected()
        injector.close()

        self.assertTrue(self.torch.equal(prior, replaced))
        self.assertFalse(record["comparison"]["exact_equal"])
        self.assertEqual(record["comparison"]["changed_element_count"], 2)

    def test_current_value_canary_exercises_the_replacement_path(self) -> None:
        model = self._model()
        injector = TemporalReplacementInjector(
            self.torch,
            self.np,
            TemporalReplacementSpec(
                "site",
                self._identity(),
                0,
                0,
                mode="current_value_canary",
            ),
        )
        injector.install(model)
        injector.begin_trial(9)
        value = self.torch.tensor([3.0, 5.0])
        with injector.inference(0):
            observed = model(value)
        record = injector.require_injected()
        injector.close()

        self.assertTrue(self.torch.equal(value, observed))
        self.assertIsNot(value, observed)
        self.assertTrue(record["comparison"]["exact_equal"])

    def test_declared_boundary_uses_the_same_port_mechanics(self) -> None:
        identity = {
            "kind": "declared_runtime_boundary",
            "event_name": "openvla.processor_output",
            "event_call_index": 0,
            "output_port": "value.pixel_values",
        }
        injector = TemporalReplacementInjector(
            self.torch,
            self.np,
            TemporalReplacementSpec("pixels", identity, 2, 1),
        )
        injector.begin_trial(3)
        prior = {"pixel_values": self.torch.tensor([1.0]), "input_ids": self.torch.tensor([7])}
        current = {"pixel_values": self.torch.tensor([9.0]), "input_ids": self.torch.tensor([8])}
        injector.boundary("openvla.processor_output", prior, policy_step=1)
        observed = injector.boundary(
            "openvla.processor_output", current, policy_step=2
        )

        self.assertEqual(observed["pixel_values"].item(), 1.0)
        self.assertEqual(observed["input_ids"].item(), 8)


if __name__ == "__main__":
    unittest.main()
