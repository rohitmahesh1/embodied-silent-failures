import unittest

from embodied_silent_failures.language_fault import (
    LanguageBlockInjector,
    tensor_change,
)


class LanguageFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is required") from error
        cls.torch = torch

    def _model(self):
        torch = self.torch

        class DynamicCache:
            def __init__(self):
                self.key_cache = []
                self.value_cache = []

            def update(self, key, value, layer_index):
                if layer_index == len(self.key_cache):
                    self.key_cache.append(key)
                    self.value_cache.append(value)
                else:
                    self.key_cache[layer_index] = torch.cat(
                        (self.key_cache[layer_index], key), dim=-2
                    )
                    self.value_cache[layer_index] = torch.cat(
                        (self.value_cache[layer_index], value), dim=-2
                    )
                return self.key_cache[layer_index], self.value_cache[layer_index]

        class Attention(torch.nn.Module):
            def __init__(self, layer_index):
                super().__init__()
                self.layer_index = layer_index
                self.k_proj = torch.nn.Identity()
                self.v_proj = torch.nn.Identity()

            def forward(self, value, past_key_value):
                key = self.k_proj(value).unsqueeze(1)
                cache_value = self.v_proj(value).unsqueeze(1)
                past_key_value.update(key, cache_value, self.layer_index)
                return value, None, past_key_value

        class Layer(torch.nn.Module):
            def __init__(self, layer_index, offset):
                super().__init__()
                self.offset = offset
                self.self_attn = Attention(layer_index)

            def forward(self, value, cache):
                attended = self.self_attn(value, cache)[0]
                return (attended + self.offset, "unchanged")

        class LanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [Layer(index, index + 1) for index in range(32)]
                )

            def forward(self, value, cache):
                for layer in self.model.layers:
                    value = layer(value, cache)[0]
                return value

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = LanguageModel()

            def forward(self, value, calls=7):
                outputs = []
                cache = DynamicCache()
                for _ in range(calls):
                    outputs.append(self.language_model(value, cache))
                return outputs

        return Model()

    def test_call_zero_replaces_only_the_final_sequence_vector(self) -> None:
        torch = self.torch
        model = self._model()
        injector = LanguageBlockInjector(torch)
        injector.install(model)
        source = torch.full((1, 1, 4), 50.0)
        value = torch.zeros((1, 3, 4))

        with injector.inference(0, replacement_layer=0, sources={0: source}):
            outputs = model(value, calls=1)
        trace = injector.last_trace
        injector.close()

        self.assertIsNotNone(trace)
        first_layer_effect = outputs[0] - sum(range(2, 33))
        self.assertTrue(torch.equal(first_layer_effect[:, :2, :], torch.ones(1, 2, 4)))
        self.assertTrue(torch.equal(trace.block_values[0], source))
        self.assertTrue(torch.equal(trace.block_values_by_call[0][0], source))
        self.assertEqual(trace.anomalies, ())

    def test_later_generation_call_uses_the_same_vector_shape(self) -> None:
        torch = self.torch
        model = self._model()
        injector = LanguageBlockInjector(torch)
        injector.install(model)
        source = torch.full((1, 1, 4), 20.0)

        with injector.inference(4, replacement_layer=10, sources={10: source}):
            model(torch.zeros((1, 1, 4)), calls=7)
        trace = injector.last_trace
        injector.close()

        self.assertEqual(trace.block_values[10].shape, source.shape)
        self.assertEqual(set(trace.block_values_by_call[10]), set(range(7)))
        self.assertEqual(set(trace.cache_values_by_call[10]["key"]), set(range(7)))
        self.assertEqual(set(trace.cache_values_by_call[10]["value"]), set(range(7)))
        self.assertEqual(
            trace.sequence_lengths_by_call[10],
            {index: 1 for index in range(7)},
        )
        self.assertEqual(trace.call_counts, {index: 7 for index in range(32)})
        self.assertEqual(trace.anomalies, ())

    def test_cache_replay_replaces_the_exact_current_entry(self) -> None:
        torch = self.torch
        model = self._model()
        injector = LanguageBlockInjector(torch)
        injector.install(model)
        residual_source = torch.full((1, 1, 4), 20.0)
        cache_source = {
            10: {
                "key": torch.full((1, 1, 1, 4), 30.0),
                "value": torch.full((1, 1, 1, 4), 40.0),
            }
        }

        with injector.inference(
            4,
            replacement_layer=10,
            sources={10: residual_source},
            cache_replacement_layers=frozenset({10}),
            cache_sources=cache_source,
        ):
            model(torch.zeros((1, 1, 4)), calls=7)
        trace = injector.last_trace
        injector.close()

        self.assertTrue(
            torch.equal(
                trace.cache_values_by_call[10]["key"][4],
                cache_source[10]["key"],
            )
        )
        self.assertTrue(
            torch.equal(
                trace.cache_values_by_call[10]["value"][4],
                cache_source[10]["value"],
            )
        )

    def test_tensor_change_reports_masking_without_thresholds(self) -> None:
        value = self.torch.tensor([[[1.0, 2.0]]])

        record = tensor_change(self.torch, value, value.clone())

        self.assertTrue(record["exact_equal"])
        self.assertEqual(record["changed_element_count"], 0)


if __name__ == "__main__":
    unittest.main()
