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
                self.num_heads = 1
                self.head_dim = 4
                self.q_proj = torch.nn.Identity()
                self.k_proj = torch.nn.Identity()
                self.v_proj = torch.nn.Identity()
                self.rotary_emb = RotaryEmbedding()

            def forward(
                self,
                value,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=True,
                cache_position=None,
            ):
                query = self.q_proj(value)
                self.rotary_emb(query, position_ids)
                key = self.k_proj(value).unsqueeze(1)
                cache_value = self.v_proj(value).unsqueeze(1)
                past_key_value.update(key, cache_value, self.layer_index)
                return value, None, past_key_value

        class RotaryEmbedding(torch.nn.Module):
            def forward(self, value, position_ids):
                shape = (value.shape[0], value.shape[1], value.shape[-1])
                return torch.ones(shape), torch.zeros(shape)

        class Layer(torch.nn.Module):
            def __init__(self, layer_index, offset):
                super().__init__()
                self.offset = offset
                self.self_attn = Attention(layer_index)
                self.post_attention_layernorm = torch.nn.Identity()

            def forward(self, value, cache, position_ids, cache_position):
                attended = self.self_attn(
                    value,
                    position_ids=position_ids,
                    past_key_value=cache,
                    cache_position=cache_position,
                )[0]
                residual = self.post_attention_layernorm(attended)
                return (residual + self.offset, "unchanged")

        class LanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [Layer(index, index + 1) for index in range(32)]
                )

            def forward(self, value, cache, position_ids, cache_position):
                for layer in self.model.layers:
                    value = layer(value, cache, position_ids, cache_position)[0]
                return value

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = LanguageModel()

            def new_cache(self):
                return DynamicCache()

            def forward(
                self,
                value,
                cache,
                position,
                past_key_values=None,
                pixel_values=None,
            ):
                position_ids = torch.arange(
                    position,
                    position + value.shape[1],
                    device=value.device,
                ).unsqueeze(0)
                return self.language_model(
                    value,
                    cache,
                    position_ids,
                    position_ids[0],
                )

        return Model()

    def test_call_zero_replaces_only_the_final_sequence_vector(self) -> None:
        torch = self.torch
        model = self._model()
        injector = LanguageBlockInjector(torch)
        injector.install(model)
        source = torch.full((1, 1, 4), 50.0)
        value = torch.zeros((1, 3, 4))
        cache = model.new_cache()

        with injector.inference(0, replacement_layer=0, sources={0: source}):
            output = model(value, cache, position=0)
        trace = injector.last_trace
        injector.close()

        self.assertIsNotNone(trace)
        first_layer_effect = output - sum(range(2, 33))
        self.assertTrue(torch.equal(first_layer_effect[:, :2, :], torch.ones(1, 2, 4)))
        self.assertTrue(torch.equal(trace.block_values[0], source))
        self.assertTrue(torch.equal(trace.block_values_by_call[0][0], source))
        self.assertEqual(trace.anomalies, ("model observed 1 generation calls",))

    def test_later_generation_call_uses_the_same_vector_shape(self) -> None:
        torch = self.torch
        model = self._model()
        injector = LanguageBlockInjector(torch)
        injector.install(model)
        source = torch.full((1, 1, 4), 20.0)
        cache = model.new_cache()

        with injector.inference(4, replacement_layer=10, sources={10: source}):
            for position in range(7):
                model(torch.zeros((1, 1, 4)), cache, position)
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
        cache = model.new_cache()

        with injector.inference(
            4,
            replacement_layer=10,
            sources={10: residual_source},
            cache_replacement_layers=frozenset({10}),
            cache_sources=cache_source,
        ):
            for position in range(7):
                model(torch.zeros((1, 1, 4)), cache, position)
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

    def test_internal_capture_records_attention_conditioning_and_cuts(self) -> None:
        torch = self.torch
        model = self._model()
        injector = LanguageBlockInjector(torch)
        injector.install(model)
        cache = model.new_cache()

        with injector.inference(
            3,
            capture_internal_state=True,
            capture_context_state=True,
        ):
            for position in range(7):
                legacy_cache = (
                    tuple(zip(cache.key_cache, cache.value_cache, strict=True))
                    if position == 1
                    else None
                )
                model(
                    torch.zeros((1, 1, 4)),
                    cache,
                    position,
                    past_key_values=legacy_cache,
                    pixel_values=torch.zeros((1, 3, 2, 2)),
                )
        trace = injector.last_trace
        injector.close()

        self.assertEqual(set(trace.decoder_position_ids_by_call), set(range(7)))
        self.assertEqual(set(trace.decoder_cache_positions_by_call), set(range(7)))
        self.assertEqual(set(trace.block_inputs_by_call[0]), set(range(7)))
        self.assertEqual(set(trace.post_attention_residuals_by_call[0]), set(range(7)))
        self.assertEqual(set(trace.attention_queries_by_call[0]), set(range(7)))
        self.assertEqual(set(trace.prompt_cache), set(range(32)))
        self.assertEqual(trace.prompt_cache_format, "transformers_legacy_tuple")
        self.assertEqual(trace.anomalies, ())

    def test_tensor_change_reports_masking_without_thresholds(self) -> None:
        value = self.torch.tensor([[[1.0, 2.0]]])

        record = tensor_change(self.torch, value, value.clone())

        self.assertTrue(record["exact_equal"])
        self.assertEqual(record["changed_element_count"], 0)


if __name__ == "__main__":
    unittest.main()
