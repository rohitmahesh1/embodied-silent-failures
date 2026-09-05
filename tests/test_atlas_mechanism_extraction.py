import unittest

import numpy as np

from embodied_silent_failures.atlas_mechanism_extraction import vector_difference


class AtlasMechanismExtractionTests(unittest.TestCase):
    def test_vector_difference_uses_symmetric_scale(self) -> None:
        result = vector_difference(np, [1.0, 0.0], [0.0, 1.0])

        self.assertFalse(result["exact_equal"])
        self.assertAlmostEqual(result["difference_l2"], 2**0.5)
        self.assertAlmostEqual(
            result["symmetric_normalized_difference_l2"], 2**0.5
        )

    def test_zero_vectors_have_finite_normalized_difference(self) -> None:
        result = vector_difference(np, [0.0, 0.0], [0.0, 0.0])

        self.assertTrue(result["exact_equal"])
        self.assertEqual(result["symmetric_normalized_difference_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
