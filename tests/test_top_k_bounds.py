import unittest
from core.vector_store import _validate_k, MAX_TOP_K
from core.rag_engine import _validate_top_k


class TestTopKValidation(unittest.TestCase):

    def test_valid_top_k_values(self):
        """1 se MAX_TOP_K tak valid values pass honi chahiyein."""
        for val in [1, 4, 10, 20]:
            self.assertEqual(_validate_k(val), val)
            self.assertEqual(_validate_top_k(val), val)

    def test_top_k_exceeds_upper_bound(self):
        """MAX_TOP_K se barhi values par ValueError raise hona chahiye."""
        invalid_values = [MAX_TOP_K + 1, 50, 1000, 1_000_000]
        for val in invalid_values:
            with self.assertRaises(ValueError):
                _validate_k(val)
            with self.assertRaises(ValueError):
                _validate_top_k(val)

    def test_top_k_below_lower_bound(self):
        """0 ya negative values par ValueError raise hona chahiye."""
        for val in [0, -1, -100]:
            with self.assertRaises(ValueError):
                _validate_k(val)
            with self.assertRaises(ValueError):
                _validate_top_k(val)

    def test_top_k_invalid_types(self):
        """String, float, bool par TypeError raise hona chahiye."""
        invalid_inputs = [True, False, "10", 3.14, None]
        for val in invalid_inputs:
            with self.assertRaises(TypeError):
                _validate_k(val)
            with self.assertRaises(TypeError):
                _validate_top_k(val)


if __name__ == "__main__":
    unittest.main()