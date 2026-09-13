import unittest
from unittest.mock import MagicMock, patch
import core.transcriber as t


class TestTranscriberRuntimeConfig(unittest.TestCase):

    def setUp(self):
        t.unload_model()

    def tearDown(self):
        t.unload_model()

    @patch("whisper.load_model")
    def test_model_cache_reuses_identical_config(self, mock_load):
        """Same config par model reload nahi hona chahiye."""
        mock_instance = MagicMock()
        mock_load.return_value = mock_instance

        m1 = t.load_model("small", "cpu")
        m2 = t.load_model("small", "cpu")

        self.assertIs(m1, m2)
        mock_load.assert_called_once_with("small", device="cpu")

    @patch("whisper.load_model")
    def test_config_change_triggers_reload(self, mock_load):
        """Model ya device badalne par purana model unload ho kar naya load ho."""
        mock_model_cpu = MagicMock()
        mock_model_gpu = MagicMock()
        mock_load.side_effect = [mock_model_cpu, mock_model_gpu]

        # 1. Load on CPU
        m1 = t.load_model("small", "cpu")
        self.assertEqual(t._cached_model_key, ("small", "cpu"))

        # 2. Load on CUDA (Config changed)
        m2 = t.load_model("base", "cuda")
        self.assertEqual(t._cached_model_key, ("base", "cuda"))
        self.assertIsNot(m1, m2)

        self.assertEqual(mock_load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
