import unittest
import threading
from unittest.mock import MagicMock, patch
import core.transcriber as t


class TestTranscriberThreadSafety(unittest.TestCase):

    def setUp(self):
        t._model = None

    @patch("whisper.load_model")
    def test_concurrent_load_model_is_called_exactly_once(self, mock_load):
        """10 concurrent threads call karein to bhi model sirf 1 dafa load hona chahiye."""
        mock_instance = MagicMock()
        mock_load.return_value = mock_instance

        results = []
        errors = []

        def worker():
            try:
                m = t.load_model()
                results.append(m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        for m in results:
            self.assertIs(m, mock_instance)
        mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
