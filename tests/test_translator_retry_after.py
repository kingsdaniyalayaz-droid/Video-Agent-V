import unittest
from unittest.mock import MagicMock, patch
from core.translator import extract_retry_after, _invoke_translation_with_retry, RETRY_AFTER_CEILING


class TestTranslatorRetryAfter(unittest.TestCase):

    def test_extract_retry_after_valid(self):
        """Valid Retry-After correctly extracted and clamped."""
        exc = Exception("Rate limit")
        exc.retry_after = 5.0
        self.assertEqual(extract_retry_after(exc), 5.0)

        # Clamping check
        exc.retry_after = 3600.0
        self.assertEqual(extract_retry_after(exc), RETRY_AFTER_CEILING)

    def test_extract_retry_after_malformed(self):
        """Invalid/negative values return None or 0.0."""
        exc = Exception("Rate limit")
        for bad_val in [-5, "invalid", None, float("nan")]:
            exc.retry_after = bad_val
            self.assertIn(extract_retry_after(exc), (None, 0.0, 0))

    @patch("threading.Event.wait", return_value=False)
    def test_invoke_uses_server_delay_over_smaller_exp_delay(self, mock_wait):
        """Attempt 1 par agar exp_delay 1s ho aur server 5s maange to 5.0s wait hona chahiye."""
        mock_chain = MagicMock()
        rate_limit_exc = Exception("429 Too Many Requests")
        rate_limit_exc.retry_after = 5.0
        
        mock_response = MagicMock()
        mock_response.content = "Success"
        mock_chain.invoke.side_effect = [rate_limit_exc, mock_response]

        _invoke_translation_with_retry(
            mock_chain,
            {"text": "test text"},
            chunk_index=1,
            total_chunks=1,
        )
        self.assertTrue(mock_wait.called)
        # Verify that timeout passed to wait was 5.0
        _, kwargs = mock_wait.call_args
        self.assertEqual(kwargs.get("timeout"), 5.0)


if __name__ == "__main__":
    unittest.main()
