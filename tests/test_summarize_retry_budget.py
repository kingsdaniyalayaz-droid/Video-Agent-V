import unittest
from unittest.mock import MagicMock, patch
from core.summarize import (
    _extract_retry_delay,
    _invoke_with_retry,
    MAX_RETRY_AFTER_DELAY,
)


class TestSummarizeRetryBudget(unittest.TestCase):

    def test_excessive_retry_after_is_clamped(self):
        """1 ghante (3600s) jaisa bada delay MAX_RETRY_AFTER_DELAY par clamp hona chahiye."""
        exc = Exception("Rate limit")
        exc.retry_after = 3600.0
        delay = _extract_retry_delay(exc, default_delay=2.0)
        self.assertEqual(delay, MAX_RETRY_AFTER_DELAY)

    def test_malformed_retry_after_fallbacks(self):
        """NaN, Infinity, negative, aur invalid strings fallback delay par jayen."""
        for malformed in ["NaN", "Infinity", -10, "invalid_str", None]:
            exc = Exception("Rate limit")
            exc.retry_after = malformed
            delay = _extract_retry_delay(exc, default_delay=5.0)
            self.assertEqual(delay, 5.0)

    @patch("time.sleep", return_value=None)
    def test_total_retry_budget_exhaustion(self, mock_sleep):
        """Total accumulated sleep budget exceed hone par TimeoutError raise ho."""
        mock_llm = MagicMock()
        rate_limit_exc = Exception("429 Too Many Requests")
        rate_limit_exc.retry_after = MAX_RETRY_AFTER_DELAY
        mock_llm.invoke.side_effect = rate_limit_exc

        with self.assertRaises(TimeoutError):
            _invoke_with_retry(
                operation_name="test_summary",
                runnable=mock_llm,
                payload={"text": "sample text"},
                max_retries=5,
                base_delay=1.0,
            )


if __name__ == "__main__":
    unittest.main()
