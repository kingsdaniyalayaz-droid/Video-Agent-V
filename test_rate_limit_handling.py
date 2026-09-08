"""Focused tests for HTTP 429 / rate-limit handling (mocks only; no real API)."""

import os
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import patch

from core.roman_urdu_translator import (
    RETRY_AFTER_CEILING,
    PermanentAPIError,
    RateLimitError,
    RetryableAPIError,
    TranslationPipelineError,
    _invoke_llm_with_retries,
    _is_rate_limit_error,
    _is_retryable_error,
    _parse_retry_after,
    _rate_limit_retry_after,
    translate_chunk,
    translate_to_roman_urdu,
)


class _FakeResponse:
    def __init__(self, status_code=None, retry_after=None):
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after


class _Fake429Error(RuntimeError):
    def __init__(self, message="Rate limit exceeded, retry later.", retry_after=None):
        super().__init__(message)
        self.response = _FakeResponse(status_code=429, retry_after=retry_after)


class _Fake401Error(RuntimeError):
    def __init__(self, message="Invalid API key"):
        super().__init__(message)
        self.response = _FakeResponse(status_code=401)


class _Fake500Error(RuntimeError):
    def __init__(self):
        super().__init__("Internal server error")
        self.response = _FakeResponse(status_code=500)


class _RaisingLLM:
    """invoke() raises the configured error on every call and counts calls."""

    def __init__(self, error):
        self._error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise self._error


class RateLimitClassificationTests(unittest.TestCase):
    def test_http_429_is_rate_limit(self):
        self.assertTrue(_is_rate_limit_error(_Fake429Error()))
        self.assertTrue(_is_rate_limit_error(RetryableAPIError("rate limit exceeded")))

    def test_text_only_rate_limit_signature_detected(self):
        # Some providers surface the Mistral code-1300 style "rate limit"
        # message without a clean status attribute.
        exc = RuntimeError("Mistral API error 1300: Rate limit reached, retry later.")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_other_temporary_errors_are_not_rate_limit(self):
        self.assertFalse(_is_rate_limit_error(_Fake500Error()))
        self.assertTrue(_is_retryable_error(_Fake500Error()))  # still retryable

    def test_permanent_errors_are_not_rate_limit(self):
        self.assertFalse(_is_rate_limit_error(_Fake401Error()))


class RetryAfterParsingTests(unittest.TestCase):
    def test_delta_seconds_parsed(self):
        self.assertEqual(_parse_retry_after("7"), 7.0)
        self.assertEqual(_parse_retry_after(7), 7.0)
        self.assertEqual(_parse_retry_after("5.5"), 5.5)

    def test_invalid_or_missing_returns_none(self):
        for bad in (None, "", "abc", "0", "-3", "   "):
            self.assertIsNone(_parse_retry_after(bad), repr(bad))

    def test_huge_retry_after_is_capped(self):
        self.assertEqual(_parse_retry_after("999999"), RETRY_AFTER_CEILING)

    def test_http_date_formats(self):
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
        parsed = _parse_retry_after(future)
        self.assertIsNotNone(parsed)
        self.assertGreater(parsed, 0.0)
        self.assertLessEqual(parsed, 30.0)
        past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=30))
        self.assertIsNone(_parse_retry_after(past))

    def test_header_read_from_response_or_exception(self):
        from_response = RuntimeError("429")
        from_response.response = _FakeResponse(status_code=429, retry_after="12")
        self.assertEqual(_rate_limit_retry_after(from_response), 12.0)
        on_exc = RuntimeError("429")
        on_exc.headers = {"Retry-After": "9"}
        self.assertEqual(_rate_limit_retry_after(on_exc), 9.0)
        self.assertIsNone(_rate_limit_retry_after(_Fake429Error(retry_after=None)))
        self.assertIsNone(_rate_limit_retry_after(RuntimeError("no headers at all")))


class RateLimitBackoffTests(unittest.TestCase):
    def test_retry_after_is_used_when_available(self):
        llm = _RaisingLLM(_Fake429Error(retry_after="7"))
        sleeps: list[float] = []
        with patch("core.roman_urdu_translator.random.uniform", return_value=0.0):
            with self.assertRaises(RateLimitError) as ctx:
                _invoke_llm_with_retries(llm, [], max_attempts=2, sleeper=sleeps.append)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(sleeps, [7.0])  # Retry-After honored exactly (zero jitter)
        exc = ctx.exception
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.retry_after, 7.0)
        self.assertTrue(exc.retryable)
        self.assertIsInstance(exc, RetryableAPIError)  # umbrella compatibility kept

    def test_exponential_backoff_fallback_without_retry_after(self):
        llm = _RaisingLLM(_Fake429Error(retry_after=None))
        sleeps: list[float] = []
        with patch("core.roman_urdu_translator.random.uniform", return_value=0.0):
            with self.assertRaises(RateLimitError) as ctx:
                _invoke_llm_with_retries(llm, [], max_attempts=3, sleeper=sleeps.append)
        # attempt 1 -> 5s, attempt 2 -> 10s, attempt 3 exhausts.
        self.assertEqual(sleeps, [5.0, 10.0])
        self.assertIsNone(ctx.exception.retry_after)
        self.assertEqual(llm.calls, 3)

    def test_generic_temporary_errors_still_use_generic_backoff(self):
        llm = _RaisingLLM(_Fake500Error())
        sleeps: list[float] = []
        with patch("core.roman_urdu_translator.random.uniform", return_value=0.0):
            with self.assertRaises(RetryableAPIError) as ctx:
                _invoke_llm_with_retries(llm, [], max_attempts=2, sleeper=sleeps.append)
        self.assertNotIsInstance(ctx.exception, RateLimitError)
        self.assertEqual(sleeps, [3.0])  # API_BASE_DELAY schedule unchanged


class PermanentErrorTests(unittest.TestCase):
    def test_permanent_errors_are_not_retried(self):
        llm = _RaisingLLM(_Fake401Error())
        sleeps: list[float] = []
        with self.assertRaises(PermanentAPIError):
            _invoke_llm_with_retries(llm, [], max_attempts=5, sleeper=sleeps.append)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(sleeps, [])


class SecretRedactionTests(unittest.TestCase):
    def test_api_key_never_appears_in_exceptions(self):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "sk-supersecret-abc"}, clear=False):
            # 401 permanent path: message contains the key, must be redacted.
            llm = _RaisingLLM(_Fake401Error("401 invalid key sk-supersecret-abc"))
            with self.assertRaises(PermanentAPIError) as ctx:
                _invoke_llm_with_retries(llm, [], max_attempts=2, sleeper=lambda _s: None)
            self.assertNotIn("sk-supersecret-abc", str(ctx.exception))
            # 429 path: message is static and must not contain the key at all.
            llm2 = _RaisingLLM(_Fake429Error("429 sk-supersecret-abc rate limit"))
            with self.assertRaises(RateLimitError) as ctx2:
                _invoke_llm_with_retries(llm2, [], max_attempts=1, sleeper=lambda _s: None)
            self.assertNotIn("sk-supersecret-abc", str(ctx2.exception))


class QualityRetrySeparationTests(unittest.TestCase):
    def test_rate_limit_does_not_consume_quality_retries(self):
        # Real translate_chunk path: a 429 exhausts API retries and raises a
        # rate_limited TranslationPipelineError at quality attempt 1/3 --
        # the remaining quality attempts (2 and 3) are never touched.
        llm = _RaisingLLM(_Fake429Error(retry_after="4"))
        sleeps: list[float] = []
        with patch("core.roman_urdu_translator.random.uniform", return_value=0.0):
            with self.assertRaises(TranslationPipelineError) as ctx:
                translate_chunk(
                    chunk="Aaj hum is topic per baat karenge.",
                    llm=llm,
                    chunk_number=1,
                    api_max_attempts=2,
                    sleeper=sleeps.append,
                )
        err = ctx.exception
        self.assertTrue(err.rate_limited)
        self.assertEqual(err.retry_after, 4.0)
        self.assertIn("rate-limited (HTTP 429)", str(err))
        self.assertIn("quality attempt 1/", str(err))  # stopped at first quality attempt
        self.assertEqual(llm.calls, 2)  # exactly API_MAX retries, not 2*3
        self.assertEqual(sleeps, [4.0])  # Retry-After honored at chunk level too


class ChunkPreservationTests(unittest.TestCase):
    def test_completed_chunks_preserved_when_later_chunk_rate_limited(self):
        transcript = "A" * 120 + "\n" + "B" * 120  # splits into 2 chunks at max_chars=100

        def fake_translate_chunk(**kwargs):
            if kwargs["chunk_number"] == 1:
                return "TRANSLATED-CHUNK-ONE"
            raise _Fake429Error(retry_after="3")

        with patch("core.roman_urdu_translator.translate_chunk", side_effect=fake_translate_chunk), \
             patch("core.roman_urdu_translator._get_llm", return_value=object()):
            with self.assertRaises(TranslationPipelineError) as ctx:
                translate_to_roman_urdu(transcript, max_chars=100)

        err = ctx.exception
        self.assertEqual(err.completed_chunks, ["TRANSLATED-CHUNK-ONE"])
        self.assertEqual(err.partial_translation, "TRANSLATED-CHUNK-ONE")
        self.assertIn("completed chunks preserved: 1", str(err))


if __name__ == "__main__":
    unittest.main(verbosity=2)
