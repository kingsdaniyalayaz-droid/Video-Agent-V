"""App-level translation error-handling tests (pure classification helpers).

No Streamlit, no network, no real Mistral calls: only the typed exceptions
and the pure helpers in ``core.translation_error_handling`` are exercised.
"""

import os
import unittest
from unittest.mock import patch

from core.roman_urdu_translator import (
    PermanentAPIError,
    RateLimitError,
    RetryableAPIError,
    TranslationPipelineError,
)
from core.translation_error_handling import (
    MAX_EXCEPTION_CHAIN_LENGTH,
    classify_translation_error,
    contains_rate_limit,
    get_completed_chunk_count,
    get_partial_translation,
    get_retry_after,
)


def _build_rate_limit_chain(completed_chunks=None, partial=""):
    """Build the same exception-wrapping shape the backend produces for 429."""
    if completed_chunks is None:
        completed_chunks = []
    try:
        try:
            raise RateLimitError(
                "Translation API rate-limited after 5 attempts (HTTP 429). "
                "Try again in a few minutes.",
                retry_after=4.0,
            )
        except RateLimitError as rate_exc:
            raise TranslationPipelineError(
                "Chunk 1 rate-limited (HTTP 429) at quality attempt 1/3.",
                rate_limited=True,
                retry_after=4.0,
            ) from rate_exc
    except TranslationPipelineError as inner_exc:
        raise TranslationPipelineError(
            f"Translation failed at chunk 1/3; completed chunks preserved: "
            f"{len(completed_chunks)}; last error: {inner_exc}",
            completed_chunks=completed_chunks,
            partial_translation=partial,
        ) from inner_exc


class RateLimitClassificationTests(unittest.TestCase):
    def test_direct_rate_limit_error_is_rate_limit(self):
        exc = RateLimitError("Translation API rate-limited after 5 attempts (HTTP 429).",
                             retry_after=7.0)
        info = classify_translation_error(exc)
        self.assertEqual(info["category"], "rate_limit")
        self.assertIn("Rate Limit", info["message"])
        self.assertEqual(info["retry_after"], 7.0)

    def test_wrapped_rate_limit_chain_still_rate_limit(self):
        with self.assertRaises(TranslationPipelineError) as ctx:
            _build_rate_limit_chain(completed_chunks=["chunk-one"], partial="chunk-one")
        info = classify_translation_error(ctx.exception)
        self.assertEqual(info["category"], "rate_limit")
        self.assertIn("Rate Limit", info["message"])
        # Retry-After survives the wrapping.
        self.assertEqual(info["retry_after"], 4.0)
        # Production failure said "preserved: 0" for the 3-chunk case.
        with self.assertRaises(TranslationPipelineError) as ctx_zero:
            _build_rate_limit_chain()
        info_zero = classify_translation_error(ctx_zero.exception)
        self.assertEqual(info_zero["category"], "rate_limit")
        self.assertEqual(info_zero["completed_chunks"], 0)

    def test_contains_rate_limit_sees_through_chain(self):
        with self.assertRaises(TranslationPipelineError) as ctx:
            _build_rate_limit_chain()
        self.assertTrue(contains_rate_limit(ctx.exception))

    def test_wrapped_chain_exposes_partial_progress(self):
        with self.assertRaises(TranslationPipelineError) as ctx:
            _build_rate_limit_chain(completed_chunks=["c1", "c2"], partial="c1\n\nc2")
        exc = ctx.exception
        self.assertEqual(get_completed_chunk_count(exc), 2)
        self.assertEqual(get_partial_translation(exc), "c1\n\nc2")


class OtherClassificationTests(unittest.TestCase):
    def test_retryable_api_error_is_temporary(self):
        exc = RetryableAPIError("Translation API failed after 5 attempts: timeout")
        info = classify_translation_error(exc)
        self.assertEqual(info["category"], "temporary")
        self.assertIn("Temporary", info["message"])

    def test_permanent_api_error_is_permanent(self):
        exc = PermanentAPIError("Translation API failed permanently: invalid API key")
        info = classify_translation_error(exc)
        self.assertEqual(info["category"], "permanent")
        self.assertIn("Configuration", info["message"])

    def test_unexpected_exception_is_generic(self):
        info = classify_translation_error(RuntimeError("something odd happened"))
        self.assertEqual(info["category"], "unexpected")
        self.assertIn("unexpected error", info["message"].casefold())

    def test_classification_is_type_based_not_string_based(self):
        # A plain RuntimeError whose text smells like a rate limit must NOT be
        # classified as one: we rely on the typed exceptions, not strings.
        exc = RuntimeError("rate limit exceeded, too many requests, HTTP 429")
        info = classify_translation_error(exc)
        self.assertNotEqual(info["category"], "rate_limit")
        self.assertEqual(info["category"], "unexpected")


class SecretRedactionTests(unittest.TestCase):
    def test_api_key_never_appears_in_user_facing_messages(self):
        secret = "sk-supersecret-abc123"
        with patch.dict(os.environ, {"MISTRAL_API_KEY": secret}, clear=False):
            exc = PermanentAPIError(
                "Translation API failed permanently: 401 invalid key sk-supersecret-abc123"
            )
            info = classify_translation_error(exc)
            self.assertNotIn(secret, info["message"])
            self.assertNotIn(secret, info["detail"])
            rate_exc = RateLimitError(
                "429 sk-supersecret-abc123 rate limit", retry_after=2.0
            )
            rate_info = classify_translation_error(rate_exc)
        self.assertNotIn(secret, rate_info["message"])
        self.assertNotIn(secret, rate_info["detail"])


class ChainSafetyTests(unittest.TestCase):
    def test_cyclic_chain_cannot_loop_forever(self):
        first = RuntimeError("a")
        second = RuntimeError("b")
        first.__cause__ = second
        second.__cause__ = first  # cycle
        # Must return promptly (a loop would hang the test runner).
        info = classify_translation_error(first)
        self.assertEqual(info["category"], "unexpected")

    def test_self_referencing_chain_terminates(self):
        exc = RuntimeError("self-loop")
        exc.__cause__ = exc
        info = classify_translation_error(exc)
        self.assertEqual(info["category"], "unexpected")

    def test_deep_chain_is_bounded(self):
        first = RuntimeError("deep-0")
        current = first
        for index in range(1, MAX_EXCEPTION_CHAIN_LENGTH * 3):
            nxt = RuntimeError(f"deep-{index}")
            current.__cause__ = nxt
            current = nxt
        info = classify_translation_error(first)
        self.assertEqual(info["category"], "unexpected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
