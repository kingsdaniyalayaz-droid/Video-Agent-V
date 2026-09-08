"""Regression tests for the centralized LLM provider layer (tests A-H).

It must be safe to run with NO real API keys: everything is mocked and no
network call is ever made.

    Run:  python -m pytest tests/test_llm_provider.py -v
    Or:   python -m unittest tests.test_llm_provider -v

Coverage
--------
A. Primary provider success
B. Primary HTTP 429 -> fallback success
C. Primary 429 -> fallback failure (original errors preserved)
D. Non-retryable authentication error does NOT endlessly retry
E. Title generation uses the centralized provider
F. Summary generation uses the centralized provider
G. Roman Urdu translator uses the centralized provider
H. Existing successful functionality remains unchanged
"""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Env must be set BEFORE importing core modules (they read env at import
# time).  The values here are test-only; nothing is ever sent anywhere.
# ---------------------------------------------------------------------------

os.environ.setdefault("MISTRAL_API_KEY", "test-mistral-key")
os.environ.setdefault("MISTRAL_MODEL", "mistral-small-latest")
os.environ.setdefault("FALLBACK_PROVIDER", "")
os.environ.setdefault("FALLBACK_API_KEY", "")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from core import roman_urdu_translator as ru_mod
from core import summarize as summarize_mod
from core.llm_provider import (
    FailoverChatModel,
    ProviderExhaustedError,
    classify_error,
    is_rate_limit_error,
    is_retryable_error,
)


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _Fake429Error(RuntimeError):
    """HTTP 429 with Retry-After, shaped like a LangChain/httpx error."""

    def __init__(self, message="Rate limit exceeded, retry later.", retry_after=None):
        super().__init__(message)
        self.response = _FakeResponse(
            429, {"Retry-After": retry_after} if retry_after else {}
        )


class _Fake401Error(RuntimeError):
    def __init__(self, message="Invalid API key"):
        super().__init__(message)
        self.response = _FakeResponse(401)


class _Fake500Error(RuntimeError):
    def __init__(self):
        super().__init__("Internal server error")
        self.response = _FakeResponse(500)


class _FakeClient:
    """Minimal LangChain-style client exposing ``_generate``/``_agenerate``."""

    def __init__(self, content="ok", error=None, name="fake"):
        self._content = content
        self._error = error
        self.name = name
        self.generate_calls = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.generate_calls += 1
        if self._error is not None:
            raise self._error
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._content))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _counting_llm(content: str):
    """A real Runnable (RunnableLambda) that returns ``content`` and counts calls.

    RunnableLambda is a genuine LangChain Runnable, so ``prompt | llm | parser``
    chains work exactly as they do in production -- no pydantic subclassing.
    """
    calls = {"n": 0}

    def _respond(messages):
        calls["n"] += 1
        return AIMessage(content=content)

    return RunnableLambda(_respond), calls


# ---------------------------------------------------------------------------
# A. Primary provider success
# ---------------------------------------------------------------------------

class PrimarySuccessTests(unittest.TestCase):
    def test_primary_success_returns_without_touching_fallback(self):
        primary = _FakeClient(content="PRIMARY-ANSWER")
        fallback = _FakeClient(content="FALLBACK-ANSWER")
        model = FailoverChatModel(primary=primary, fallback=fallback)

        result = model.invoke([HumanMessage(content="hello")])

        self.assertEqual(result.content, "PRIMARY-ANSWER")
        self.assertEqual(primary.generate_calls, 1)
        self.assertEqual(fallback.generate_calls, 0)

    def test_failover_model_is_langchain_compatible(self):
        model = FailoverChatModel(primary=_FakeClient(content="x"))
        self.assertIsInstance(model, BaseChatModel)
        self.assertEqual(model._llm_type, "failover_chat_model")


# ---------------------------------------------------------------------------
# B. Primary HTTP 429 -> fallback success
# ---------------------------------------------------------------------------

class FailoverSuccessTests(unittest.TestCase):
    def test_primary_429_falls_back_and_returns(self):
        primary = _FakeClient(error=_Fake429Error(retry_after="7"))
        fallback = _FakeClient(content="FALLBACK-ANSWER")
        model = FailoverChatModel(primary=primary, fallback=fallback)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = model.invoke([HumanMessage(content="hello")])

        self.assertEqual(result.content, "FALLBACK-ANSWER")
        self.assertEqual(primary.generate_calls, 1)
        self.assertEqual(fallback.generate_calls, 1)
        log = out.getvalue()
        self.assertIn("rate limited (HTTP 429)", log)
        self.assertIn("Switching to fallback", log)
        self.assertIn("Fallback provider successful", log)

    def test_primary_500_falls_back(self):
        primary = _FakeClient(error=_Fake500Error())
        fallback = _FakeClient(content="RECOVERED")
        model = FailoverChatModel(primary=primary, fallback=fallback)
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.invoke([HumanMessage(content="hello")])
        self.assertEqual(result.content, "RECOVERED")


# ---------------------------------------------------------------------------
# C. Primary 429 -> fallback failure: original errors preserved, no loop
# ---------------------------------------------------------------------------

class FailoverExhaustedTests(unittest.TestCase):
    def test_both_providers_fail_raises_with_original_errors(self):
        primary_error = _Fake429Error(retry_after="7")
        fallback_error = _Fake500Error()
        primary = _FakeClient(error=primary_error)
        fallback = _FakeClient(error=fallback_error)
        model = FailoverChatModel(primary=primary, fallback=fallback)

        with self.assertRaises(ProviderExhaustedError) as ctx:
            model.invoke([HumanMessage(content="hello")])

        self.assertIs(ctx.exception.primary_error, primary_error)
        self.assertIs(ctx.exception.fallback_error, fallback_error)
        self.assertTrue(ctx.exception.rate_limited)  # primary was a 429
        self.assertEqual(primary.generate_calls, 1)
        self.assertEqual(fallback.generate_calls, 1)  # exactly one attempt each

    def test_no_fallback_configured_raises_original(self):
        primary_error = _Fake429Error()
        primary = _FakeClient(error=primary_error)
        model = FailoverChatModel(primary=primary, fallback=None)
        with self.assertRaises(RuntimeError) as ctx:
            model.invoke([HumanMessage(content="hello")])
        self.assertIs(ctx.exception, primary_error)


# ---------------------------------------------------------------------------
# D. Non-retryable authentication error does NOT endlessly retry
# ---------------------------------------------------------------------------

class PermanentErrorTests(unittest.TestCase):
    def test_auth_error_raises_immediately_no_fallback_no_retries(self):
        primary = _FakeClient(error=_Fake401Error())
        fallback = _FakeClient(content="FALLBACK-ANSWER")
        model = FailoverChatModel(primary=primary, fallback=fallback)

        with self.assertRaises(_Fake401Error) as ctx:
            model.invoke([HumanMessage(content="hello")])

        self.assertIsInstance(ctx.exception, _Fake401Error)
        self.assertEqual(primary.generate_calls, 1)   # exactly one attempt
        self.assertEqual(fallback.generate_calls, 0)  # fallback never tried


# ---------------------------------------------------------------------------
# Error classification (shared, centralized)
# ---------------------------------------------------------------------------

class ErrorClassificationTests(unittest.TestCase):
    def test_429_is_rate_limit(self):
        self.assertEqual(classify_error(_Fake429Error()), "rate_limit")
        self.assertTrue(is_rate_limit_error(_Fake429Error()))
        self.assertTrue(is_retryable_error(_Fake429Error()))

    def test_500_is_retryable_but_not_rate_limit(self):
        self.assertEqual(classify_error(_Fake500Error()), "retryable")
        self.assertFalse(is_rate_limit_error(_Fake500Error()))
        self.assertTrue(is_retryable_error(_Fake500Error()))

    def test_401_is_permanent(self):
        self.assertEqual(classify_error(_Fake401Error()), "permanent")
        self.assertFalse(is_rate_limit_error(_Fake401Error()))
        self.assertFalse(is_retryable_error(_Fake401Error()))

    def test_text_only_rate_limit_signature_detected(self):
        exc = RuntimeError("Mistral API error 1300: Rate limit reached, retry later.")
        self.assertEqual(classify_error(exc), "rate_limit")


# ---------------------------------------------------------------------------
# E. Title generation uses the centralized provider
# ---------------------------------------------------------------------------

class TitleGenerationProviderTests(unittest.TestCase):
    def test_generate_title_routes_through_centralized_provider(self):
        fake_llm, calls = _counting_llm("Quarterly Planning Meeting")
        transcript = (
            "Board meeting about the quarterly plan and the 2026 roadmap. "
            "Sales numbers, hiring and budget were discussed in detail. "
        ) * 40
        with patch.object(summarize_mod, "get_llm", return_value=fake_llm):
            title = summarize_mod.generate_title(transcript, source_type="meeting")
        self.assertEqual(title, "Quarterly Planning Meeting")
        self.assertEqual(calls["n"], 1)

    def test_summarize_get_llm_is_the_centralized_provider(self):
        fake = object()
        with patch.object(summarize_mod, "_provider_get_llm", return_value=fake):
            self.assertIs(summarize_mod.get_llm(), fake)


# ---------------------------------------------------------------------------
# F. Summary generation uses the centralized provider
# ---------------------------------------------------------------------------

class SummaryGenerationProviderTests(unittest.TestCase):
    def test_summarize_routes_through_centralized_provider(self):
        fake_llm, calls = _counting_llm("FINAL SUMMARY TEXT")
        transcript = ("This meeting covered the 2026 roadmap and budget. ") * 60
        with patch.object(summarize_mod, "get_llm", return_value=fake_llm):
            result = summarize_mod.summarize(transcript, source_type="meeting")
        self.assertEqual(result, "FINAL SUMMARY TEXT")
        self.assertGreaterEqual(calls["n"], 1)


# ---------------------------------------------------------------------------
# G. Roman Urdu translator uses the centralized provider
# ---------------------------------------------------------------------------

class TranslatorProviderTests(unittest.TestCase):
    def setUp(self):
        for fn in (ru_mod.get_llm, ru_mod.get_translator_llm, ru_mod.get_reviewer_llm):
            try:
                fn.cache_clear()
            except AttributeError:
                pass

    def test_translator_factory_uses_centralized_provider(self):
        fake = object()
        with patch.object(ru_mod, "_provider_get_chat_model", return_value=fake) as mk:
            got = ru_mod._get_llm(model="mistral-small-latest", temperature=0.1)
        self.assertIs(got, fake)
        kwargs = mk.call_args.kwargs
        self.assertEqual(kwargs["model"], "mistral-small-latest")
        self.assertEqual(kwargs["temperature"], 0.1)

    def test_mistral_client_factory_passes_response_format_to_provider(self):
        fake = object()
        with patch.object(ru_mod, "_provider_get_chat_model", return_value=fake) as mk, \
                contextlib.redirect_stdout(io.StringIO()):
            got = ru_mod._create_mistral_client(
                model="m2", role="Semantic Reviewer", temperature=0.0,
                response_format={"type": "json_object"},
            )
        self.assertIs(got, fake)
        self.assertEqual(mk.call_args.kwargs["model"], "m2")
        self.assertEqual(mk.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_public_llm_accessors_return_provider_model(self):
        fake = object()
        with patch.object(ru_mod, "_provider_get_chat_model", return_value=fake):
            self.assertIs(ru_mod.get_llm(), fake)
            self.assertIs(ru_mod.get_translator_llm(), fake)


# ---------------------------------------------------------------------------
# H. Existing successful functionality remains unchanged
# ---------------------------------------------------------------------------

class ExistingFunctionalityUnchangedTests(unittest.TestCase):
    def test_roman_urdu_quality_validation_unchanged(self):
        text = "Aap apna kaam kar sakte hain. Aapka account active hai."
        result = ru_mod.validate_roman_urdu_quality(text, text)
        self.assertTrue(result["passed"])

    def test_title_empty_output_still_raises(self):
        fake_llm, _calls = _counting_llm("")
        with patch.object(summarize_mod, "get_llm", return_value=fake_llm):
            with self.assertRaises(RuntimeError):
                summarize_mod.generate_title(
                    "A meeting about quarterly planning and budget. " * 40, "meeting"
                )

    def test_transcript_splitting_unchanged(self):
        chunks = summarize_mod.split_transcript("Line one.\nLine two.\nLine three.")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("Line one.", "".join(chunks))

    def test_rate_limit_classification_helpers_still_exported(self):
        # The project's typed exceptions and retry helpers are untouched.
        from core.roman_urdu_translator import (
            _invoke_llm_with_retries,
            _is_rate_limit_error as ru_is_rate_limit,
            translate_to_roman_urdu,
        )
        self.assertTrue(callable(_invoke_llm_with_retries))
        self.assertTrue(callable(translate_to_roman_urdu))
        self.assertTrue(ru_is_rate_limit(_Fake429Error()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
