from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core import roman_urdu_translator as translator
from core import llm_provider


def _deterministic_pass(*_args, **_kwargs):
    return {
        "passed": True,
        "score": 100,
        "critical_issues": [],
        "minor_issues": [],
        "feedback": "",
        "retry_safe": True,
    }


def _run_chunk(reviewer_responses: list[str], *, quality_attempts: int = 3):
    calls: list[str] = []

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        calls.append(operation_name)
        if operation_name == "Translation API":
            return SimpleNamespace(content="Yeh aik durust tarjuma hai")
        return SimpleNamespace(content=reviewer_responses.pop(0))

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _source, text: text),
    ):
        result = translator.translate_chunk(
            "This is a source chunk.",
            llm=object(),
            reviewer_llm=object(),
            max_retries=quality_attempts,
            api_max_attempts=1,
            semantic_review_max_attempts=3,
            retry_delay=0,
            sleeper=lambda _delay: None,
        )
    return result, calls


# =====================================================================
# Test 1: Dedicated reviewer model configuration is selected
# =====================================================================

def test_reviewer_uses_dedicated_config_when_set(monkeypatch, capsys):
    """When SEMANTIC_REVIEW_PROVIDER/MODEL are set, get_reviewer_llm
    must pass them as explicit identity overrides to get_chat_model."""
    original_provider = translator.SEMANTIC_REVIEW_PROVIDER
    original_model = translator.SEMANTIC_REVIEW_MODEL
    try:
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_PROVIDER", "google")
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_MODEL", "gemini-1.5-flash")

        client = object()
        with patch.object(
            translator._llm_provider, "get_chat_model", return_value=client
        ) as factory:
            result = translator.get_reviewer_llm()

        assert result is client
        kwargs = factory.call_args.kwargs
        assert kwargs["provider"] == "google"
        assert kwargs["model"] == "gemini-1.5-flash"
        assert kwargs["role"] == "Semantic Reviewer"
        output = capsys.readouterr().out
        assert "Dedicated" in output
    finally:
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_PROVIDER", original_provider)
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_MODEL", original_model)


# =====================================================================
# Test 2: Translator configuration remains unchanged
# =====================================================================

def test_translator_config_unchanged_by_reviewer_settings(monkeypatch, capsys):
    """Translator client must NOT be affected by SEMANTIC_REVIEW_* settings."""
    original_provider = translator.SEMANTIC_REVIEW_PROVIDER
    original_model = translator.SEMANTIC_REVIEW_MODEL
    try:
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_PROVIDER", "google")
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_MODEL", "gemini-1.5-flash")

        client = object()
        with patch.object(
            translator._llm_provider, "get_chat_model", return_value=client
        ) as factory:
            result = translator.get_translator_llm()

        assert result is client
        kwargs = factory.call_args.kwargs
        assert "provider" not in kwargs
        assert kwargs["role"] == "Translator"
    finally:
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_PROVIDER", original_provider)
        monkeypatch.setattr(translator, "SEMANTIC_REVIEW_MODEL", original_model)


# =====================================================================
# Test 3: "User Safety: safe" is treated as invalid reviewer output
# =====================================================================

def test_user_safety_safe_is_invalid_reviewer_output():
    """'User Safety: safe' must be detected as invalid and yield review_error."""
    for invalid_text in ("User Safety: safe", "  User Safety: safe  ", "safe", "unsafe"):
        result = translator._parse_review(SimpleNamespace(content=invalid_text))
        assert result.get("review_error") == "invalid_safety_response", (
            f"'{invalid_text}' should be detected as invalid safety response"
        )
        assert result["passed"] is False


# =====================================================================
# Test 4: Reviewer claim of Devanagari rejected when deterministic
#          inspection confirms Latin-only output
# =====================================================================

def test_hallucinated_devanagari_claim_rejected():
    """When a reviewer claims Devanagari script but the candidate is
    Latin-only, the claim must be treated as hallucinated."""
    latin_only_candidate = "Aaj hum Python programming seekhenge"

    review = {
        "passed": False,
        "score": 20,
        "critical_issues": ["Uses Hindi script (Devanagari) instead of Roman Urdu"],
        "semantic_errors": [],
        "style_suggestions": [],
        "feedback": "Contains Devanagari characters",
    }

    result = translator._verify_reviewer_script_claims(review, latin_only_candidate)
    assert result.get("review_error") == "hallucinated_script_claim"
    assert result["critical_issues"] == []
    assert "hallucinated" in result["feedback"].lower()


def test_hallucinated_non_latin_claim_rejected():
    """Reviewer claiming non-Latin output on a Latin-only candidate must be
    treated as hallucinated."""
    candidate = "Python ek mashhoor programming language hai"

    review = {
        "passed": False,
        "score": 30,
        "critical_issues": ["Contains non-Latin characters"],
        "semantic_errors": ["Not natural Pakistani Roman Urdu"],
        "style_suggestions": [],
        "feedback": "Uses non-Roman Urdu script",
    }

    result = translator._verify_reviewer_script_claims(review, candidate)
    assert result.get("review_error") == "hallucinated_script_claim"


# =====================================================================
# Test 5: A real forbidden-script candidate is still rejected
# =====================================================================

def test_real_forbidden_script_still_rejected():
    """When the candidate truly contains forbidden script, the reviewer
    claim must be accepted (not treated as hallucinated)."""
    candidate_with_urdu = "\u0622\u062c \u06c1\u0645 Python \u0633\u06cc\u06a9\u06be\u06cc\u06ba \u06af\u06d2"

    review = {
        "passed": False,
        "score": 0,
        "critical_issues": ["Contains Urdu script characters"],
        "semantic_errors": [],
        "style_suggestions": [],
        "feedback": "Uses Arabic/Urdu script",
    }

    result = translator._verify_reviewer_script_claims(review, candidate_with_urdu)
    assert result.get("review_error") is None
    assert result["passed"] is False
    assert result["critical_issues"] == ["Contains Urdu script characters"]


# =====================================================================
# Test 6: Invalid reviewer responses retry reviewer without
#          regenerating translation
# =====================================================================

def test_empty_review_is_retried_without_regenerating_translation():
    result, calls = _run_chunk([
        "",
        '{"passed": true, "score": 100, "critical_issues": [], "semantic_errors": [], "style_suggestions": []}',
    ])

    assert result == "Yeh aik durust tarjuma hai"
    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 2


def test_invalid_json_review_is_retried_without_regenerating_translation():
    result, calls = _run_chunk([
        "not valid JSON",
        '{"passed": true, "score": 100, "critical_issues": [], "semantic_errors": [], "style_suggestions": []}',
    ])

    assert result == "Yeh aik durust tarjuma hai"
    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 2


# =====================================================================
# Test 7: Repeated identical invalid reviewer responses fail early
# =====================================================================

def test_repeated_identical_invalid_responses_fail_early():
    """When the reviewer returns the same invalid response twice in a row,
    the pipeline must fail early instead of wasting all retry attempts."""
    calls: list[str] = []
    reviewer_responses = ["", "", ""]

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        calls.append(operation_name)
        if operation_name == "Translation API":
            return SimpleNamespace(content="Yeh aik durust tarjuma hai")
        return SimpleNamespace(content=reviewer_responses.pop(0))

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _s, t: t),
        pytest.raises(translator.TranslationPipelineError, match="Semantic reviewer failed after"),
    ):
        translator.translate_chunk(
            "Test source.",
            llm=object(),
            reviewer_llm=object(),
            max_retries=3,
            api_max_attempts=1,
            semantic_review_max_attempts=5,
            retry_delay=0,
            sleeper=lambda _: None,
        )

    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 5


def test_repeated_user_safety_responses_fail_early():
    """'User Safety: safe' repeated twice should fail early."""
    calls: list[str] = []

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        calls.append(operation_name)
        if operation_name == "Translation API":
            return SimpleNamespace(content="Yeh aik durust tarjuma hai")
        return SimpleNamespace(content="User Safety: safe")

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _s, t: t),
        pytest.raises(translator.TranslationPipelineError, match="Semantic reviewer failed after"),
    ):
        translator.translate_chunk(
            "Test source.",
            llm=object(),
            reviewer_llm=object(),
            max_retries=3,
            api_max_attempts=1,
            semantic_review_max_attempts=5,
            retry_delay=0,
            sleeper=lambda _: None,
        )

    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 5


# =====================================================================
# Test 8: Reviewer timeouts do not consume translation quality attempts
# =====================================================================

def test_reviewer_timeout_retries_without_regenerating_translation():
    calls: list[str] = []
    reviewer_attempts = 0

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        nonlocal reviewer_attempts
        calls.append(operation_name)
        if operation_name == "Translation API":
            return SimpleNamespace(content="Yeh aik durust tarjuma hai")
        reviewer_attempts += 1
        if reviewer_attempts == 1:
            raise translator.RetryableAPIError("semantic review timed out")
        return SimpleNamespace(content=(
            '{"passed": true, "score": 100, "critical_issues": [], '
            '"semantic_errors": [], "style_suggestions": []}'
        ))

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _source, text: text),
    ):
        result = translator.translate_chunk(
            "This is a source chunk.", llm=object(), reviewer_llm=object(),
            api_max_attempts=1, semantic_review_max_attempts=3,
            semantic_review_timeout=1, retry_delay=0, sleeper=lambda _delay: None,
        )

    assert result == "Yeh aik durust tarjuma hai"
    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 2


def test_reviewer_errors_exhaust_reviewer_attempts_without_translation_retry():
    calls: list[str] = []

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        calls.append(operation_name)
        return SimpleNamespace(content="Yeh aik durust tarjuma hai" if operation_name == "Translation API" else "")

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _source, text: text),
        pytest.raises(translator.TranslationPipelineError, match="Semantic reviewer failed"),
    ):
        translator.translate_chunk(
            "This is a source chunk.", llm=object(), reviewer_llm=object(),
            api_max_attempts=1, semantic_review_max_attempts=3,
            retry_delay=0, sleeper=lambda _delay: None,
        )

    assert calls.count("Translation API") == 1
    # Runs exactly semantic_review_max_attempts
    assert calls.count("semantic review") == 3


# =====================================================================
# Test 9: Genuine semantic failures still trigger semantic correction
# =====================================================================

def test_genuine_semantic_failure_triggers_retranslation():
    """A genuine semantic failure (not a hallucinated script claim) must
    still trigger a new translation attempt."""
    result, calls = _run_chunk([
        '{"passed": false, "score": 20, "critical_issues": [], "semantic_errors": ["meaning changed"], "style_suggestions": []}',
        '{"passed": true, "score": 100, "critical_issues": [], "semantic_errors": [], "style_suggestions": []}',
    ])

    assert result == "Yeh aik durust tarjuma hai"
    assert calls.count("Translation API") == 2
    assert calls.count("semantic review") == 2


# =====================================================================
# Test 10: RateLimitError contract remains preserved
# =====================================================================

def test_reviewer_rate_limit_keeps_rate_limit_error_contract():
    calls: list[str] = []

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        calls.append(operation_name)
        if operation_name == "Translation API":
            return SimpleNamespace(content="Yeh aik durust tarjuma hai")
        raise translator.RateLimitError("reviewer rate-limited", retry_after=17)

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _source, text: text),
        pytest.raises(translator.TranslationPipelineError) as error,
    ):
        translator.translate_chunk(
            "This is a source chunk.", llm=object(), reviewer_llm=object(),
            api_max_attempts=1, semantic_review_max_attempts=3,
            retry_delay=0, sleeper=lambda _delay: None,
        )

    assert error.value.rate_limited is True
    assert error.value.retry_after == 17
    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 1


# =====================================================================
# Additional: Hallucination guard in full pipeline
# =====================================================================

def test_hallucinated_script_claim_retries_reviewer_not_translation():
    """When reviewer hallucinates a Devanagari claim on Latin-only output,
    the pipeline should retry the reviewer (not regenerate translation)."""
    calls: list[str] = []
    review_attempts = 0

    def invoke(_llm, _messages, *, operation_name, **_kwargs):
        nonlocal review_attempts
        calls.append(operation_name)
        if operation_name == "Translation API":
            return SimpleNamespace(content="Yeh aik durust tarjuma hai")
        review_attempts += 1
        if review_attempts == 1:
            return SimpleNamespace(content=(
                '{"passed": false, "score": 20, '
                '"critical_issues": ["Uses Hindi script (Devanagari) instead of Roman Urdu"], '
                '"semantic_errors": [], "style_suggestions": []}'
            ))
        return SimpleNamespace(content=(
            '{"passed": true, "score": 95, "critical_issues": [], '
            '"semantic_errors": [], "style_suggestions": []}'
        ))

    with (
        patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke),
        patch.object(translator, "validate_roman_urdu_quality", side_effect=_deterministic_pass),
        patch.object(translator, "_restore_chunk_timestamps", side_effect=lambda _s, t: t),
    ):
        result = translator.translate_chunk(
            "This is a source chunk.",
            llm=object(),
            reviewer_llm=object(),
            max_retries=3,
            api_max_attempts=1,
            semantic_review_max_attempts=3,
            retry_delay=0,
            sleeper=lambda _: None,
        )

    assert result == "Yeh aik durust tarjuma hai"
    assert calls.count("Translation API") == 1
    assert calls.count("semantic review") == 2


def test_no_script_claim_passes_through_unchanged():
    """A review without script claims should not be altered by the guard."""
    candidate = "Aaj hum Python programming seekhenge"
    review = {
        "passed": True,
        "score": 95,
        "critical_issues": [],
        "semantic_errors": [],
        "style_suggestions": ["Minor wording improvement possible"],
        "feedback": "Good translation",
    }

    result = translator._verify_reviewer_script_claims(review, candidate)
    assert result is review


def test_reviewer_request_timeout_is_retryable():
    class SlowReviewer:
        def invoke(self, _messages):
            time.sleep(0.05)

    with pytest.raises(translator.RetryableAPIError, match="TimeoutError"):
        translator._invoke_llm_with_retries(
            SlowReviewer(),
            [],
            operation_name="semantic review",
            max_attempts=1,
            request_timeout=0.001,
            sleeper=lambda _delay: None,
        )


def test_large_semantic_payload_is_reviewed_in_bounded_sections():
    calls: list[list[object]] = []
    source = "A" * 6000
    translation = "B" * 6000

    def invoke(_llm, messages, **_kwargs):
        calls.append(messages)
        return SimpleNamespace(content=(
            '{"passed": true, "score": 100, "critical_issues": [], '
            '"semantic_errors": [], "style_suggestions": []}'
        ))

    with patch.object(translator, "_invoke_llm_with_retries", side_effect=invoke):
        review = translator.review_roman_urdu_translation(
            source,
            translation,
            llm=object(),
            deterministic_result=_deterministic_pass(),
            max_payload_chars=2000,
        )

    assert review["passed"] is True
    assert len(calls) == 6
    assert all(len(messages[1].content) < 2500 for messages in calls)
