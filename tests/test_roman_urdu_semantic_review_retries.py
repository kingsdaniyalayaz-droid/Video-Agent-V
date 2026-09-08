from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core import roman_urdu_translator as translator


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


def test_valid_semantic_failure_uses_existing_translation_retry():
    result, calls = _run_chunk([
        '{"passed": false, "score": 20, "critical_issues": [], "semantic_errors": ["meaning changed"], "style_suggestions": []}',
        '{"passed": true, "score": 100, "critical_issues": [], "semantic_errors": [], "style_suggestions": []}',
    ])

    assert result == "Yeh aik durust tarjuma hai"
    assert calls.count("Translation API") == 2
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
    assert calls.count("semantic review") == 3
