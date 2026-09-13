from __future__ import annotations

from unittest.mock import patch

import pytest

from core import llm_provider
from core import roman_urdu_translator as translator


@pytest.fixture(autouse=True)
def _isolated_provider_runtime():
    previous = llm_provider.get_runtime_config()
    llm_provider.clear_runtime_config()
    yield
    if previous is None:
        llm_provider.clear_runtime_config()
    else:
        llm_provider.set_runtime_config(previous)


def _activate_openrouter_runtime():
    return llm_provider.set_runtime_config(llm_provider.LLMConfig(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="openrouter-runtime-secret",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.42,
    ))


def test_openrouter_runtime_does_not_require_mistral_api_key(monkeypatch, capsys):
    _activate_openrouter_runtime()
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    client = object()

    with patch.object(translator._llm_provider, "get_chat_model", return_value=client) as factory:
        assert translator.get_translator_llm() is client

    assert factory.call_args.kwargs == {"role": "Translator", "response_format": None}
    output = capsys.readouterr().out
    assert "openrouter" in output
    assert "openrouter-runtime-secret" not in output


def test_runtime_configuration_precedes_legacy_mistral_defaults(monkeypatch):
    _activate_openrouter_runtime()
    monkeypatch.setattr(translator, "MISTRAL_MODEL", "mistral-small-latest")
    client = object()

    with patch.object(translator._llm_provider, "get_chat_model", return_value=client) as factory:
        assert translator._get_llm(model=translator.MISTRAL_MODEL) is client

    kwargs = factory.call_args.kwargs
    assert kwargs["role"] == "Translator"
    assert "model" not in kwargs
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_translator_and_reviewer_use_centralized_provider_layer(capsys):
    _activate_openrouter_runtime()
    translator_client = object()
    reviewer_client = object()

    with (
        patch.object(translator, "SEMANTIC_REVIEW_PROVIDER", None),
        patch.object(translator, "SEMANTIC_REVIEW_MODEL", None),
        patch.object(
            translator._llm_provider,
            "get_chat_model",
            side_effect=[translator_client, reviewer_client],
        ) as factory,
    ):
        assert translator.get_translator_llm() is translator_client
        assert translator.get_reviewer_llm() is reviewer_client

    calls = factory.call_args_list
    assert calls[0].kwargs == {"role": "Translator", "response_format": None}
    assert calls[1].kwargs == {
        "role": "Semantic Reviewer",
        "response_format": {"type": "json_object"},
        "timeout": translator.SEMANTIC_REVIEW_TIMEOUT,
    }
    assert "openrouter-runtime-secret" not in capsys.readouterr().out
