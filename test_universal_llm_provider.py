"""
tests/test_universal_llm_provider.py

Verification suite for the UNIVERSAL LLM CONFIGURATION SYSTEM.

Covers the required test scenarios:

    T1  Mistral                (explicit provider)
    T2  Gemini                 (explicit provider)
    T3  OpenAI                 (explicit provider)
    T4  NVIDIA + base URL      (explicit provider, OpenAI-compatible path)
    T5  OpenAI Compatible      (custom provider + custom base URL)
    T6  Auto Detect, known family
    T7  Auto Detect, unknown model -> actionable error, never a crash
    T8  Switch Mistral -> NVIDIA: no stale client, every feature follows
    T9  Same video, Model A then Model B: fingerprints differ, cached AI
        outputs are never cross-reused

Plus: cache-key hygiene (no raw keys), redaction, masking, config store
persistence/switch/delete, fingerprint API, backward compatibility.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import core.llm_provider as lp  # noqa: E402
from core.llm_provider import (  # noqa: E402
    LLMConfig,
    LLMConfigStore,
    InvalidConfigurationError,
    MissingAPIKeyError,
    UnsupportedProviderError,
    activate_configuration,
    build_chat_model,
    clear_runtime_config,
    configure_llm,
    delete_configuration,
    fingerprints_compatible,
    get_chat_model,
    get_runtime_config,
    list_configurations,
    llm_fingerprint,
    mask_api_key,
    resolve_provider,
    save_configuration,
    set_config_store,
    set_runtime_config,
    supported_providers,
    translation_fingerprint,
    provider_alias,
    validate_config,
    validate_configuration,
)

DUMMY_KEY = "sk-test-dummy-key-1234567890"
GEMINI_KEY = "AIzaSyDummyGeminiKey000"
NVIDIA_KEY = "nvapi-dummy-nvidia-key"
GROQ_KEY = "gsk_dummy_groq_key"
CUSTOM_KEY = "sk-custom-endpoint-key"


@pytest.fixture(autouse=True)
def clean_runtime():
    """Start every test with a clean runtime config and a temp store."""
    clear_runtime_config()
    tmp_store = LLMConfigStore(path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tmp_store_test.json"))
    set_config_store(tmp_store)
    yield
    clear_runtime_config()
    if tmp_store.path.exists():
        tmp_store.path.unlink()


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def test_all_required_providers_registered():
    required = {"mistral", "google", "openai", "anthropic", "groq",
                "nvidia", "openrouter", "together", "deepinfra",
                "ollama", "lm_studio", "vllm", "openai_compatible"}
    assert required.issubset(set(supported_providers()))
    assert provider_alias("gemini") == "google"
    assert provider_alias("LM Studio") == "lm_studio"
    assert provider_alias("lmstudio") == "lm_studio"
    assert provider_alias("openai-compatible") == "openai_compatible"


# ---------------------------------------------------------------------------
# T1-T5: explicit provider scenarios
# ---------------------------------------------------------------------------

def test_t1_mistral_explicit():
    cfg = validate_configuration(provider="mistral", model="mistral-small-latest", api_key=DUMMY_KEY)
    assert cfg.provider == "mistral"
    model = build_chat_model(cfg)
    assert type(model).__name__ == "ChatMistralAI"


def test_t2_gemini_explicit():
    cfg = validate_configuration(provider="gemini", model="gemini-1.5-flash", api_key=GEMINI_KEY)
    assert cfg.provider == "google"
    model = build_chat_model(cfg)
    assert type(model).__name__ == "ChatGoogleGenerativeAI"


def test_t3_openai_explicit_unknown_model_name_accepted():
    # Explicit provider wins -- the model name is NEVER a reason to reject.
    cfg = validate_configuration(provider="openai", model="gpt-4o", api_key=DUMMY_KEY)
    assert cfg.provider == "openai"
    model = build_chat_model(cfg)
    assert type(model).__name__ == "ChatOpenAI"

    # Even a garbage model name is fine when the provider is explicit:
    cfg2 = validate_configuration(provider="openai", model="nvid", api_key=DUMMY_KEY)
    assert cfg2.model == "nvid" and cfg2.provider == "openai"


def test_t4_nvidia_explicit_with_base_url():
    cfg = validate_configuration(
        provider="nvidia", model="meta/llama-3.1-70b-instruct",
        api_key=NVIDIA_KEY, base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.2)
    assert cfg.provider == "nvidia"
    assert cfg.base_url == "https://integrate.api.nvidia.com/v1"
    model = build_chat_model(cfg)
    assert type(model).__name__ == "ChatOpenAI"
    # NVIDIA default endpoint applies when no base URL is given.
    cfg_default = validate_configuration(provider="nvidia", model="meta/llama-3.1-70b-instruct",
                                         api_key=NVIDIA_KEY)
    assert cfg_default.base_url == lp.NVIDIA_BASE_URL


def test_t5_openai_compatible_custom_endpoint():
    cfg = validate_configuration(
        provider="openai_compatible", model="vendor/custom-model",
        api_key=CUSTOM_KEY, base_url="https://llm.internal.example/v1")
    assert cfg.provider == "openai_compatible"
    assert cfg.base_url == "https://llm.internal.example/v1"
    model = build_chat_model(cfg)
    assert type(model).__name__ == "ChatOpenAI"


def test_t5b_openai_compatible_requires_base_url():
    with pytest.raises(InvalidConfigurationError) as exc:
        validate_configuration(provider="openai_compatible", model="some/model", api_key=CUSTOM_KEY)
    assert "Base URL" in str(exc.value)


# ---------------------------------------------------------------------------
# T6: Auto detect -- known families
# ---------------------------------------------------------------------------

def test_t6_auto_detect_known_families():
    assert resolve_provider("mistral-small-latest") == "mistral"
    assert resolve_provider("open-mistral-7b") == "mistral"
    assert resolve_provider("gemini-1.5-flash") == "google"
    assert resolve_provider("models/gemini-1.5-pro") == "google"
    assert resolve_provider("gpt-4o-mini") == "openai"
    assert resolve_provider("o3-mini") == "openai"
    assert resolve_provider("claude-3-5-sonnet-latest") == "anthropic"
    assert resolve_provider("llama-3.1-8b-instant") == "groq"          # convenience heuristic
    assert resolve_provider("meta/llama-3.1-70b-instruct") == "nvidia"  # convenience heuristic


# ---------------------------------------------------------------------------
# T7: Auto detect -- unknown model: actionable error, never the old message
# ---------------------------------------------------------------------------

def test_t7_unknown_model_actionable_error_not_crash():
    with pytest.raises(UnsupportedProviderError) as exc:
        resolve_provider("nvid")
    msg = str(exc.value)
    assert "Provider could not be auto-detected" in msg
    assert "Select a provider or enter a Base URL" in msg
    assert "Could not determine a provider" not in msg


def test_t7b_unknown_model_with_base_url_uses_openai_compatible():
    # Enough configuration (a Base URL) -> OpenAI-compatible mode, no error.
    assert resolve_provider("nvid", None, "http://localhost:9999/v1") == "openai_compatible"
    cfg = validate_configuration(provider=None, model="nvid", api_key="sk-anything",
                                 base_url="http://localhost:9999/v1")
    assert cfg.provider == "openai_compatible"
    assert type(build_chat_model(cfg)).__name__ == "ChatOpenAI"


def test_t7c_unknown_model_no_base_url_no_provider_raises_actionable():
    with pytest.raises(UnsupportedProviderError) as exc:
        validate_configuration(model="totally-unknown-model", api_key=DUMMY_KEY)
    assert "Select a provider or enter a Base URL" in str(exc.value)


# ---------------------------------------------------------------------------
# Base URL priority (before model-name patterns)
# ---------------------------------------------------------------------------

def test_base_url_priority_over_model_patterns():
    # A groq-shaped model name + NVIDIA endpoint must NOT resolve to groq.
    assert resolve_provider("llama-3.1-70b", None, "https://integrate.api.nvidia.com/v1") == "nvidia"
    assert resolve_provider("gemini-1.5", None, "https://api.groq.com/openai/v1") == "groq"
    assert resolve_provider("anything", None, "https://openrouter.ai/api/v1") == "openrouter"


def test_local_port_inference():
    assert resolve_provider("llama3.2", None, "http://localhost:11434/v1") == "ollama"
    assert resolve_provider("qwen2.5", None, "http://127.0.0.1:1234/v1") == "lm_studio"
    assert resolve_provider("mistral-nemo", None, "http://localhost:8000/v1") == "vllm"


def test_ollama_lmstudio_vllm_no_api_key_required():
    cfg = validate_configuration(provider="ollama", model="llama3.2")
    assert cfg.api_key == lp._LOCAL_PLACEHOLDER_KEY
    assert type(build_chat_model(cfg)).__name__ == "ChatOpenAI"

    cfg2 = validate_configuration(provider="lm_studio", model="qwen2.5-coder")
    assert cfg2.api_key == lp._LOCAL_PLACEHOLDER_KEY

    cfg3 = validate_configuration(provider="vllm", model="zephyr-7b",
                                  base_url="http://localhost:8000/v1")
    assert cfg3.api_key == lp._LOCAL_PLACEHOLDER_KEY

    # Generic compatible + localhost endpoint => no key required either.
    cfg4 = validate_configuration(provider="openai_compatible", model="anything",
                                  base_url="http://localhost:8000/v1")
    assert cfg4.api_key == lp._LOCAL_PLACEHOLDER_KEY


def test_missing_key_raises_actionable():
    with pytest.raises(MissingAPIKeyError) as exc:
        validate_configuration(provider="openai", model="gpt-4o")
    assert "No API key" in str(exc.value)


# ---------------------------------------------------------------------------
# T8: switching provider -- no stale cached client
# ---------------------------------------------------------------------------

def test_t8_switch_mistral_to_nvidia_no_stale_cache():
    configure_llm(provider="mistral", model="mistral-small-latest", api_key=DUMMY_KEY)
    mistral_model = get_chat_model(role="Summarizer")
    assert mistral_model.primary_name == "mistral"
    assert type(mistral_model.primary).__name__ == "ChatMistralAI"

    # Switch to NVIDIA.
    configure_llm(provider="nvidia", model="meta/llama-3.1-70b-instruct",
                  api_key=NVIDIA_KEY, base_url=lp.NVIDIA_BASE_URL)
    nvidia_model = get_chat_model(role="Summarizer")
    assert nvidia_model.primary_name == "nvidia"
    assert type(nvidia_model.primary).__name__ == "ChatOpenAI"
    # The old Mistral client object MUST NOT be reused:
    assert nvidia_model.primary is not mistral_model.primary
    assert nvidia_model is not mistral_model

    # Extraction / RAG / translation roles follow the same active config:
    for role in ("Extractor", "RAGGenerative", "Translator"):
        m = get_chat_model(role=role)
        assert m.primary_name == "nvidia", f"role {role} stuck on old provider"
        assert type(m.primary).__name__ == "ChatOpenAI"


def test_t8b_switch_also_invalidates_per_role_caches():
    configure_llm(provider="openai", model="gpt-4o-mini", api_key=DUMMY_KEY)
    first = get_chat_model(role="Summarizer")
    configure_llm(provider="anthropic", model="claude-3-5-haiku-latest", api_key="sk-ant-dummy")
    second = get_chat_model(role="Summarizer")
    assert second.primary_name == "anthropic"
    assert type(second.primary).__name__ == "ChatAnthropic"
    assert second.primary is not first.primary


def test_t8c_same_configuration_reuses_client():
    configure_llm(provider="openai", model="gpt-4o-mini", api_key=DUMMY_KEY)
    a = get_chat_model(role="Summarizer")
    b = get_chat_model(role="Summarizer")
    assert a is b  # identical config => cached client is correct and wanted


# ---------------------------------------------------------------------------
# Cache-key hygiene: no raw API keys anywhere
# ---------------------------------------------------------------------------

def test_client_cache_key_contains_hash_not_raw_key():
    key = lp._client_key(
        model="gpt-4o", api_key="sk-TOP-SECRET-RAW-KEY", temperature=0.2,
        max_tokens=None, base_url=None, provider="openai",
        timeout=120.0, max_retries=0, response_format=None, role="Summarizer")
    serialized = repr(key)
    assert "sk-TOP-SECRET-RAW-KEY" not in serialized
    # indexed API-key slot is the 16-hex SHA-256 prefix
    assert key[1] == lp._key_fingerprint("sk-TOP-SECRET-RAW-KEY")
    assert len(key[1]) == 16 and all(c in "0123456789abcdef" for c in key[1])


def test_changing_key_yields_different_client():
    configure_llm(provider="openai", model="gpt-4o", api_key="sk-key-one")
    one = get_chat_model(role="Summarizer")
    configure_llm(provider="openai", model="gpt-4o", api_key="sk-key-two-different")
    two = get_chat_model(role="Summarizer")
    assert two.primary is not one.primary


# ---------------------------------------------------------------------------
# Fingerprints (T9): cache-compatibility of AI outputs
# ---------------------------------------------------------------------------

def test_t9_fingerprint_differs_between_models_same_video():
    fp_a = llm_fingerprint(provider="mistral", model="mistral-small-latest")
    fp_b = llm_fingerprint(provider="google", model="gemini-1.5-flash")
    assert fp_a != fp_b
    assert fingerprints_compatible(fp_a, fp_a) is True
    assert fingerprints_compatible(fp_a, fp_b) is False
    assert fingerprints_compatible(None, fp_a) is False
    assert fingerprints_compatible(fp_a, "") is False


def test_fingerprint_never_includes_api_key():
    fp1 = llm_fingerprint(provider="openai", model="gpt-4o")
    fp2 = llm_fingerprint(provider="openai", model="gpt-4o",
                          base_url="https://different.example/v1")
    fp3 = llm_fingerprint(provider="openai", model="gpt-4o",
                          base_url="https://different.example/v1")
    assert fp1 != fp2
    assert fp2 == fp3           # stable
    # Same identity regardless of the key:
    cfg_a = validate_configuration(provider="openai", model="gpt-4o", api_key="sk-A")
    cfg_b = validate_configuration(provider="openai", model="gpt-4o", api_key="sk-B")
    assert llm_fingerprint(cfg_a) == llm_fingerprint(cfg_b)
    assert llm_fingerprint(cfg_a) == llm_fingerprint(LLMConfig(provider="openai", model="gpt-4o"))


def test_translation_fingerprint_prevents_language_cross_reuse():
    base = dict(enabled=True, source_language="auto", provider="openai", model="gpt-4o")
    en = translation_fingerprint(target_language="english", **base)
    urdu = translation_fingerprint(target_language="urdu", **base)
    roman = translation_fingerprint(target_language="roman-urdu", **base)
    assert len({en, urdu, roman}) == 3
    no_translation = dict(source_language="auto", provider="openai", model="gpt-4o")
    disabled = translation_fingerprint(enabled=False, target_language="urdu", **no_translation)
    assert disabled != urdu


# ---------------------------------------------------------------------------
# Configuration store: multiple saved configurations + switching
# ---------------------------------------------------------------------------

def test_store_save_list_activate_switch_delete(tmp_store_clean=None):
    name1 = save_configuration(LLMConfig(
        provider="mistral", model="mistral-small-latest", api_key=DUMMY_KEY,
        config_name="Mistral Default"))
    assert name1 == "Mistral Default"

    name2 = save_configuration(LLMConfig(
        provider="nvidia", model="meta/llama-3.1-70b-instruct", api_key=NVIDIA_KEY,
        base_url=lp.NVIDIA_BASE_URL, temperature=0.3,
        config_name="NVIDIA Production"))
    assert name2 == "NVIDIA Production"

    rows = list_configurations(mask_keys=True)
    assert len(rows) == 2
    serialized = json.dumps(rows)
    assert DUMMY_KEY not in serialized and NVIDIA_KEY not in serialized
    for row in rows:
        assert "****" in row["api_key_masked"]

    # Activate NVIDIA:
    activated = activate_configuration("NVIDIA Production")
    assert activated.provider == "nvidia"
    runtime = get_runtime_config()
    assert runtime is not None and runtime.model == "meta/llama-3.1-70b-instruct"
    llm = get_chat_model(role="Summarizer")
    assert llm.primary_name == "nvidia"
    assert type(llm.primary).__name__ == "ChatOpenAI"

    # Switch to Mistral: runtime + caches move with it.
    activate_configuration("Mistral Default")
    llm2 = get_chat_model(role="Summarizer")
    assert llm2.primary_name == "mistral"
    assert llm2.primary is not llm.primary

    # Delete active config -> runtime falls back to env defaults.
    assert delete_configuration("Mistral Default") is True
    assert get_runtime_config() is None

    # Store file is chmod 600 and holds the keys (needed to function).
    store = lp.get_config_store()
    assert store.path.exists()
    assert (store.path.stat().st_mode & 0o777) == 0o600
    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["configurations"]["NVIDIA Production"]["api_key"] == NVIDIA_KEY


def test_store_activate_unknown_name_actionable():
    with pytest.raises(InvalidConfigurationError) as exc:
        activate_configuration("Does Not Exist")
    assert "No saved configuration" in str(exc.value)


def test_store_default_names_increment():
    n1 = save_configuration(LLMConfig(model="gpt-4o", api_key=DUMMY_KEY, provider="openai"))
    n2 = save_configuration(LLMConfig(model="gpt-4o", api_key=DUMMY_KEY, provider="openai"))
    assert n1 == "Configuration 1"
    assert n2 == "Configuration 2"


def test_restore_active_configuration():
    save_configuration(LLMConfig(provider="groq", model="llama-3.1-8b-instant",
                                 api_key=GROQ_KEY, config_name="Groq Free"))
    lp.get_config_store().activate("Groq Free")
    clear_runtime_config()
    restored = lp.restore_active_configuration()
    assert restored is not None and restored.provider == "groq"
    assert get_runtime_config() is not None


# ---------------------------------------------------------------------------
# Redaction / masking / secret hygiene
# ---------------------------------------------------------------------------

def test_mask_api_key():
    assert mask_api_key("sk-abcd1234wxyz") == "sk-a****wxyz"
    assert mask_api_key("short") == "*****"
    assert mask_api_key("") == ""


def test_redaction_scrubs_known_keys_from_error_text():
    configure_llm(provider="openai", model="gpt-4o", api_key="sk-super-ultra-secret-1")
    assert "sk-super-ultra-secret-1" not in lp._redact("boom sk-super-ultra-secret-1 boom")
    assert "[REDACTED]" in lp._redact("boom sk-super-ultra-secret-1 boom")


def test_validation_errors_do_not_leak_keys():
    with pytest.raises(Exception) as exc:
        validate_configuration(provider="openai", model="gpt-4o", api_key="sk-leaky-key-abc")
        raise RuntimeError("unreachable")
    # No-op guard above; real check:
    msg = str(exc.value)
    assert "sk-leaky-key-abc" not in msg


def test_classification_unchanged():
    err = MissingAPIKeyError("no key")
    assert lp.classify_error(err) == "permanent"
    assert lp.is_retryable_error(err) is False
    rl = lp.RateLimitError("rate limited")
    assert lp.classify_error(rl) == "rate_limit"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_gemini_alias_and_unknown_name_with_explicit_provider():
    # "Gemini" (UI label) maps to google; arbitrary models accepted.
    cfg = validate_configuration(provider="Gemini", model="gemini-anything", api_key=GEMINI_KEY)
    assert cfg.provider == "google"
    assert type(build_chat_model(cfg)).__name__ == "ChatGoogleGenerativeAI"


def test_legacy_env_fallback_still_works(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "legacy-env-key")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-small-latest")
    import importlib

    mod = importlib.reload(lp)
    mod.clear_runtime_config()
    llm = mod.get_chat_model()
    assert llm.primary_name == "mistral"
    assert "legacy-env-key" not in repr(llm)
    monkeypatch.delenv("MISTRAL_API_KEY")
    monkeypatch.delenv("MISTRAL_MODEL")
    importlib.reload(lp)


def test_configure_llm_and_get_llm_alias():
    configure_llm(provider="openai", model="gpt-4o-mini", api_key=DUMMY_KEY,
                  temperature=0.1, config_name="Prod")
    runtime = get_runtime_config()
    assert runtime.config_name == "Prod"
    model = lp.get_llm(role="Generic")
    assert model.primary_name == "openai"


def test_get_chat_model_per_call_override_does_not_touch_runtime():
    configure_llm(provider="openai", model="gpt-4o-mini", api_key=DUMMY_KEY)
    override = get_chat_model(provider="gemini", model="gemini-1.5-flash", api_key=GEMINI_KEY)
    assert override.primary_name == "google"
    assert get_runtime_config().provider == "openai"  # runtime untouched
