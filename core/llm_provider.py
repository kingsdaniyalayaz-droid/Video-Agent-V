"""
core/llm_provider.py

UNIVERSAL LLM PROVIDER LAYER -- the SINGLE place where the application
creates LLM chat models.

Every AI feature (summarization, title generation, extraction, RAG
generation, translation, prompt generation, ...) builds its chat model
through this module.  This module owns provider selection, provider
resolution, validation, client caching, failover, secret redaction, and
the persisted multi-configuration store.  No other module imports
ChatMistralAI / ChatOpenAI / ChatGoogleGenerativeAI / ChatAnthropic
directly.

ARCHITECTURE (provider-driven, NOT model-list-driven)
-----------------------------------------------------

    app.py / UI (AI Model Settings)
        |
        v
    configure_llm(...) / activate_configuration(name)
        |
        v
    core/llm_provider (THIS MODULE)
        +-- native providers: mistral, google(gemini), openai, anthropic
        |      built with their official LangChain SDKs
        |
        +-- OpenAI-compatible providers: groq, nvidia, openrouter,
        |      tokenrouter, together, deepinfra, ollama, lm_studio, vllm,
        |      openai_compatible (any custom endpoint)
        |      ALL built through ONE generic ChatOpenAI path
        |
        v
    FailoverChatModel (LangChain BaseChatModel)
        |
        +-- Summary        (core/summarize.py)
        +-- Extractor      (core/prompt_generator.py / extractor modules)
        +-- RAG generation (core/rag_engine.py)
        +-- Translation    (core/translator.py)

PROVIDER RESOLUTION PRIORITY (exactly as specified)
----------------------------------------------------

    1. EXPLICIT PROVIDER SELECTED BY USER   -> always wins
    2. BASE URL PROVIDED                    -> OpenAI-compatible endpoint
       (unless the explicitly selected provider needs its native SDK)
    3. AUTO-DETECT FROM KNOWN MODEL FAMILIES -> convenience fallback only
    4. UNKNOWN MODEL                        -> NEVER the old misleading
       "Could not determine a provider for model X" crash.

       Instead:
         * base_url present  -> OpenAI-compatible mode
         * otherwise         -> actionable validation error:
           "Provider could not be auto-detected for model 'X'.
            Select a provider or enter a Base URL."
       The model name alone never causes a rejection.

CONFIGURATION PRIORITY (when resolving the active LLM)
------------------------------------------------------

    1. Active user runtime configuration (configure_llm / activate_configuration)
    2. Explicit per-call function arguments (get_chat_model(..., model=...))
    3. Saved configuration (activated from the store)
    4. Environment fallback (MISTRAL_API_KEY / MISTRAL_MODEL / FALLBACK_*)
       -- kept for backward compatibility, never overrides the runtime config

CACHE DISCIPLINE
----------------

    * `_build_failover_model` is lru-cached, but the cache key is a full
      identity fingerprint (provider, model, temperature, base_url,
      response_format, timeout, max_retries, and a SHA-256 of the API key
      -- never the raw key).  Switching provider/model/key/base_url
      therefore always builds a fresh client: no stale Mistral client
      survives an NVIDIA switch.
    * set_runtime_config / clear_runtime_config / activate_configuration
      also clear the cache explicitly.

SECURITY
--------

    * API keys are never printed, logged, or embedded in exception text.
      Every known secret is redacted from error messages before they
      reach the UI.
    * llm_fingerprint() / llm_identity() -- used to version cached AI
      outputs in the database -- NEVER include the API key.
    * The configuration store persists keys to a chmod-600 JSON file
      (LLM_CONFIGS_PATH, default ./data/llm_configs.json).  Keys are
      returned through the store API only to code that needs to call the
      API; list_configurations() masks them.

Backward compatibility: MISTRAL_* / FALLBACK_* environment variables and
the legacy function names (create_primary_client, create_fallback_client,
get_llm) keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

load_dotenv()

# ---------------------------------------------------------------------------
# Legacy environment defaults (optional -- no fixed provider is required)
# ---------------------------------------------------------------------------

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()
MISTRAL_TEMPERATURE = float(os.getenv("MISTRAL_TEMPERATURE", "0.2"))
MISTRAL_TIMEOUT = float(os.getenv("MISTRAL_TIMEOUT", "120.0"))
MISTRAL_MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "0"))
MISTRAL_MAX_TOKENS = int(os.getenv("MISTRAL_MAX_TOKENS", "2048"))

FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "").strip().lower()
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "").strip()
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "").strip()
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "").strip() or None
FALLBACK_TEMPERATURE = float(os.getenv("FALLBACK_TEMPERATURE", "0.2"))
FALLBACK_TIMEOUT = float(os.getenv("FALLBACK_TIMEOUT", "120.0"))
FALLBACK_MAX_RETRIES = int(os.getenv("FALLBACK_MAX_RETRIES", "0"))

# Reasonable per-provider defaults used only when FALLBACK_MODEL is unset.
_DEFAULT_FALLBACK_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google": "gemini-1.5-flash",
    "mistral": "codestral-latest",
}

_FALLBACK_DISABLED_VALUES = {"", "none", "false", "off", "0", "disabled"}

# Legacy alias kept only for any external import that referenced the old
# constant.  The active provider is now dynamic -- do not use this.
PRIMARY_PROVIDER_NAME = "mistral"

_status_printed = False


# ---------------------------------------------------------------------------
# Well-known OpenAI-compatible endpoints (defaults / detection hints)
# ---------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
TOKENROUTER_BASE_URL = "https://api.tokenrouter.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"

# Dummy key sent to local servers (Ollama/LM Studio/vLLM) that ignore it.
_LOCAL_PLACEHOLDER_KEY = "not-required"

# Endpoint -> provider hint, used to label OpenAI-compatible endpoints when
# the user supplies a Base URL without selecting a provider.
_ENDPOINT_HINTS: tuple[tuple[str, Optional[str]], ...] = (
    ("groq.com", "groq"),
    ("openai.com", "openai"),
    ("nvidia.com", "nvidia"),
    ("openrouter.ai", "openrouter"),
    ("tokenrouter.com", "tokenrouter"),
    ("together.xyz", "together"),
    ("together.ai", "together"),
    ("deepinfra.com", "deepinfra"),
    ("ollama.com", "ollama"),
)

_LOCAL_PORT_HINTS: dict[int, str] = {
    11434: "ollama",
    1234: "lm_studio",
    8000: "vllm",
}

_LOCAL_HOST_MARKERS = ("localhost", "127.0.0.1", "0.0.0.0")


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """One LLM provider configuration (active runtime config or saved one).

    ``provider`` is optional: when omitted it is auto-detected from
    ``model`` and ``base_url`` (see :func:`resolve_provider`).  ``api_key``
    is never printed, logged or embedded in exception text.  ``config_name``
    is an optional user-facing label (e.g. "NVIDIA Production").
    """

    model: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    max_retries: int = 0
    base_url: Optional[str] = None
    response_format: Optional[dict] = None
    provider: Optional[str] = None
    config_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Typed exceptions (provider scope)
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Primary provider returned HTTP 429 / rate-limit quota exhausted."""

    status_code = 429
    retryable = True
    rate_limited = True

    def __init__(self, message: str, original: Optional[BaseException] = None,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.original = original
        self.retry_after = retry_after
        self.response = getattr(original, "response", None)


class RetryableProviderError(Exception):
    """Temporary provider failure (5xx, timeout, network) -- eligible for failover."""

    retryable = True
    rate_limited = False

    def __init__(self, message: str, original: Optional[BaseException] = None,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.original = original
        self.status_code = status_code
        self.response = getattr(original, "response", None)


class PermanentProviderError(Exception):
    """Non-retryable configuration / auth error (400/401/403 ...)."""

    retryable = False
    rate_limited = False

    def __init__(self, message: str, original: Optional[BaseException] = None,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.original = original
        self.status_code = status_code
        self.response = getattr(original, "response", None)


class ProviderExhaustedError(Exception):
    """Both the primary and the fallback provider failed for one request.

    The original exceptions are preserved on ``primary_error`` /
    ``fallback_error`` -- with any secrets redacted -- so higher layers can
    classify the failure exactly as before.
    """

    retryable = True  # transient at this level: a later retry may succeed

    def __init__(self, message: str,
                 primary_error: Optional[BaseException] = None,
                 fallback_error: Optional[BaseException] = None,
                 rate_limited: bool = False,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        self.rate_limited = rate_limited
        self.status_code = status_code
        self.retry_after = _extract_retry_after(primary_error)
        if self.status_code is None:
            self.status_code = _status_code_of(primary_error) or _status_code_of(fallback_error)


# -- User-friendly validation exceptions -----------------------------------
# All subclass PermanentProviderError so existing classify_error() /
# is_retryable_error() logic keeps treating them as non-retryable.

class MissingAPIKeyError(PermanentProviderError):
    """A provider requires an API key but none was supplied."""


class UnsupportedProviderError(PermanentProviderError):
    """The model name / provider cannot be mapped to a known provider."""


class InvalidConfigurationError(PermanentProviderError):
    """The runtime configuration is malformed (empty model, bad temperature, ...)."""


class MissingSDKError(PermanentProviderError):
    """The SDK required by the selected provider is not installed."""


# ---------------------------------------------------------------------------
# Redaction -- never leak API keys into logs or UI error messages
# ---------------------------------------------------------------------------

# Every secret the module has ever seen (env keys + runtime keys) is tracked
# here so _redact() can scrub any of them out of error text.
_KNOWN_SECRETS: set[str] = set()


def _register_secret(secret: str) -> None:
    if secret:
        _KNOWN_SECRETS.add(secret)


for _env_secret in (MISTRAL_API_KEY, FALLBACK_API_KEY):
    _register_secret(_env_secret)


def _redact(value: str) -> str:
    for secret in tuple(_KNOWN_SECRETS):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


def _safe_text(exc: Optional[BaseException]) -> str:
    return _redact(str(exc)) if exc is not None else ""


def _redact_exception(exc: BaseException) -> BaseException:
    """Return a copy of ``exc`` with any known secrets removed from its text.

    The exception type is preserved when possible so that higher layers can
    keep classifying by type; otherwise it is wrapped in a
    ``PermanentProviderError`` (still classified as non-retryable).
    """
    text = _redact(str(exc))
    if text == str(exc):
        return exc
    try:
        clone = type(exc)(text)
        clone.__cause__ = exc.__cause__
        clone.__context__ = exc.__context__
        return clone
    except Exception:
        wrapped = PermanentProviderError(
            f"{type(exc).__name__}: {text}", original=exc,
            status_code=_status_code_of(exc))
        wrapped.__cause__ = exc
        return wrapped


def mask_api_key(api_key: str) -> str:
    """Human-safe masked form of an API key for UI display.

    Never a reversible encoding: only the first and last 4 characters are
    kept.  Returns "" for an empty key.
    """
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}****{key[-4:]}"


# ---------------------------------------------------------------------------
# Error classification (shared by the failover model; lives here only)
# ---------------------------------------------------------------------------

def _status_code_of(exc: Optional[BaseException]) -> Optional[int]:
    if exc is None:
        return None
    for container in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status"):
            candidate = getattr(container, attr, None)
            try:
                if candidate is not None:
                    return int(candidate)
            except (TypeError, ValueError):
                continue
    match = re.search(r"\b(400|401|403|408|409|429|500|502|503|504|529)\b", str(exc))
    return int(match.group(1)) if match else None


def _extract_retry_after(exc: Optional[BaseException]) -> Optional[float]:
    if exc is None:
        return None
    try:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) or {}
        if hasattr(headers, "get"):
            raw = headers.get("Retry-After") or headers.get("retry-after")
            if raw is not None:
                return min(float(str(raw).strip()), 120.0)
    except (TypeError, ValueError):
        return None
    return None


def classify_error(exc: BaseException) -> str:
    """Return ``rate_limit`` | ``retryable`` | ``permanent``.

    Type/duck-based first (``rate_limited``/``RateLimitError``), then HTTP
    status, then conservative text markers.  Returns ``rate_limit`` only
    when there is real evidence of one.
    """
    if isinstance(exc, RateLimitError) or getattr(exc, "rate_limited", False) is True:
        return "rate_limit"
    text = str(exc).casefold()
    status = _status_code_of(exc)
    if status == 429 or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if status in {408, 409, 500, 502, 503, 504, 529}:
        return "retryable"
    if status in {400, 401, 403}:
        return "permanent"
    if any(marker in text for marker in (
        "timeout", "timed out", "temporarily unavailable", "service unavailable",
        "bad gateway", "gateway timeout", "connection reset", "connection aborted",
        "connection error", "network error", "overloaded", "internal server error",
        "server error",
    )):
        return "retryable"
    if any(marker in text for marker in (
        "invalid api key", "authentication", "unauthorized", "forbidden",
        "permission denied", "bad request", "invalid_request",
    )):
        return "permanent"
    return "retryable" if getattr(exc, "retryable", False) is True else "permanent"


def is_rate_limit_error(exc: BaseException) -> bool:
    return classify_error(exc) == "rate_limit"


def is_retryable_error(exc: BaseException) -> bool:
    return classify_error(exc) in {"rate_limit", "retryable"}


# ---------------------------------------------------------------------------
# Provider registry -- the modular extension point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    """One registered provider: detection hints + client builder.

    All OpenAI-compatible providers (NVIDIA, Groq, OpenRouter, TokenRouter,
    Together, DeepInfra, Ollama, LM Studio, vLLM, generic) share ONE builder
    (``_build_openai``) -- no duplicated provider implementations.
    """

    name: str
    sdk: str                                   # pip package name (for error text)
    native_sdk: bool                           # True -> official provider SDK
    patterns: tuple[str, ...]                  # convenience model-name hints (regexes)
    build: Callable[[LLMConfig], BaseChatModel]
    default_base_url: Optional[str] = None
    api_key_required: bool = True
    label: str = ""                            # UI label (defaults to name)


def _build_mistral(config: LLMConfig) -> BaseChatModel:
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError as exc:  # pragma: no cover
        raise MissingSDKError(
            "Provider 'mistral' requires the 'langchain-mistralai' SDK. "
            "Install it with: pip install langchain-mistralai"
        ) from exc
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_retries": config.max_retries,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens or MISTRAL_MAX_TOKENS,
        "api_key": config.api_key,
    }
    if config.base_url:
        # ChatMistralAI exposes the custom endpoint as ``endpoint``.
        kwargs["endpoint"] = config.base_url
    if config.response_format is not None:
        kwargs["model_kwargs"] = {"response_format": config.response_format}
    return ChatMistralAI(**kwargs)  # type: ignore[call-arg]


def _build_google(config: LLMConfig) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover
        raise MissingSDKError(
            "Provider 'google' requires the 'langchain-google-genai' SDK. "
            "Install it with: pip install langchain-google-genai"
        ) from exc
    return ChatGoogleGenerativeAI(  # type: ignore[call-arg]
        model=config.model,
        temperature=config.temperature,
        max_retries=config.max_retries,
        timeout=config.timeout,
        max_output_tokens=config.max_tokens or MISTRAL_MAX_TOKENS,
        api_key=config.api_key,
    )


def _build_openai(config: LLMConfig) -> BaseChatModel:
    """THE single OpenAI-compatible builder.

    Used by provider "openai" and by every OpenAI-compatible provider:
    groq, nvidia, openrouter, together, deepinfra, ollama, lm_studio,
    vllm and openai_compatible (any custom endpoint).  There is exactly
    ONE implementation of this path in the whole application.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise MissingSDKError(
            "This provider requires the 'langchain-openai' SDK. "
            "Install it with: pip install langchain-openai"
        ) from exc
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_retries": config.max_retries,
        "timeout": config.timeout,
        "api_key": config.api_key,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.response_format is not None:
        kwargs["model_kwargs"] = {"response_format": config.response_format}
    return ChatOpenAI(**kwargs)  # type: ignore[call-arg]


def _build_anthropic(config: LLMConfig) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover
        raise MissingSDKError(
            "Provider 'anthropic' requires the 'langchain-anthropic' SDK. "
            "Install it with: pip install langchain-anthropic"
        ) from exc
    return ChatAnthropic(  # type: ignore[call-arg]
        model=config.model,
        temperature=config.temperature,
        max_retries=config.max_retries,
        timeout=config.timeout,
        max_tokens=config.max_tokens or MISTRAL_MAX_TOKENS,
        api_key=config.api_key,
    )


def _spec(provider: str, sdk: str, native: bool, build: Callable[[LLMConfig], BaseChatModel],
          patterns: tuple[str, ...] = (), *, default_base_url: Optional[str] = None,
          api_key_required: bool = True, label: str = "") -> ProviderSpec:
    return ProviderSpec(name=provider, sdk=sdk, native_sdk=native, patterns=patterns,
                        build=build, default_base_url=default_base_url,
                        api_key_required=api_key_required, label=label or provider)


_PROVIDERS: dict[str, ProviderSpec] = {
    "mistral": _spec("mistral", "langchain-mistralai", True, _build_mistral,
                     (r"^(mistral|open-mistral|open-mixtral|open-codestral|"
                      r"pixtral|ministral|codestral|devstral)([-/:_]|$)",),
                     label="Mistral"),
    "google": _spec("google", "langchain-google-genai", True, _build_google,
                    (r"^(gemini|models/gemini)([-/:_]|$)",),
                    label="Gemini"),
    "openai": _spec("openai", "langchain-openai", True, _build_openai,
                    (r"^(gpt|o1|o3|o4|o5|chatgpt)([-/:_]|$)",),
                    label="OpenAI"),
    "anthropic": _spec("anthropic", "langchain-anthropic", True, _build_anthropic,
                       (r"^claude([-/:_]|$)",),
                       label="Anthropic"),
    "groq": _spec("groq", "langchain-openai", False, _build_openai,
                  (r"^(llama|gemma2?|mixtral|qwen|deepseek)([-/:_]|$)",),
                  default_base_url=GROQ_BASE_URL, label="Groq"),
    "nvidia": _spec("nvidia", "langchain-openai", False, _build_openai,
                    (r"^(meta|nvidia)/",),   # convenience: meta/llama-... is NVIDIA-hosted
                    default_base_url=NVIDIA_BASE_URL, label="NVIDIA"),
     "openrouter": _spec("openrouter", "langchain-openai", False, _build_openai, (),
                         default_base_url=OPENROUTER_BASE_URL, label="Open Router"),
    "tokenrouter": _spec("tokenrouter", "langchain-openai", False, _build_openai, (),
                          default_base_url=TOKENROUTER_BASE_URL, label="TokenRouter"),
    "together": _spec("together", "langchain-openai", False, _build_openai, (),
                      default_base_url=TOGETHER_BASE_URL, label="Together AI"),
    "deepinfra": _spec("deepinfra", "langchain-openai", False, _build_openai, (),
                       default_base_url=DEEPINFRA_BASE_URL, label="DeepInfra"),
    "ollama": _spec("ollama", "langchain-openai", False, _build_openai, (),
                    default_base_url=OLLAMA_BASE_URL, api_key_required=False, label="Ollama"),
    "lm_studio": _spec("lm_studio", "langchain-openai", False, _build_openai, (),
                       default_base_url=LM_STUDIO_BASE_URL, api_key_required=False,
                       label="LM Studio"),
    "vllm": _spec("vllm", "langchain-openai", False, _build_openai, (),
                  default_base_url=VLLM_BASE_URL, api_key_required=False, label="vLLM"),
    "openai_compatible": _spec("openai_compatible", "langchain-openai", False, _build_openai, (),
                               api_key_required=True, label="OpenAI Compatible"),
}

# Friendly aliases so callers can say provider="gemini" instead of "google".
_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "gpt": "openai",
    "google": "google",
    "openai": "openai",
    "anthropic": "anthropic",
    "mistral": "mistral",
    "groq": "groq",
    "nvidia": "nvidia",
    "nv": "nvidia",
    "nim": "nvidia",
    "openrouter": "openrouter",
    "tokenrouter": "tokenrouter",
    "token router": "tokenrouter",
    "token-router": "tokenrouter",
    "together": "together",
    "togetherai": "together",
    "deepinfra": "deepinfra",
    "ollama": "ollama",
    "lm_studio": "lm_studio",
    "lmstudio": "lm_studio",
    "lm-studio": "lm_studio",
    "vllm": "vllm",
    "openai_compatible": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "openai compatible": "openai_compatible",
    "compatible": "openai_compatible",
    "custom": "openai_compatible",
    "lm studio": "lm_studio",
    "open router": "openrouter",
    "together ai": "together",
}

# Providers whose models run behind an explicit user endpoint; the UI's
# "OpenAI Compatible" entry maps to this canonical name.
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"


def supported_providers() -> list[str]:
    """Canonical names of all registered providers (sorted)."""
    return sorted(_PROVIDERS)


def provider_label(provider: str) -> str:
    """Human-friendly label for a canonical provider name."""
    canonical = provider_alias(provider)
    spec = _PROVIDERS.get(canonical)
    return spec.label if spec else canonical


def provider_alias(provider: str) -> str:
    """Normalize a user-supplied provider string to its canonical name."""
    p = (provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(p, p)


# ---------------------------------------------------------------------------
# Provider resolution -- priority exactly as specified
# ---------------------------------------------------------------------------

def _normalize_base_url(base_url: Optional[str]) -> str:
    return (base_url or "").strip().rstrip("/").lower()


def _infer_openai_compatible_provider(base_url: str) -> Optional[str]:
    """Best-effort label for a Base URL (never authoritative; OpenAI-compatible)."""
    lower = _normalize_base_url(base_url)
    if not lower:
        return None
    for marker, provider in _ENDPOINT_HINTS:
        if marker in lower:
            return provider
    if any(h in lower for h in _LOCAL_HOST_MARKERS):
        # localhost => infer by port; unknown local port => generic compatible
        port_match = re.search(r":(\d{2,5})(?:/|$)", lower)
        if port_match:
            return _LOCAL_PORT_HINTS.get(int(port_match.group(1)), OPENAI_COMPATIBLE_PROVIDER)
        return OPENAI_COMPATIBLE_PROVIDER
    return None


def resolve_provider(model: str = "", provider: Optional[str] = None,
                     base_url: Optional[str] = None) -> str:
    """Resolve the provider for a model with optional hints.

    Priority (exactly as specified):

    1. Explicit ``provider`` -- always used, never overruled by names/keys.
    2. ``base_url`` -- treated as an OpenAI-compatible endpoint (labeled
       with the well-known endpoint when recognized).
    3. Known model-family patterns -- convenience fallback only.
    4. Unknown model -- raises :class:`UnsupportedProviderError` with an
       ACTIONABLE message ('Select a provider or enter a Base URL').

    A model name is NEVER rejected by itself: the only failure path is
    ``provider`` + ``base_url`` + ``model`` all being unhelpful, and that
    failure tells the user exactly what to do next.
    """
    # ``auto`` is a UI sentinel, not a provider.  Treat it exactly like an
    # omitted provider so an explicit selection can never be confused with
    # model-name detection.
    if provider and provider_alias(provider) not in {"", "auto", "autodetect", "auto_detect", "auto detect"}:
        canonical = provider_alias(provider)
        if canonical not in _PROVIDERS:
            raise UnsupportedProviderError(
                f"Unsupported provider {provider!r}. Supported providers: "
                f"{', '.join(supported_providers())}. "
                "Leave provider empty (or pick 'Auto Detect') to auto-detect."
            )
        return canonical

    model = (model or "").strip()
    base = (base_url or "").strip()

    if base:
        # Priority 2: a supplied Base URL means an OpenAI-compatible endpoint.
        return _infer_openai_compatible_provider(base) or OPENAI_COMPATIBLE_PROVIDER

    # Priority 3: convenience auto-detection from known model families.
    for name, spec in _PROVIDERS.items():
        for pattern in spec.patterns:
            if re.match(pattern, model, re.IGNORECASE):
                return name

    # Priority 4: unknown model -> actionable validation, never a crash.
    raise UnsupportedProviderError(
        f"Provider could not be auto-detected for model {model!r}. "
        "Select a provider or enter a Base URL. "
        "Any OpenAI-compatible endpoint works with any model name."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _provider_requires_api_key(provider: str, base_url: Optional[str]) -> bool:
    spec = _PROVIDERS.get(provider)
    if spec is None:
        return True
    if not spec.api_key_required:
        return False
    # Local endpoints normally need no key even when the provider is generic.
    lower = _normalize_base_url(base_url)
    if any(h in lower for h in _LOCAL_HOST_MARKERS):
        return False
    return True


def _default_base_url_for(provider: str) -> Optional[str]:
    spec = _PROVIDERS.get(provider)
    return spec.default_base_url if spec is not None else None


def validate_config(config: LLMConfig) -> LLMConfig:
    """Validate a runtime config, raising user-friendly exceptions.

    Returns a normalized copy with ``provider`` resolved, well-known
    default Base URLs applied, and an API-key policy enforced (local
    servers such as Ollama / LM Studio / vLLM do not require a key).
    """
    if not isinstance(config, LLMConfig):
        raise InvalidConfigurationError(
            f"Expected an LLMConfig instance, got {type(config).__name__}."
        )

    model = (config.model or "").strip()
    if not model:
        raise InvalidConfigurationError(
            "No model name provided. Set 'model' in the runtime config, or "
            "pass model=... to get_chat_model()."
        )

    api_key = (config.api_key or "").strip()
    base_url = (config.base_url or "").strip() or None
    provider = resolve_provider(model, config.provider, base_url)

    temperature = 0.2 if config.temperature is None else config.temperature
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        raise InvalidConfigurationError(
            f"Invalid temperature {config.temperature!r}: expected a number "
            "between 0.0 and 2.0."
        ) from None
    if not (0.0 <= temperature <= 2.0):
        raise InvalidConfigurationError(
            f"Invalid temperature {temperature}: must be between 0.0 and 2.0."
        )

    if config.max_tokens is not None:
        try:
            if int(config.max_tokens) <= 0:
                raise InvalidConfigurationError(
                    f"Invalid max_tokens {config.max_tokens!r}: must be a positive integer."
                )
        except (TypeError, ValueError):
            raise InvalidConfigurationError(
                f"Invalid max_tokens {config.max_tokens!r}: must be a positive integer."
            ) from None

    if config.timeout is not None and config.timeout <= 0:
        raise InvalidConfigurationError(
            f"Invalid timeout {config.timeout}: must be a positive number of seconds."
        )

    if config.max_retries is not None and config.max_retries < 0:
        raise InvalidConfigurationError(
            f"Invalid max_retries {config.max_retries}: must be >= 0."
        )

    # A generic OpenAI-compatible endpoint without a Base URL cannot work.
    if provider == OPENAI_COMPATIBLE_PROVIDER and not base_url:
        raise InvalidConfigurationError(
            "Provider 'OpenAI Compatible' requires a Base URL (e.g. "
            "https://integrate.api.nvidia.com/v1). Enter one or select a "
            "provider from the list."
        )

    # API-key policy: required for cloud providers, optional for local ones.
    if _provider_requires_api_key(provider, base_url):
        if not api_key:
            raise MissingAPIKeyError(
                f"No API key provided for provider {provider!r} / model {model!r}. "
                "Enter the API key in AI Model Settings, or pass api_key=... "
                "to get_chat_model() / configure_llm(...)."
            )
    else:
        # Local servers ignore the key, but the OpenAI SDK still wants a value.
        api_key = api_key or _LOCAL_PLACEHOLDER_KEY

    # Apply the provider's well-known default Base URL when none was given.
    if base_url is None:
        base_url = _default_base_url_for(provider)

    return replace(
        config,
        model=model,
        api_key=api_key,
        temperature=temperature,
        base_url=base_url,
        provider=provider,
    )


def validate_configuration(*, provider: Optional[str] = None, model: str = "",
                           api_key: str = "", base_url: Optional[str] = None,
                           temperature: float = 0.2, max_tokens: Optional[int] = None,
                           timeout: float = 120.0, max_retries: int = 0,
                           response_format: Optional[dict] = None,
                           config_name: Optional[str] = None) -> LLMConfig:
    """Public keyword-style validator used by the UI / config store.

    Equivalent to ``validate_config(LLMConfig(...))``.  Raises
    :class:`MissingAPIKeyError`, :class:`UnsupportedProviderError` or
    :class:`InvalidConfigurationError` with actionable messages.
    """
    return validate_config(LLMConfig(
        model=model, api_key=api_key, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout, max_retries=max_retries,
        base_url=base_url, response_format=response_format,
        provider=provider, config_name=config_name,
    ))


# ---------------------------------------------------------------------------
# Active runtime configuration
# ---------------------------------------------------------------------------

_runtime_config: Optional[LLMConfig] = None

_legacy_source_logged = False


def _log_config_activation(config: Optional[LLMConfig], source: str) -> None:
    """Safe diagnostic log: config name + provider + model only.

    The API key is NEVER included in this output.
    """
    if config is None:
        print(f"[llm_provider] No runtime model configured; using legacy env defaults "
              f"(source: {source})")
        return
    name = f" ({config.config_name})" if config.config_name else ""
    print(f"[llm_provider] Active model: provider={config.provider} "
          f"model={config.model}{name} (source: {source})")


def set_runtime_config(config: LLMConfig) -> LLMConfig:
    """Set the active runtime configuration used by ``get_chat_model()``.

    Validates the config first and returns the normalized copy.  The API key
    is registered for redaction so it can never leak into logs/errors.

    Replacing the runtime configuration also invalidates every cached LLM
    client, so the next ``get_chat_model()`` call builds the client for the
    newly activated model/provider instead of reusing a stale one.
    """
    global _runtime_config, _legacy_source_logged
    normalized = validate_config(config)
    _register_secret(normalized.api_key)
    _runtime_config = normalized
    _build_failover_model.cache_clear()
    _KEY_REGISTRY.clear()
    _legacy_source_logged = False
    _log_config_activation(normalized, source="runtime")
    return normalized


def configure_llm(*, model: str, api_key: str = "", temperature: float = 0.2,
                  max_tokens: Optional[int] = None, timeout: float = 120.0,
                  max_retries: int = 0, base_url: Optional[str] = None,
                  response_format: Optional[dict] = None,
                  provider: Optional[str] = None,
                  config_name: Optional[str] = None) -> LLMConfig:
    """Clean public interface: configure the active runtime LLM from values.

    Equivalent to ``set_runtime_config(LLMConfig(...))`` but written for
    callers that prefer plain keyword arguments.
    """
    return set_runtime_config(LLMConfig(
        model=model, api_key=api_key, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout, max_retries=max_retries,
        base_url=base_url, response_format=response_format,
        provider=provider, config_name=config_name,
    ))


def get_runtime_config() -> Optional[LLMConfig]:
    """Return the active runtime config (or None if none was set)."""
    return _runtime_config


def clear_runtime_config() -> None:
    """Forget the active runtime config (env defaults apply again)."""
    global _runtime_config, _legacy_source_logged
    _runtime_config = None
    _build_failover_model.cache_clear()
    _KEY_REGISTRY.clear()
    _legacy_source_logged = False
    _log_config_activation(None, source="legacy")


# ---------------------------------------------------------------------------
# Client factory -- the single dynamic builder
# ---------------------------------------------------------------------------

def build_chat_model(config: LLMConfig) -> BaseChatModel:
    """Build the LangChain chat model for a validated runtime config."""
    normalized = validate_config(config)
    spec = _PROVIDERS[normalized.provider]
    return spec.build(normalized)


def create_primary_client(*, model: Optional[str] = None,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          response_format: Optional[dict] = None) -> BaseChatModel:
    """[Deprecated] Legacy Mistral primary builder.

    Kept for backward compatibility with any external caller.  Prefer
    :func:`build_chat_model` / :func:`get_chat_model` which are fully dynamic.
    """
    if _runtime_config is not None:
        # An active runtime config takes priority over the legacy env
        # defaults, so this legacy helper can never build a Mistral client
        # that overrides the activated runtime model.
        active = replace(
            _runtime_config,
            model=model if model is not None else _runtime_config.model,
            temperature=temperature if temperature is not None else _runtime_config.temperature,
            max_tokens=max_tokens if max_tokens is not None else _runtime_config.max_tokens,
            response_format=response_format if response_format is not None else _runtime_config.response_format,
        )
        return build_chat_model(active)

    config = LLMConfig(
        model=model or MISTRAL_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=MISTRAL_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens,
        timeout=MISTRAL_TIMEOUT,
        max_retries=MISTRAL_MAX_RETRIES,
        response_format=response_format,
        provider="mistral",
    )
    return build_chat_model(config)


def create_fallback_client(*, model: Optional[str] = None,
                           temperature: Optional[float] = None,
                           max_tokens: Optional[int] = None,
                           response_format: Optional[dict] = None) -> Optional[BaseChatModel]:
    """Build the fallback chat model from ``FALLBACK_PROVIDER``.

    Returns None when no fallback is configured (or its API key is missing --
    a missing fallback key must never crash the app; it simply disables the
    fallback with a clear console note).
    """
    provider = FALLBACK_PROVIDER
    if provider in _FALLBACK_DISABLED_VALUES:
        return None
    if not FALLBACK_API_KEY:
        print(f"⚠️  FALLBACK_PROVIDER={provider!r} set but FALLBACK_API_KEY is missing — "
              "fallback disabled. Set FALLBACK_API_KEY to enable it.")
        return None

    resolved_model = model or FALLBACK_MODEL or _DEFAULT_FALLBACK_MODELS.get(provider, "gpt-4o-mini")
    temperature = FALLBACK_TEMPERATURE if temperature is None else temperature
    max_retries = FALLBACK_MAX_RETRIES
    timeout = FALLBACK_TIMEOUT

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "temperature": temperature,
                "max_retries": max_retries,
                "timeout": timeout,
                "api_key": FALLBACK_API_KEY,
            }
            if FALLBACK_BASE_URL:
                kwargs["base_url"] = FALLBACK_BASE_URL
            if response_format is not None:
                kwargs["model_kwargs"] = {"response_format": response_format}
            return ChatOpenAI(**kwargs)  # type: ignore[call-arg]

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(  # type: ignore[call-arg]
                model=resolved_model,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                api_key=FALLBACK_API_KEY,
            )

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(  # type: ignore[call-arg]
                model=resolved_model,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                max_output_tokens=max_tokens or MISTRAL_MAX_TOKENS,
                api_key=FALLBACK_API_KEY,
            )

        if provider == "mistral":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(  # type: ignore[call-arg]
                model=resolved_model,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                max_tokens=max_tokens or MISTRAL_MAX_TOKENS,
                api_key=FALLBACK_API_KEY,
            )
    except ImportError as exc:  # pragma: no cover
        raise PermanentProviderError(
            f"Fallback provider {provider!r} requires an SDK that is not installed: {exc}"
        ) from exc

    raise PermanentProviderError(
        f"Unsupported FALLBACK_PROVIDER={provider!r}. "
        "Use one of: openai, anthropic, google, mistral (or leave empty to disable)."
    )


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------

def print_provider_status() -> None:
    """Print the provider configuration banner (keys never printed)."""
    global _status_printed
    runtime = _runtime_config
    if runtime is not None:
        active_provider = runtime.provider or "unknown"
        active_model = runtime.model
        source = f"runtime config{runtime.config_name and ' (' + runtime.config_name + ')' or ''}"
    else:
        active_model = MISTRAL_MODEL
        source = "environment defaults"
        if not MISTRAL_API_KEY:
            active_provider = "not configured (set a runtime config or MISTRAL_API_KEY)"
        else:
            try:
                active_provider = resolve_provider(active_model, None, None)
            except UnsupportedProviderError:
                active_provider = "unknown (select a provider or enter a Base URL)"
    fallback_name = FALLBACK_PROVIDER if FALLBACK_PROVIDER not in _FALLBACK_DISABLED_VALUES else "Not configured"
    fallback_model = (
        FALLBACK_MODEL
        or _DEFAULT_FALLBACK_MODELS.get(FALLBACK_PROVIDER, "-")
        if FALLBACK_PROVIDER not in _FALLBACK_DISABLED_VALUES
        else "-"
    )
    print("=" * 50)
    print("LLM PROVIDER STATUS")
    print("=" * 50)
    print(f"Active Provider   : {active_provider}")
    print(f"Active Model      : {active_model}")
    print(f"Config Source     : {source}")
    print(f"Fallback Provider : {fallback_name}")
    print(f"Fallback Model    : {fallback_model}")
    print("=" * 50)
    _status_printed = True


# ---------------------------------------------------------------------------
# Failover chat model -- the object every LangChain chain keeps using
# ---------------------------------------------------------------------------

class FailoverChatModel(BaseChatModel):
    """LangChain-compatible chat model that tries primary then fallback.

    Compatible with ``chain.invoke(...)``, ``prompt | llm | parser``,
    ``llm.invoke(...)`` and ``llm.bind(...)`` because it IS a BaseChatModel.
    The primary is the dynamically-selected provider; the optional fallback
    is still driven by the legacy ``FALLBACK_*`` environment vars.
    """

    _llm_type: ClassVar[str] = "failover_chat_model"
    model_name: str = "failover_chat_model"

    primary_name: str = ""
    fallback_name: str = ""

    _primary: Any = PrivateAttr()
    _fallback: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, primary: BaseChatModel,
                 fallback: Optional[BaseChatModel] = None,
                 primary_name: str = "",
                 fallback_name: Optional[str] = None,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self._primary = primary
        self._fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name or (FALLBACK_PROVIDER if fallback else "")

    @property
    def primary(self) -> BaseChatModel:
        return self._primary

    @property
    def fallback(self) -> Optional[BaseChatModel]:
        return self._fallback

    # -- core dispatch ---------------------------------------------------

    def _run_primary(self, messages, stop=None, run_manager=None, **kwargs):
        return self._primary._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _run_fallback(self, messages, stop=None, run_manager=None, **kwargs):
        return self._fallback._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        try:
            result = self._run_primary(messages, stop=stop, run_manager=run_manager, **kwargs)
            return result
        except Exception as primary_error:
            category = classify_error(primary_error)
            if category == "permanent":
                # Auth / configuration problems are deterministic: no retry,
                # no fallback (fallback would hit the same class of error).
                raise _redact_exception(primary_error)
            if self._fallback is None:
                raise _redact_exception(primary_error)

            self._log_failover(primary_error, category)

            try:
                result = self._run_fallback(messages, stop=stop, run_manager=run_manager, **kwargs)
                print("✅ Fallback provider successful")
                return result
            except Exception as fallback_error:
                raise ProviderExhaustedError(
                    "Both LLM providers failed for one request. "
                    f"Primary ({self.primary_name}): {_safe_text(primary_error)} | "
                    f"Fallback ({self.fallback_name}): {_safe_text(fallback_error)}",
                    primary_error=_redact_exception(primary_error),
                    fallback_error=_redact_exception(fallback_error),
                    rate_limited=(category == "rate_limit"),
                    status_code=_status_code_of(fallback_error),
                ) from _redact_exception(fallback_error)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        try:
            return await self._primary._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as primary_error:
            category = classify_error(primary_error)
            if category == "permanent" or self._fallback is None:
                raise _redact_exception(primary_error)
            self._log_failover(primary_error, category)
            try:
                result = await self._fallback._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                print("✅ Fallback provider successful")
                return result
            except Exception as fallback_error:
                raise ProviderExhaustedError(
                    "Both LLM providers failed for one request."
                    f"Primary ({self.primary_name}): {_safe_text(primary_error)} | "
                    f"Fallback ({self.fallback_name}): {_safe_text(fallback_error)}",
                    primary_error=_redact_exception(primary_error),
                    fallback_error=_redact_exception(fallback_error),
                    rate_limited=(category == "rate_limit"),
                    status_code=_status_code_of(fallback_error),
                ) from _redact_exception(fallback_error)

    def _log_failover(self, primary_error: Exception, category: str) -> None:
        status = _status_code_of(primary_error)
        hint = _extract_retry_after(primary_error)
        if category == "rate_limit" or status == 429:
            print(f"⚠️  Primary provider rate limited (HTTP 429)"
                  + (f" — Retry-After {hint:.0f}s" if hint else "") + "")
        elif status in (500, 502, 503, 504, 529):
            print(f"⚠️  Primary provider temporary failure (HTTP {status})")
        else:
            print(f"⚠️  Primary provider temporary failure: {_safe_text(primary_error)[:200]}")
        print(f"🔄 Switching to fallback provider ({self.fallback_name or 'none'})...")


def _key_fingerprint(api_key: str) -> str:
    """Stable short hash of an API key -- used ONLY as a cache identity.

    The raw key is never part of any cache key, fingerprint or log.
    Identical keys (same literal) always produce the same fingerprint, so
    repeated calls for the same configuration still share one client.
    """
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


# In-memory fingerprint -> key recovery map (populated by every
# get_chat_model() call).  The key is NEVER part of a cache key, fingerprint
# or log -- this registry only lets the cached builder recover the key it
# needs to construct the client.  It is cleared together with the client
# cache whenever the runtime configuration changes.
_KEY_REGISTRY: dict[str, str] = {}


def _register_key_fingerprint(api_key: str) -> str:
    fp = _key_fingerprint(api_key)
    if fp:
        _KEY_REGISTRY[fp] = api_key
    return fp


def _canonical_response_format(
    response_format: Optional[dict],
) -> Optional[str]:
    """Canonical, deterministic cache-identity representation of a
    response_format dict.

    JSON canonicalization keeps every JSON value type distinct in the cache
    key - True vs "True", 10 vs "10", 0.5 vs "0.5", nested dicts and lists -
    so equivalent configurations collide on the same key while meaningfully
    different ones never share a cached client.  The runtime response_format
    is rebuilt from this string with json.loads(), which restores the
    ORIGINAL Python types exactly (bools stay bools, ints stay ints, ...).
    Never store raw API keys in cache keys.
    """
    if response_format is None:
        return None
    return json.dumps(
        response_format,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _client_key(*, model: str, api_key: str, temperature: float,
                max_tokens: Optional[int], base_url: Optional[str],
                provider: str, timeout: float, max_retries: int,
                response_format: Optional[dict], role: Optional[str]) -> tuple:
    # fmt_key is IDENTITY ONLY -- a canonical type-preserving string.  The
    # original response_format dict is rebuilt separately (json.loads) in
    # _build_failover_model(), so the runtime value never suffers type loss.
    fmt_key = _canonical_response_format(response_format)
    # api_key enters the key ONLY as a one-way hash -- never the raw secret.
    return (model, _key_fingerprint(api_key), temperature, max_tokens, base_url,
            provider, timeout, max_retries, fmt_key, role)


@lru_cache(maxsize=32)
def _build_failover_model(key: tuple) -> FailoverChatModel:
    (model, _key_fp, temperature, max_tokens, base_url, provider,
     timeout, max_retries, fmt_key, role) = key
    # Rebuild from the canonical string: the json round-trip restores the
    # ORIGINAL Python types (bool stays bool, int stays int, float stays
    # float, nested dicts/lists stay nested) -- never the lossy str() form.
    response_format = None
    if fmt_key is not None:
        response_format = json.loads(fmt_key)
    # The raw key is never part of the cache key -- only its one-way
    # fingerprint.  The key itself is recovered from the in-memory registry
    # populated by every get_chat_model() call.  Provider + model + base_url
    # determine WHICH endpoint semantics apply; the fingerprint guarantees a
    # DIFFERENT client whenever the key changes.
    effective_key = _KEY_REGISTRY.get(_key_fp, "") if _key_fp else ""
    config = validate_config(LLMConfig(
        model=model, api_key=effective_key, temperature=temperature,
        max_tokens=max_tokens, timeout=timeout, max_retries=max_retries,
        base_url=base_url, response_format=response_format, provider=provider,
    ))
    primary = build_chat_model(config)
    # The legacy env fallback (FALLBACK_PROVIDER / FALLBACK_API_KEY) is only
    # attached in pure legacy mode.  When a runtime config is active it must
    # not silently override the activated model.
    fallback = None
    if _runtime_config is None:
        fallback = create_fallback_client(model=model, temperature=temperature,
                                          max_tokens=max_tokens,
                                          response_format=response_format)
    return FailoverChatModel(primary=primary, fallback=fallback,
                             primary_name=config.provider,
                             fallback_name=FALLBACK_PROVIDER if fallback else None)


# ---------------------------------------------------------------------------
# Public factory -- the ONLY entry point the rest of the app should use
# ---------------------------------------------------------------------------

def get_chat_model(model: Optional[str] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   response_format: Optional[dict] = None,
                   role: Optional[str] = None,
                   api_key: Optional[str] = None,
                   base_url: Optional[str] = None,
                   provider: Optional[str] = None,
                   config: Optional[LLMConfig] = None) -> FailoverChatModel:
    """Return the centralized, LangChain-compatible chat model.

    Resolution order:

    1. ``config`` -- an explicit :class:`LLMConfig` wins over everything.
    2. Explicit identity args (``api_key`` / ``base_url`` / ``provider`` /
       ``model``) -- built on top of the active runtime config (or env
       defaults when none is set).
    3. Otherwise the active runtime config set via ``configure_llm`` /
       ``set_runtime_config`` / ``activate_configuration``.
    4. Otherwise the legacy environment defaults (``MISTRAL_*``).

    ``role`` is a free-text label used only for cache separation and logs
    (e.g. "Translator", "Summarizer").
    """
    global _status_printed
    if not _status_printed:
        print_provider_status()

    if config is not None:
        active = validate_config(config)
    else:
        active = _resolve_active_config(
            model=model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, api_key=api_key,
            base_url=base_url, provider=provider,
        )
    _register_secret(active.api_key)
    _register_key_fingerprint(active.api_key)

    return _build_failover_model(_client_key(
        model=active.model, api_key=active.api_key, temperature=active.temperature,
        max_tokens=active.max_tokens, base_url=active.base_url,
        provider=active.provider, timeout=active.timeout,
        max_retries=active.max_retries, response_format=active.response_format,
        role=role,
    ))


def _resolve_active_config(*, model: Optional[str], temperature: Optional[float],
                           max_tokens: Optional[int],
                           response_format: Optional[dict],
                           api_key: Optional[str], base_url: Optional[str],
                           provider: Optional[str]) -> LLMConfig:
    """Resolve the effective config from runtime config / explicit args / env.

    An active runtime config always takes priority over the legacy ``.env``
    defaults.  Explicit arguments that merely echo the legacy env values
    (``MISTRAL_MODEL`` / ``MISTRAL_API_KEY`` / ``FALLBACK_BASE_URL``) are
    treated as "not provided", so a legacy/default Mistral configuration can
    never silently override the activated runtime model.
    """
    global _legacy_source_logged
    runtime = _runtime_config

    if runtime is not None:
        # Neutralize legacy env echoes so they cannot override the runtime
        # configuration.  (In pure legacy mode these are the intended defaults
        # and are not neutralized.)
        if model is not None and model.strip() == MISTRAL_MODEL:
            model = None
        if api_key is not None and MISTRAL_API_KEY and api_key.strip() == MISTRAL_API_KEY:
            api_key = None
        if base_url is not None and base_url == FALLBACK_BASE_URL:
            base_url = None

        identity_override = (api_key is not None or base_url is not None
                             or provider is not None or model is not None)

        if not identity_override:
            # Pure tuning overrides on top of the active runtime config.
            return validate_config(replace(
                runtime,
                temperature=temperature if temperature is not None else runtime.temperature,
                max_tokens=max_tokens if max_tokens is not None else runtime.max_tokens,
                response_format=response_format if response_format is not None else runtime.response_format,
            ))

        # Explicit identity override.  Caller-supplied identity fields win;
        # the rest inherit from the runtime config.
        eff_model = model if model is not None else runtime.model
        eff_key = api_key if api_key is not None else runtime.api_key
        eff_base = base_url if base_url is not None else runtime.base_url

        if provider is not None:
            eff_provider = provider_alias(provider)
        elif model is not None:
            eff_provider = None  # re-detect from the new model
        else:
            eff_provider = runtime.provider

        # Drop an inherited key/base_url that does not belong to the
        # provider this call will actually use, so a wrong key is never
        # sent to an API.
        if eff_provider is not None and eff_provider != runtime.provider:
            eff_key = api_key if api_key is not None else ""
            eff_base = base_url if base_url is not None else None
        elif eff_provider is None:
            try:
                detected = resolve_provider(eff_model, None, eff_base)
            except UnsupportedProviderError:
                detected = None
            if detected != runtime.provider:
                eff_key = api_key if api_key is not None else ""
                eff_base = base_url if base_url is not None else None

        return validate_config(replace(
            runtime,
            model=eff_model,
            api_key=eff_key,
            temperature=temperature if temperature is not None else runtime.temperature,
            max_tokens=max_tokens if max_tokens is not None else runtime.max_tokens,
            base_url=eff_base,
            response_format=response_format if response_format is not None else runtime.response_format,
            provider=eff_provider,
        ))

    # No runtime config: legacy environment defaults (+ any explicit args).
    if not _legacy_source_logged:
        _legacy_source_logged = True
        _log_config_activation(None, source="legacy")
    return validate_config(LLMConfig(
        model=model if model is not None else MISTRAL_MODEL,
        api_key=api_key if api_key is not None else MISTRAL_API_KEY,
        temperature=temperature if temperature is not None else MISTRAL_TEMPERATURE,
        max_tokens=max_tokens,
        timeout=MISTRAL_TIMEOUT,
        max_retries=MISTRAL_MAX_RETRIES,
        base_url=base_url,
        response_format=response_format,
        provider=provider,
    ))


def get_llm(model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            response_format: Optional[dict] = None,
            role: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = None,
            config: Optional[LLMConfig] = None) -> FailoverChatModel:
    """Backward-compatible alias of :func:`get_chat_model`."""
    return get_chat_model(model=model, temperature=temperature, max_tokens=max_tokens,
                          response_format=response_format, role=role,
                          api_key=api_key, base_url=base_url, provider=provider,
                          config=config)


# ---------------------------------------------------------------------------
# LLM fingerprints -- version AI-generated outputs for cache compatibility
# ---------------------------------------------------------------------------

def llm_identity(config: Optional[LLMConfig] = None,
                 *, provider: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None) -> tuple[str, str, str]:
    """Return ``(provider, model, base_url)`` of the LLM that WILL be used.

    Used to persist reproducible output metadata -- never contains an API key.

    Resolution: explicit ``config`` > explicit args > active runtime config
    > legacy env defaults.
    """
    if config is not None and isinstance(config, LLMConfig):
        p, m, b = config.provider, (config.model or "").strip(), _normalize_base_url(config.base_url)
        if not p:
            try:
                p = resolve_provider(m, None, None)
            except UnsupportedProviderError:
                p = "unknown"
        return p, m, b

    if _runtime_config is not None:
        return llm_identity(_runtime_config)

    m = (model or "").strip() or MISTRAL_MODEL
    b = _normalize_base_url(base_url)
    p = provider_alias(provider or "")
    if not p:
        try:
            p = resolve_provider(m, None, base_url or None)
        except UnsupportedProviderError:
            p = "unknown"
    return p, m, b


def llm_fingerprint(config: Optional[LLMConfig] = None,
                    *, provider: Optional[str] = None,
                    model: Optional[str] = None,
                    base_url: Optional[str] = None) -> str:
    """Stable short fingerprint of the ACTIVE LLM identity.

    Includes provider, model and normalized base_url -- NEVER the API key.
    Two configurations that would be served differently (e.g. Mistral
    ``mistral-small`` vs NVIDIA ``meta/llama-...``) always differ.  Used by
    the database to decide whether cached AI outputs (summaries, titles,
    extractions) may be reused.
    """
    p, m, b = llm_identity(config, provider=provider, model=model, base_url=base_url)
    if not m and not p and not b:
        return ""
    payload = "|".join([p or "", m or "", b or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def fingerprints_compatible(cached_fingerprint: Optional[str],
                            active_fingerprint: Optional[str]) -> bool:
    """True only when a cached AI output was produced by the SAME LLM.

    An empty/None fingerprint never matches -- a cached summary produced by
    an unknown (legacy) configuration is not reused blindly.
    """
    if not cached_fingerprint or not active_fingerprint:
        return False
    return cached_fingerprint == active_fingerprint


def translation_fingerprint(*, enabled: bool, source_language: str,
                            target_language: str, provider: str, model: str) -> str:
    """Fingerprint of a translation request (cache-compatibility key).

    ``enabled`` -> target language -> translation engine identity, so a
    cached English translation is never served for a Urdu / Roman Urdu
    request (and vice versa).  No API key involved.
    """
    payload = "|".join([
        "1" if enabled else "0",
        (source_language or "").strip().lower(),
        (target_language or "").strip().lower(),
        provider_alias(provider),
        (model or "").strip(),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Persistent configuration store (multiple saved configurations)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
CONFIG_STORE_PATH = os.getenv(
    "LLM_CONFIGS_PATH", os.path.join(_DEFAULT_CONFIGS_DIR, "llm_configs.json")
)


class LLMConfigStore:
    """Persistent multi-configuration store (JSON-backed, chmod 600).

    Each saved configuration holds: name, provider, model, api_key,
    base_url, temperature (plus optional timeout / max_retries /
    max_tokens / response_format).  The API key is stored so the saved
    configuration is actually usable, but:

      * the file is created with mode 0o600 (owner read/write only),
      * keys are never printed or logged,
      * ``list()`` returns masked keys only,
      * every key is registered for redaction in error messages.

    Switching configurations re-activates the runtime LLM and clears all
    cached clients, so summary / RAG / extraction / translation all move to
    the newly selected model immediately.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or CONFIG_STORE_PATH)
        self._configs: dict[str, dict[str, Any]] = {}
        self._active_name: Optional[str] = None
        self._loaded = False

    # -- persistence ----------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"⚠️  Could not read LLM config store {self.path}: {exc} — starting empty.")
            return
        self._configs = dict(raw.get("configurations") or {})
        self._active_name = (raw.get("__meta__") or {}).get("active") or None
        for entry in self._configs.values():
            _register_secret(str(entry.get("api_key", "")))

    def _persist(self) -> None:
        payload = {
            "__meta__": {"active": self._active_name},
            "configurations": self._configs,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # -- CRUD -----------------------------------------------------------

    def save(self, config: LLMConfig, name: Optional[str] = None,
             overwrite: bool = True) -> str:
        """Validate and persist a configuration.  Returns its name."""
        # Load persisted configurations BEFORE name resolution: automatic
        # names, collisions, and the exists-check must run against the FULL
        # saved set, never a not-yet-loaded empty dict.  _load() is guarded
        # by self._loaded, so an already-loaded store is not re-read.
        self._load()
        normalized = validate_config(config)
        name = (name or normalized.config_name or "").strip()
        if not name:
            existing = set(self._configs.keys())
            index = 1
            while f"Configuration {index}" in existing:
                index += 1
            name = f"Configuration {index}"
        if not overwrite and name in self._configs:
            raise InvalidConfigurationError(
                f"A configuration named {name!r} already exists."
            )
        _register_secret(normalized.api_key)
        self._configs[name] = {
            "provider": normalized.provider,
            "model": normalized.model,
            "api_key": normalized.api_key,
            "base_url": normalized.base_url,
            "temperature": normalized.temperature,
            "timeout": normalized.timeout,
            "max_retries": normalized.max_retries,
            "max_tokens": normalized.max_tokens,
            "response_format": normalized.response_format,
        }
        self._persist()
        # Overwriting the ACTIVE configuration must update the running LLM.
        if self._active_name == name:
            self.activate(name)
        return name

    def get(self, name: str) -> Optional[LLMConfig]:
        self._load()
        entry = self._configs.get(name)
        if entry is None:
            return None
        return LLMConfig(
            model=entry["model"], api_key=entry["api_key"],
            temperature=entry["temperature"], base_url=entry.get("base_url"),
            timeout=entry.get("timeout", 120.0),
            max_retries=entry.get("max_retries", 0),
            max_tokens=entry.get("max_tokens"),
            response_format=entry.get("response_format"),
            provider=entry.get("provider"),
            config_name=name,
        )

    def names(self) -> list[str]:
        self._load()
        return sorted(self._configs.keys())

    def list(self, mask_keys: bool = True) -> list[dict[str, Any]]:
        """Row-safe listing for the UI.  Raw API keys never appear."""
        self._load()
        rows: list[dict[str, Any]] = []
        for name, entry in self._configs.items():
            rows.append({
                "name": name,
                "provider": entry.get("provider"),
                "model": entry.get("model"),
                "base_url": entry.get("base_url"),
                "temperature": entry.get("temperature"),
                "api_key_masked": mask_api_key(str(entry.get("api_key", ""))),
                "active": name == self._active_name,
            })
        return rows

    def active_name(self) -> Optional[str]:
        self._load()
        return self._active_name

    def activate(self, name: str) -> LLMConfig:
        """Activate a saved configuration: valid + sets runtime + clears cache.

        After this call every ``get_chat_model()`` (summary, RAG,
        extraction, translation) uses the newly selected provider/model.
        """
        config = self.get(name)
        if config is None:
            raise InvalidConfigurationError(
                f"No saved configuration named {name!r}. Saved: "
                f"{', '.join(self.names()) or '(none)'}."
            )
        normalized = set_runtime_config(config)
        self._active_name = name
        self._persist()
        return normalized

    def delete(self, name: str) -> bool:
        self._load()
        if name not in self._configs:
            return False
        del self._configs[name]
        if self._active_name == name:
            self._active_name = None
            clear_runtime_config()
        self._persist()
        return True


_store: Optional[LLMConfigStore] = None


def get_config_store() -> LLMConfigStore:
    """Module-level default configuration store (lazily created)."""
    global _store
    if _store is None:
        _store = LLMConfigStore()
    return _store


def set_config_store(store: LLMConfigStore) -> LLMConfigStore:
    """Swap the default store (used by tests / alternate data dirs)."""
    global _store
    _store = store
    return store


def save_configuration(config: LLMConfig, name: Optional[str] = None,
                       overwrite: bool = True) -> str:
    """Persist an LLM configuration (see :class:`LLMConfigStore`)."""
    return get_config_store().save(config, name=name, overwrite=overwrite)


def activate_configuration(name: str) -> LLMConfig:
    """Activate a saved configuration (updates the running LLM everywhere)."""
    return get_config_store().activate(name)


def delete_configuration(name: str) -> bool:
    """Delete a saved configuration."""
    return get_config_store().delete(name)


def list_configurations(mask_keys: bool = True) -> list[dict[str, Any]]:
    """UI-safe listing of saved configurations (keys masked)."""
    return get_config_store().list(mask_keys=mask_keys)


def restore_active_configuration() -> Optional[LLMConfig]:
    """Re-activate the previously active saved configuration, if any.

    Called once at application startup (from the UI).  Safe: a corrupted or
    invalid saved configuration degrades to a console warning instead of a
    crash -- the legacy env fallback still applies.
    """
    store = get_config_store()
    name = store.active_name()
    if not name:
        return None
    try:
        return store.activate(name)
    except Exception as exc:  # noqa: BLE001 -- startup must never crash
        print(f"⚠️  Could not restore saved LLM configuration {name!r}: {_redact(str(exc))}")
        return None


# ---------------------------------------------------------------------------
# Module self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nLLM provider module loaded successfully.")
    print_provider_status()
    print("\nRuntime configuration example:")
    print("  configure_llm(model='gpt-4o-mini', api_key='sk-...', temperature=0.3)")
    print("  llm = get_chat_model(model='mistral-large-latest', provider='mistral')")
    print("  llm = get_chat_model(model='meta/llama-3.1-70b-instruct', provider='nvidia',")
    print("                       api_key='...', base_url='https://integrate.api.nvidia.com/v1')")
    print("\nNo API call was made.")
