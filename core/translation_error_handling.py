"""Pure error-classification helpers for the Streamlit Roman Urdu UI.

Kept deliberately free of Streamlit and other UI imports so these helpers
can be unit-tested without a browser and reused anywhere in the app. Every
function here is pure: it inspects exception chains and returns plain,
user-safe data.

The backend (``core.roman_urdu_translator``) raises typed exceptions
(``RateLimitError``, ``RetryableAPIError``, ``PermanentAPIError``) and wraps
them in ``TranslationPipelineError``; these helpers unwrap that chain safely.
Classification is type-based; string matching is never required when the
typed exceptions are available.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional

from core.roman_urdu_translator import (
    PermanentAPIError,
    RateLimitError,
    RetryableAPIError,
    TranslationPipelineError,
)

# Hard ceiling so a crafted or cyclic exception chain can never loop forever.
MAX_EXCEPTION_CHAIN_LENGTH = 16

RATE_LIMIT_MESSAGE = (
    "⚠️ Mistral AI Rate Limit Reached\n\n"
    "The translation service is temporarily busy or your API rate limit has "
    "been reached.\n"
    "Please wait a few minutes and try again.\n\n"
    "Your original transcript has NOT been deleted."
)

TEMPORARY_ERROR_MESSAGE = (
    "⚠️ Temporary Translation Service Error\n\n"
    "The translation service is temporarily unavailable.\n"
    "Please try again shortly."
)

PERMANENT_ERROR_MESSAGE = (
    "❌ Translation Configuration Error\n\n"
    "The translation service could not be started correctly.\n"
    "Please check the API configuration."
)

UNEXPECTED_ERROR_MESSAGE = (
    "❌ Translation Failed\n\n"
    "An unexpected error occurred while generating the Roman Urdu translation."
)


def _iter_exception_chain(exc: Optional[BaseException]) -> Iterator[BaseException]:
    """Yield an exception and its ``__cause__``/``__context__`` chain, safely.

    Tracks visited object ids so a cyclic chain cannot loop forever, and caps
    the walk at ``MAX_EXCEPTION_CHAIN_LENGTH``.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    steps = 0
    while current is not None and steps < MAX_EXCEPTION_CHAIN_LENGTH:
        ident = id(current)
        if ident in seen:
            return
        seen.add(ident)
        yield current
        nested = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
        current = nested if isinstance(nested, BaseException) else None
        steps += 1


def _chain_matches(exc: Optional[BaseException], error_type: type) -> bool:
    return any(isinstance(err, error_type) for err in _iter_exception_chain(exc))


def contains_rate_limit(exc: Optional[BaseException]) -> bool:
    """True when a ``RateLimitError`` (or backend-marked 429) is in the chain."""
    for err in _iter_exception_chain(exc):
        if isinstance(err, RateLimitError):
            return True
        if getattr(err, "rate_limited", False) is True:
            return True
    return False


def get_retry_after(exc: Optional[BaseException]) -> Optional[float]:
    """Return the first Retry-After hint found in the chain, else None."""
    for err in _iter_exception_chain(exc):
        hint = getattr(err, "retry_after", None)
        if hint is not None:
            try:
                return float(hint)
            except (TypeError, ValueError):
                continue
    return None


def get_completed_chunk_count(exc: Optional[BaseException]) -> int:
    """Return how many translation chunks completed before the failure."""
    for err in _iter_exception_chain(exc):
        chunks = getattr(err, "completed_chunks", None)
        if isinstance(chunks, (list, tuple)) and chunks:
            return len(chunks)
    return 0


def get_partial_translation(exc: Optional[BaseException]) -> str:
    """Return preserved partial translation text when the backend exposed it."""
    for err in _iter_exception_chain(exc):
        partial = getattr(err, "partial_translation", None)
        if isinstance(partial, str) and partial.strip():
            return partial
    return ""


def _redact(value: str) -> str:
    secret = os.getenv("MISTRAL_API_KEY", "")
    if secret:
        value = value.replace(secret, "[REDACTED]")
    return value


def _safe_detail(exc: Optional[BaseException]) -> str:
    if exc is None:
        return ""
    return _redact(str(exc))


def classify_translation_error(exc: Optional[BaseException]) -> dict[str, Any]:
    """Classify a translation exception for UI rendering.

    Returns a dict with only safe, user-facing keys (never API keys or raw
    tracebacks):

    - ``category``: ``rate_limit`` | ``temporary`` | ``permanent`` | ``unexpected``
    - ``message``: user-facing message (may contain a single newline group)
    - ``detail``: redacted backend detail for logs / expanders (may be empty)
    - ``retry_after``: optional Retry-After hint in seconds
    - ``completed_chunks``: number of chunks preserved before the failure
    - ``partial_translation``: preserved partial text (empty when unavailable)

    Classification is type-based first (unwrapping the full cause/context
    chain); a plain exception with rate-limit-looking text but no typed
    exception is deliberately NOT classified as a rate limit.
    """
    if contains_rate_limit(exc):
        return {
            "category": "rate_limit",
            "message": RATE_LIMIT_MESSAGE,
            "detail": _safe_detail(exc),
            "retry_after": get_retry_after(exc),
            "completed_chunks": get_completed_chunk_count(exc),
            "partial_translation": get_partial_translation(exc),
        }
    if _chain_matches(exc, PermanentAPIError):
        return {
            "category": "permanent",
            "message": PERMANENT_ERROR_MESSAGE,
            "detail": _safe_detail(exc),
            "retry_after": None,
            "completed_chunks": get_completed_chunk_count(exc),
            "partial_translation": get_partial_translation(exc),
        }
    if _chain_matches(exc, RetryableAPIError):
        return {
            "category": "temporary",
            "message": TEMPORARY_ERROR_MESSAGE,
            "detail": _safe_detail(exc),
            "retry_after": get_retry_after(exc),
            "completed_chunks": get_completed_chunk_count(exc),
            "partial_translation": get_partial_translation(exc),
        }
    return {
        "category": "unexpected",
        "message": UNEXPECTED_ERROR_MESSAGE,
        "detail": _safe_detail(exc),
        "retry_after": None,
        "completed_chunks": get_completed_chunk_count(exc),
        "partial_translation": get_partial_translation(exc),
    }
