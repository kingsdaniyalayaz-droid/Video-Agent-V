"""
core/translator.py

Translation engine for Video Agent, backed by the centralized LLM
provider layer (core.llm_provider).

Pipeline:

    Whisper Transcript
            ↓
       Translation
            ↓
   Active LLM Provider (core.llm_provider)
            ↓
     Translated Text

Supported:
    - English
    - Urdu
    - Hindi
    - Hinglish

Uses:
    LangChain LCEL
    core.llm_provider (centralized model/provider management)
"""

from __future__ import annotations
import os

import math
import math
import re
import time
import threading

from dotenv import load_dotenv

from core.llm_provider import get_chat_model, get_runtime_config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

TRANSLATION_TOKEN_CHUNK_SIZE = int(os.getenv("TRANSLATION_TOKEN_CHUNK_SIZE", "1200"))
TRANSLATION_CHAR_CHUNK_SIZE = int(os.getenv("TRANSLATION_CHAR_CHUNK_SIZE", "3500"))

# Maximum number of retries after the initial request attempt.
TRANSLATION_MAX_RETRIES = 3
TRANSLATION_RETRY_BASE_DELAY = 1.0
TRANSLATION_RETRY_MAX_DELAY = 8.0

RETRY_AFTER_CEILING = min(float(os.getenv("RETRY_AFTER_CEILING", "60.0")), 120.0)
TOTAL_DEADLINE_SECONDS = float(os.getenv("TOTAL_DEADLINE_SECONDS", "120.0"))


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = {
    "english",
    "hindi",
    "urdu",
    "hinglish",
}


# ============================================================
# LANGUAGE ALIASES
# ============================================================

LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",

    "hi": "hindi",
    "hin": "hindi",

    "ur": "urdu",
    "urd": "urdu",

    "hing": "hinglish",
}


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================

def normalize_language(
    language: str,
) -> str:
    """
    Normalize language names and aliases.

    Examples:

        en       → english
        hi       → hindi
        ur       → urdu
        hing     → hinglish
    """

    if not language:

        raise ValueError(
            "Language cannot be empty."
        )

    normalized = (
        language
        .strip()
        .lower()
    )

    normalized = LANGUAGE_ALIASES.get(
        normalized,
        normalized,
    )

    return normalized


# ============================================================
# LANGUAGE VALIDATION
# ============================================================

def validate_language(
    language: str,
) -> str:
    """
    Normalize and validate language.
    """

    normalized = normalize_language(
        language
    )

    if normalized not in SUPPORTED_LANGUAGES:

        supported = ", ".join(
            sorted(SUPPORTED_LANGUAGES)
        )

        raise ValueError(
            f"Unsupported language: "
            f"{language}. "
            f"Supported languages: "
            f"{supported}"
        )

    return normalized


# ============================================================
# LLM — CENTRALIZED PROVIDER
# ============================================================

def get_llm():
    """
    Return the translated-text LLM from the centralized provider layer
    (core.llm_provider).

    There is deliberately NO independent model cache here: the centralized
    provider already manages model caching and invalidates it on runtime
    configuration changes.  Every call resolves the currently ACTIVE
    provider/model, so runtime provider switching (Mistral -> Groq ...)
    takes effect on the next translation.

    The active provider, model, and temperature always come from the
    centralized runtime configuration.
    """

    return get_chat_model(role="Translator")


# ============================================================
# TRANSLATION PROMPT
# ============================================================

TRANSLATION_SYSTEM_PROMPT = """
You are a professional multilingual translator
working inside an AI Video Assistant.

Your task is to translate the provided text from
the source language into the target language.

STRICT RULES:

1. Preserve the original meaning and intent.
2. Do not summarize.
3. Do not add information.
4. Do not remove information.
5. Preserve names whenever possible.
6. Preserve numbers exactly.
7. Preserve dates exactly.
8. Preserve IDs exactly.
9. Preserve quantities exactly.
10. Preserve technical terminology when appropriate.
11. Preserve company names.
12. Preserve product names.
13. Preserve order numbers.
14. Preserve programming terms.
15. Handle Hindi naturally.
16. Handle Urdu naturally.
17. Handle Hinglish naturally.
18. Do not translate proper nouns unnecessarily.
19. Do not provide explanations.
20. Do not add headings.
21. Do not add notes.
22. Return ONLY the translated text.

IMPORTANT:

If the source text contains technical content,
maintain the technical meaning accurately.

If the source is Hinglish, understand the mixed
Hindi-English context before translating it.
"""


translation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            TRANSLATION_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Source Language:
{source_language}

Target Language:
{target_language}

Text:

--- TEXT START ---

{text}

--- TEXT END ---
""",
        ),
    ]
)


# ============================================================
# LCEL CHAIN
# ============================================================

def get_translation_chain():
    """
    Create the reusable LCEL translation chain.

    Flow:

        Prompt
          ↓
        Active LLM (core.llm_provider)
          ↓
        String Output
    """

    llm = get_llm()

    return (
        translation_prompt
        | llm
        | StrOutputParser()
    )


# ============================================================
# TEXT CHUNKING
# ============================================================
def _split_text_into_chunks(
    text: str,
) -> list[str]:
    """
    Split text into ordered chunks without dropping or duplicating content.
    """
    if not text:
        return []

    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=TRANSLATION_TOKEN_CHUNK_SIZE,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
    except Exception as e:
        print(f"⚠️ Tiktoken encoder not available ({e}). Falling back to char splitter.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=TRANSLATION_CHAR_CHUNK_SIZE,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

    chunks = splitter.split_text(text)
    return chunks


# ============================================================
# RETRY / BACKOFF
# ============================================================
def _is_retryable_translation_error(
    exc: Exception,
) -> bool:
    """Return True only for likely-transient translation failures."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    message = str(exc).lower()

    # Authentication, authorization, malformed requests, and missing
    # resources are normally permanent and must not be retried.
    permanent_markers = (
        "invalid api key",
        "api key is invalid",
        "authentication",
        "unauthorized",
        "forbidden",
        "bad request",
        "malformed request",
        "unsupported language",
        "invalid configuration",
    )
    if any(marker in message for marker in permanent_markers):
        return False

    status_codes = {
        int(code)
        for code in re.findall(r"[245][0-9][0-9]", message)
    }
    if status_codes & {429, 500, 502, 503, 504}:
        return True
    if status_codes & {400, 401, 403, 404}:
        return False

    transient_markers = (
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
        "service unavailable",
        "gateway",
        "server error",
        "transient",
    )
    return any(marker in message for marker in transient_markers)


def extract_retry_after(exc: Exception, default: float = 0.0) -> float:
    """Extract and validate the Retry-After value from an exception."""
    retry_after = getattr(exc, "retry_after", None)
    candidates = [retry_after]

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            candidates.extend([headers.get("retry-after"), headers.get("Retry-After")])
        except (AttributeError, TypeError):
            pass

    for value in candidates:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delay) and delay > 0:
            return min(delay, RETRY_AFTER_CEILING)

    match = re.search(
        r"(?:please\s+)?(?:try\s+again\s+in|retry\s+after|retry\s+in)\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*"
        r"(milliseconds?|ms|seconds?|secs?|sec|s)\b",
        str(exc),
        flags=re.IGNORECASE,
    )
    if match is not None:
        try:
            delay = float(match.group(1))
            unit = match.group(2).lower()
            if unit in {"millisecond", "milliseconds", "ms"}:
                delay /= 1000.0
            if math.isfinite(delay) and delay > 0:
                return min(delay, RETRY_AFTER_CEILING)
        except (TypeError, ValueError):
            pass

    return default


def interruptible_sleep(delay: float, cancel_event: threading.Event | None = None) -> None:
    """Sleep for delay seconds, but wake up early if cancel_event is set."""
    if delay <= 0:
        return
    if cancel_event is not None:
        cancel_event.wait(timeout=delay)
    else:
        # We can still use threading.Event().wait() as a safer blocking sleep
        threading.Event().wait(timeout=delay)


def _invoke_translation_with_retry(
    chain,
    payload: dict[str, str],
    *,
    chunk_index: int,
    total_chunks: int,
    cancel_event: threading.Event | None = None,
):
    """Invoke one chunk with a bounded retry, exponential backoff, and Retry-After support."""
    attempt = 1
    total_attempts = TRANSLATION_MAX_RETRIES + 1
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS

    while True:
        try:
            return chain.invoke(payload)
        except Exception as exc:
            retryable = _is_retryable_translation_error(exc)
            retries_used = attempt - 1
            if not retryable or retries_used >= TRANSLATION_MAX_RETRIES:
                raise RuntimeError(
                    f"Translation failed for chunk "
                    f"{chunk_index}/{total_chunks}. "
                    f"Source: {payload['source_language']}. "
                    f"Target: {payload['target_language']}. "
                    f"Reason: {exc}"
                ) from exc

            exp_delay = min(
                TRANSLATION_RETRY_BASE_DELAY * (2 ** retries_used),
                TRANSLATION_RETRY_MAX_DELAY,
            )
            server_delay = extract_retry_after(exc)
            delay = min(max(server_delay, exp_delay), RETRY_AFTER_CEILING)

            now = time.monotonic()
            if now + delay > deadline:
                raise TimeoutError(
                    f"Retry budget exceeded. "
                    f"Requested delay: {delay:.1f}s would exceed deadline."
                ) from exc

            print(
                f"Retrying translation chunk "
                f"{chunk_index}/{total_chunks}"
            )
            print(
                f"Attempt {attempt + 1}/{total_attempts}"
            )
            print(f"Reason: {exc}")
            print(f"Waiting {delay:.1f} seconds")
            interruptible_sleep(delay, cancel_event)
            attempt += 1


# ============================================================
# TRANSLATE TEXT
# ============================================================
def translate_text(

    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Translate text from source language
    to target language.

    Args:
        text:
            Text to translate.

        source_language:
            Source language.

        target_language:
            Target language.

    Returns:
        Translated text.
    """

    # --------------------------------------------------------
    # Validate text
    # --------------------------------------------------------

    if not text or not text.strip():

        raise ValueError(
            "Text cannot be empty."
        )


    # --------------------------------------------------------
    # Normalize languages
    # --------------------------------------------------------

    source_language = validate_language(
        source_language
    )

    target_language = validate_language(
        target_language
    )


    # --------------------------------------------------------
    # Same language
    # --------------------------------------------------------

    if source_language == target_language:

        print(
            f"Translation skipped: "
            f"{source_language} → "
            f"{target_language}"
        )

        return text.strip()


    # --------------------------------------------------------
    # Translation log
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "                 TRANSLATION"
    )

    print(
        "-" * 70
    )

    print(
        f"Source : {source_language}"
    )

    print(
        f"Target : {target_language}"
    )

    print(
        f"Characters : {len(text):,}"
    )

    active_config = get_runtime_config()
    if active_config is not None and active_config.model:
        print(
            f"Active LLM : "
            f"{active_config.provider or 'default'} / "
            f"{active_config.model}"
        )
    else:
        print(
            "Active LLM : environment default "
            "(core.llm_provider)"
        )


    # --------------------------------------------------------
    # Chain
    # --------------------------------------------------------

    chain = get_translation_chain()


    # --------------------------------------------------------
    # Translate chunks in their original order
    # --------------------------------------------------------

    chunks = _split_text_into_chunks(
        text.strip()
    )
    total_chunks = len(chunks)

    if total_chunks > 1:
        print(
            f"Translation chunks: {total_chunks}"
        )

    translated_chunks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if total_chunks > 1:
            print(
                f"Translating chunk {index}/{total_chunks}"
            )

        result = _invoke_translation_with_retry(
            chain,
            {
                "source_language": source_language,
                "target_language": target_language,
                "text": chunk,
            },
            chunk_index=index,
            total_chunks=total_chunks,
        )

        translated_chunk = (result or "").strip()
        if not translated_chunk:
            raise RuntimeError(
                f"Translation returned empty text for chunk "
                f"{index}/{total_chunks}. "
                f"Source: {source_language}. "
                f"Target: {target_language}."
            )

        translated_chunks.append(translated_chunk)

    translated_text = chr(10).join(
        translated_chunks
    ).strip()

    if not translated_text:
        raise RuntimeError(
            "Translation completed but the "
            "active LLM provider returned "
            "empty text."
        )


    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        "✅ Translation completed."
    )

    print(
        f"Translated characters: "
        f"{len(translated_text):,}"
    )

    print(
        "-" * 70
    )


    return translated_text


# ============================================================
# TRANSCRIBED TEXT → ENGLISH
# ============================================================

def translate_to_english(
    text: str,
    source_language: str,
) -> str:
    """
    Convenience function.

    Used when the application needs English
    regardless of the original language.
    """

    return translate_text(
        text=text,
        source_language=source_language,
        target_language="english",
    )


# ============================================================
# TRANSLATED TEXT → USER LANGUAGE
# ============================================================

def translate_to_language(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Generic pipeline helper.
    """

    return translate_text(
        text=text,
        source_language=source_language,
        target_language=target_language,
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nTranslation module loaded successfully."
    )

    print(
        "Active LLM provider: "
        "core.llm_provider "
        "(active runtime configuration)"
    )

    print(
        "Supported languages:"
    )

    for language in sorted(
        SUPPORTED_LANGUAGES
    ):

        print(
            f"  - {language}"
        )