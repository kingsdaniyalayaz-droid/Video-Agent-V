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

import re
import time

from dotenv import load_dotenv

from core.llm_provider import get_chat_model, get_runtime_config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

TRANSLATION_CHUNK_SIZE = 5000

# Maximum number of retries after the initial request attempt.
TRANSLATION_MAX_RETRIES = 3
TRANSLATION_RETRY_BASE_DELAY = 1.0
TRANSLATION_RETRY_MAX_DELAY = 8.0


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
    max_chars: int = TRANSLATION_CHUNK_SIZE,
) -> list[str]:
    """
    Split text into ordered chunks without dropping or duplicating content.

    Natural boundaries are preferred in this order: paragraph, newline,
    sentence, whitespace, and finally a hard character boundary.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if not text:
        return []

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_at = window.rfind(chr(10) * 2)
        if split_at >= 0:
            split_at += 2
        else:
            split_at = window.rfind(chr(10))
            if split_at >= 0:
                split_at += 1

        if split_at <= 0:
            sentence_matches = list(
                re.finditer(
                    r'''[.!?؟。！？](?:["'’”)]*)[ \t\r\n]+''',
                    window,
                )
            )
            if sentence_matches:
                split_at = sentence_matches[-1].end()

        if split_at <= 0:
            whitespace_matches = list(re.finditer(r'''[ \t\r\n]+''', window))
            if whitespace_matches:
                split_at = whitespace_matches[-1].end()

        if split_at <= 0:
            split_at = max_chars

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    if remaining:
        chunks.append(remaining)
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


def _invoke_translation_with_retry(
    chain,
    payload: dict[str, str],
    *,
    chunk_index: int,
    total_chunks: int,
):
    """Invoke one chunk with a bounded retry and exponential backoff."""
    attempt = 1
    total_attempts = TRANSLATION_MAX_RETRIES + 1

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

            delay = min(
                TRANSLATION_RETRY_BASE_DELAY * (2 ** retries_used),
                TRANSLATION_RETRY_MAX_DELAY,
            )
            print(
                f"Retrying translation chunk "
                f"{chunk_index}/{total_chunks}"
            )
            print(
                f"Attempt {attempt + 1}/{total_attempts}"
            )
            print(f"Reason: {exc}")
            print(f"Waiting {delay:.1f} seconds")
            time.sleep(delay)
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
        text.strip(),
        TRANSLATION_CHUNK_SIZE,
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