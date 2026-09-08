"""
core/summarize.py

Mistral-powered summarization engine for Video Agent.

Pipeline:

    Transcript
        ↓
    Text Chunking
        ↓
    Map Summarization
        ↓
    Intermediate Summaries
        ↓
    Reduce
        ↓
    Final Summary

Supports:

    - YouTube videos
    - Meetings
    - Long transcripts
    - Recursive reduction
    - Title generation

LLM:
    Mistral Small
"""

from __future__ import annotations

import math
import os
import re
import time

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

# ---------------------------------------------------------------------
# LLM configuration is owned centrally by core/llm_provider.
#
# This module deliberately reads NO provider env vars (MISTRAL_API_KEY,
# MISTRAL_MODEL, ...).  The active runtime configuration -- Mistral,
# Gemini, OpenAI, NVIDIA, or any OpenAI-compatible endpoint -- is selected
# once in the UI / configure_llm() and applied to every feature here,
# so switching providers at runtime works without restarting anything.
# ---------------------------------------------------------------------


# ============================================================
# SUMMARIZATION CONFIGURATION
# ============================================================

# Transcript chunk size.
#
# This is character-based, not token-based.

SUMMARY_CHUNK_SIZE = int(
    os.getenv(
        "SUMMARY_CHUNK_SIZE",
        "5000",
    )
)


SUMMARY_CHUNK_OVERLAP = int(
    os.getenv(
        "SUMMARY_CHUNK_OVERLAP",
        "300",
    )
)


# Maximum characters allowed when
# combining intermediate summaries.

REDUCE_CHUNK_SIZE = int(
    os.getenv(
        "REDUCE_CHUNK_SIZE",
        "10000",
    )
)


# Maximum reduction rounds.

MAX_REDUCTION_ROUNDS = int(
    os.getenv(
        "MAX_REDUCTION_ROUNDS",
        "5",
    )
)


# ============================================================
# API KEY VALIDATION
# ============================================================

# Deliberately removed: there is no fixed API key anymore.  Missing-key
# and unknown-provider situations produce actionable errors from
# core/llm_provider.validate_configuration() at call time instead of
# crashing the application during import -- an app configured for Gemini,
# OpenAI or NVIDIA must never fail because MISTRAL_API_KEY is unset.


# ============================================================
# SUPPORTED SOURCE TYPES
# ============================================================

SUPPORTED_SOURCE_TYPES = {
    "youtube",
    "meeting",
}


# ============================================================
# LLM
# ============================================================

def get_llm():
    """
    Return the LangChain chat model for the ACTIVE LLM configuration.

    Delegates to the centralized provider layer (core/llm_provider) so the
    summarization pipeline always follows the runtime-selected provider and
    model.  Nothing provider-specific is cached at module level here: the
    provider layer caches clients keyed on a full configuration identity
    fingerprint, so switching from Mistral to NVIDIA (or any provider /
    model) rebuilds the client instead of reusing a stale one.
    """
    from core.llm_provider import get_chat_model

    return get_chat_model(role="Summarizer")


def _active_model_banner() -> str:
    """Human-readable 'provider/model' of the ACTIVE LLM configuration."""
    from core.llm_provider import get_runtime_config

    cfg = get_runtime_config()
    if cfg is not None:
        provider = cfg.provider or "auto"
        model = cfg.model
        # Display-only guard: the stored model value may already carry
        # the provider prefix (e.g. "openrouter/free"), so never repeat
        # the provider name in the banner.
        if model and model.startswith(f"{provider}/"):
            return model
        return f"{provider}/{model}"
    return f"mistral/{os.getenv('MISTRAL_MODEL', 'mistral-small-latest')}"


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Return True only for transient LLM/API failures."""
    text = str(exc).lower()

    status_code = None
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)

    if status_code in {429, 500, 502, 503, 504}:
        return True

    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "too many requests",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "network error",
        "connection error",
    )

    return any(marker in text for marker in retryable_markers)


def _extract_retry_delay(exc: Exception) -> float | None:
    """Extract a positive, finite provider-supplied retry delay in seconds."""
    retry_after = getattr(exc, "retry_after", None)
    candidates = [retry_after]

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            candidates.extend(
                [
                    headers.get("retry-after"),
                    headers.get("Retry-After"),
                ]
            )
        except (AttributeError, TypeError):
            pass

    for value in candidates:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delay) and delay > 0:
            return delay

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
                return delay
        except (TypeError, ValueError):
            pass

    return None


def _invoke_with_retry(
    operation_name: str,
    runnable,
    payload: dict[str, str],
    max_retries: int = 4,
    base_delay: float = 1.0,
):
    """Retry transient LLM/API failures with exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return runnable.invoke(payload)
        except Exception as exc:
            last_error = exc
            if not _is_retryable_llm_error(exc):
                raise

            if attempt == max_retries:
                raise RuntimeError(
                    f"{operation_name} failed after {max_retries} attempts. "
                    f"Reason: {exc}"
                ) from exc

            delay = _extract_retry_delay(exc)
            if delay is None:
                delay = base_delay * (2 ** (attempt - 1))

            print(
                f"\nLLM API temporary error during {operation_name} "
                f"({exc}). Retrying... Attempt {attempt + 1}/{max_retries}"
            )
            print(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)

    if last_error is not None:
        raise RuntimeError(
            f"{operation_name} failed. Reason: {last_error}"
        ) from last_error

    raise RuntimeError(f"{operation_name} failed without a captured exception.")


# ============================================================
# VALIDATE TRANSCRIPT
# ============================================================

def validate_transcript(
    transcript: str,
) -> None:
    """
    Validate transcript before processing.
    """

    if not transcript:

        raise ValueError(
            "Transcript is empty."
        )

    if not transcript.strip():

        raise ValueError(
            "Transcript contains no readable text."
        )


# ============================================================
# VALIDATE SOURCE TYPE
# ============================================================

def validate_source_type(
    source_type: str,
) -> str:
    """
    Normalize and validate source type.
    """

    if not source_type:

        raise ValueError(
            "source_type cannot be empty."
        )

    normalized = (
        source_type
        .strip()
        .lower()
    )

    if normalized not in SUPPORTED_SOURCE_TYPES:

        raise ValueError(
            f"Invalid source_type: "
            f"{source_type}. "
            f"Expected: youtube or meeting."
        )

    return normalized


# ============================================================
# SPLIT TRANSCRIPT
# ============================================================

def split_transcript(
    transcript: str,
) -> list[str]:
    """
    Split a long transcript into manageable chunks.

    Uses semantic separators where possible.
    """

    validate_transcript(
        transcript
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_text(
        transcript
    )

    if not chunks:

        raise RuntimeError(
            "Transcript chunking produced "
            "no chunks."
        )

    return chunks


# ============================================================
# YOUTUBE MAP PROMPT
# ============================================================

YOUTUBE_MAP_SYSTEM_PROMPT = """
You are an expert YouTube content analyst.

You are given one portion of a YouTube video transcript.

Extract and summarize only the information actually
present in this transcript portion.

Focus on:

- Main topics
- Important explanations
- Key concepts
- Technical details
- Important facts
- Numbers and dates
- Examples
- Useful lessons
- Conclusions or observations

Rules:

1. Do not invent information.
2. Do not add your own opinion.
3. Do not make assumptions.
4. Preserve important names.
5. Preserve numbers.
6. Preserve dates.
7. Preserve technical terminology.
8. Remove repetitive statements.
9. Keep the summary concise.
10. Return only useful information.

Write a clean professional summary.
"""


# ============================================================
# MEETING MAP PROMPT
# ============================================================

MEETING_MAP_SYSTEM_PROMPT = """
You are a professional meeting analyst.

You are given one portion of a meeting transcript.

Extract and summarize only the information actually
present in this transcript portion.

Focus on:

- Main discussion points
- Decisions
- Action items
- Responsibilities
- Deadlines
- Problems
- Requirements
- Important business information
- Questions
- Agreements

Rules:

1. Do not invent information.
2. Do not assume responsibilities.
3. Do not create deadlines that were not mentioned.
4. Preserve names.
5. Preserve numbers.
6. Preserve dates.
7. Preserve technical terminology.
8. Remove repetition.
9. Clearly distinguish discussion from decisions.
10. Keep the summary concise.

Write a clean professional summary.
"""


# ============================================================
# BUILD MAP CHAIN
# ============================================================

def build_map_chain(
    source_type: str,
):
    """
    Build the chunk-level summarization chain.
    """

    source_type = validate_source_type(
        source_type
    )

    if source_type == "youtube":

        system_prompt = (
            YOUTUBE_MAP_SYSTEM_PROMPT
        )

    else:

        system_prompt = (
            MEETING_MAP_SYSTEM_PROMPT
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                """
Transcript portion:

--- START ---

{text}

--- END ---
""",
            ),
        ]
    )

    return (
        prompt
        | get_llm()
        | StrOutputParser()
    )


# ============================================================
# SUMMARIZE ONE CHUNK
# ============================================================

def summarize_chunk(
    chunk: str,
    source_type: str = "meeting",
) -> str:
    """
    Summarize one transcript chunk.
    """

    if not chunk or not chunk.strip():

        return ""

    chain = build_map_chain(
        source_type
    )

    try:

        result = _invoke_with_retry(
            operation_name="MAP summarization",
            runnable=chain,
            payload={"text": chunk.strip()},
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to summarize transcript chunk. "
            f"Reason: {exc}"
        ) from exc

    result = (
        result or ""
    ).strip()

    return result


# ============================================================
# REDUCE PROMPTS
# ============================================================

YOUTUBE_REDUCE_SYSTEM_PROMPT = """
You are a senior YouTube content analyst.

Combine the provided partial summaries into a single
accurate summary of the video.

Use this structure:

## Video Summary

A concise overview of the complete video.

## Main Topics

Major subjects discussed.

## Key Insights

Most valuable concepts and explanations.

## Important Facts

Important names, numbers, dates, technical details
and factual statements explicitly present.

## Practical Takeaways

Useful lessons or techniques presented.

## Conclusion

The overall conclusion or message of the video.

Rules:

1. Use only information from the supplied summaries.
2. Do not invent information.
3. Do not add outside knowledge.
4. Remove duplicate information.
5. Preserve important numbers and dates.
6. Preserve technical terminology.
7. Do not confuse examples with facts.
8. Keep the final result professional and readable.
"""


MEETING_REDUCE_SYSTEM_PROMPT = """
You are a senior professional meeting analyst.

Combine the provided partial summaries into one
accurate meeting summary.

Use this structure:

## Meeting Summary

Brief overview.

## Main Discussion Points

Important subjects discussed.

## Key Decisions

Decisions explicitly made.

## Action Items

Tasks explicitly assigned.

## Responsibilities

People or teams responsible when explicitly mentioned.

## Deadlines

Dates or deadlines explicitly mentioned.

## Open Questions

Questions that remain unresolved.

Rules:

1. Use only information from the supplied summaries.
2. Do not invent information.
3. Do not assign responsibility unless explicitly stated.
4. Do not create deadlines.
5. Preserve names, numbers and dates.
6. Remove duplicate information.
7. Clearly distinguish discussion from decisions.
8. Keep the result professional and concise.
"""


# ============================================================
# BUILD REDUCE CHAIN
# ============================================================

def build_reduce_chain(
    source_type: str,
):
    """
    Build final reduction chain.
    """

    source_type = validate_source_type(
        source_type
    )

    if source_type == "youtube":

        system_prompt = (
            YOUTUBE_REDUCE_SYSTEM_PROMPT
        )

    else:

        system_prompt = (
            MEETING_REDUCE_SYSTEM_PROMPT
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                """
Partial summaries:

--- START ---

{text}

--- END ---
""",
            ),
        ]
    )

    return (
        prompt
        | get_llm()
        | StrOutputParser()
    )


# ============================================================
# SPLIT SUMMARIES FOR REDUCTION
# ============================================================

def group_summaries(
    summaries: list[str],
    max_chars: int = REDUCE_CHUNK_SIZE,
    max_items: int = 6,
) -> list[str]:
    """
    Group intermediate summaries into bounded reduction batches.

    Unlike character-splitting the concatenated text, this function
    never cuts an individual summary in half. It also enforces a
    maximum number of summaries per batch so every reduction round
    makes progress toward a single final summary.
    """

    groups: list[str] = []
    current: list[str] = []
    current_chars = 0

    for summary in summaries:
        summary = (summary or "").strip()
        if not summary:
            continue

        summary_chars = len(summary)

        # Start a new group when either limit would be exceeded.
        if current and (
            len(current) >= max_items
            or current_chars + summary_chars > max_chars
        ):
            groups.append("\n\n".join(current))
            current = []
            current_chars = 0

        current.append(summary)
        current_chars += summary_chars

    if current:
        groups.append("\n\n".join(current))

    return groups


def _reduce_group(
    chain,
    group: str,
) -> str:
    """Run one bounded reduction request and validate its output."""

    if not group.strip():
        return ""

    try:
        result = _invoke_with_retry(
            operation_name="REDUCE summarization",
            runnable=chain,
            payload={"text": group},
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed during summary reduction. "
            f"Reason: {exc}"
        ) from exc

    result = (result or "").strip()

    if not result:
        raise RuntimeError(
            "Summary reduction returned empty output."
        )

    return result

def combine_summaries(
    summaries: list[str],
    source_type: str = "meeting",
) -> str:
    """
    Intermediate summaries ko hierarchical reduction ke through
    ek final summary mein combine karta hai.

    Important:
        Reduction summary-count based hai.

    Example:

        7 summaries
            ↓
        4 + 3
            ↓
        2 summaries
            ↓
        1 final summary

    Is design mein reduction non-converging nahi hoti.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not summaries:
        raise ValueError(
            "No summaries were provided."
        )

    source_type = validate_source_type(
        source_type
    )

    # Empty summaries remove karo
    current_summaries = [
        summary.strip()
        for summary in summaries
        if summary and summary.strip()
    ]

    if not current_summaries:
        raise ValueError(
            "All provided summaries are empty."
        )

    # ========================================================
    # REDUCTION CONFIGURATION
    # ========================================================

    # Ek LLM call mein maximum summaries.
    #
    # 7 summaries:
    #
    #   [1,2,3,4]
    #   [5,6,7]
    #
    # Result:
    #   2 summaries
    #
    MAX_SUMMARIES_PER_BATCH = int(
        os.getenv(
            "REDUCE_MAX_SUMMARIES_PER_BATCH",
            "4",
        )
    )

    if MAX_SUMMARIES_PER_BATCH < 2:
        MAX_SUMMARIES_PER_BATCH = 2

    # Maximum safety rounds.
    #
    # Normal case mein iski zaroorat nahi padegi,
    # kyunki har round summary count reduce karega.

    max_rounds = max(
        MAX_REDUCTION_ROUNDS,
        10,
    )

    # ========================================================
    # REDUCE CHAIN
    # ========================================================

    chain = build_reduce_chain(
        source_type
    )

    round_number = 0

    # ========================================================
    # HIERARCHICAL REDUCTION
    # ========================================================

    while len(current_summaries) > 1:

        round_number += 1

        print(
            f"\nReduce round "
            f"{round_number}: "
            f"{len(current_summaries)} summaries"
        )

        if round_number > max_rounds:

            raise RuntimeError(
                "Maximum summary reduction rounds exceeded. "
                f"Remaining summaries: "
                f"{len(current_summaries)}"
            )

        # ----------------------------------------------------
        # Create fixed-size summary batches
        # ----------------------------------------------------

        batches: list[list[str]] = []

        for start in range(
            0,
            len(current_summaries),
            MAX_SUMMARIES_PER_BATCH,
        ):

            batch = current_summaries[
                start:
                start + MAX_SUMMARIES_PER_BATCH
            ]

            batches.append(
                batch
            )

        print(
            f"Reduction batches: "
            f"{len(batches)}"
        )

        # ----------------------------------------------------
        # Reduce each batch
        # ----------------------------------------------------

        next_summaries: list[str] = []

        for index, batch in enumerate(
            batches,
            start=1,
        ):

            print(
                f"Reducing batch "
                f"{index}/{len(batches)} "
                f"({len(batch)} summaries)..."
            )

            # ------------------------------------------------
            # Single summary
            # ------------------------------------------------
            #
            # Agar last batch mein sirf ek summary hai,
            # usko LLM ko unnecessarily send nahi karna.

            if len(batch) == 1:

                next_summaries.append(
                    batch[0]
                )

                continue

            # ------------------------------------------------
            # Combine batch
            # ------------------------------------------------

            batch_text = "\n\n".join(
                f"Summary {i + 1}:\n{summary}"
                for i, summary in enumerate(batch)
            )

            try:

                reduced = _invoke_with_retry(
                    operation_name=f"REDUCE summarization (round {round_number}, batch {index})",
                    runnable=chain,
                    payload={"text": batch_text},
                )

            except Exception as exc:

                raise RuntimeError(
                    "Failed during summary reduction. "
                    f"Round: {round_number}, "
                    f"Batch: {index}. "
                    f"Reason: {exc}"
                ) from exc

            reduced = (
                reduced or ""
            ).strip()

            if not reduced:

                raise RuntimeError(
                    "Summary reduction produced "
                    "empty output. "
                    f"Round: {round_number}, "
                    f"Batch: {index}."
                )

            next_summaries.append(
                reduced
            )

        # ====================================================
        # CONVERGENCE CHECK
        # ====================================================

        previous_count = len(
            current_summaries
        )

        new_count = len(
            next_summaries
        )

        print(
            f"Reduction result: "
            f"{previous_count} → {new_count}"
        )

        # ----------------------------------------------------
        # HARD SAFETY CHECK
        # ----------------------------------------------------

        if new_count >= previous_count:

            raise RuntimeError(
                "Summary reduction did not converge. "
                f"Previous summaries: {previous_count}, "
                f"New summaries: {new_count}."
            )

        # ----------------------------------------------------
        # Move to next round
        # ----------------------------------------------------

        current_summaries = (
            next_summaries
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_summary = (
        current_summaries[0]
        .strip()
    )

    if not final_summary:

        raise RuntimeError(
            "Final summary is empty."
        )

    return final_summary

# ============================================================
# MAIN SUMMARIZATION ENGINE
# ============================================================

def summarize(
    transcript: str,
    source_type: str = "meeting",
) -> str:
    """
    Complete summarization pipeline.

    Flow:

        Transcript
            ↓
        Chunking
            ↓
        Map
            ↓
        Partial summaries
            ↓
        Reduce
            ↓
        Final summary
    """

    validate_transcript(
        transcript
    )

    source_type = validate_source_type(
        source_type
    )

    print("\n" + "=" * 70)

    print(
        "                  SUMMARIZATION"
    )

    print("=" * 70)

    print(
        f"Source type : {source_type}"
    )

    from core.llm_provider import get_runtime_config
    _active_cfg = get_runtime_config()
    _provider = (_active_cfg.provider if _active_cfg else None) or "auto"
    _model = (_active_cfg.model if _active_cfg else None) or "auto"
    print(
        f"Provider    : {_provider}"
    )
    print(
        f"Model       : {_model}"
    )

    print(
        f"Transcript  : "
        f"{len(transcript):,} characters"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    chunks = split_transcript(
        transcript
    )

    print(
        f"Transcript chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # MAP
    # --------------------------------------------------------

    chunk_summaries = []

    print(
        "\nStarting MAP summarization..."
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"Processing chunk "
            f"{index}/{len(chunks)}..."
        )

        summary = summarize_chunk(
            chunk=chunk,
            source_type=source_type,
        )

        if summary:

            chunk_summaries.append(
                summary
            )

    if not chunk_summaries:

        raise RuntimeError(
            "MAP summarization produced "
            "no summaries."
        )

    print(
        f"\nMAP completed."
    )

    print(
        f"Generated summaries: "
        f"{len(chunk_summaries)}"
    )

    # --------------------------------------------------------
    # REDUCE
    # --------------------------------------------------------

    print(
        "\nStarting REDUCE..."
    )

    final_summary = combine_summaries(
        summaries=chunk_summaries,
        source_type=source_type,
    )

    if not final_summary:

        raise RuntimeError(
            "Final summary is empty."
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\n✅ Summarization completed."
    )

    print(
        f"Final summary characters: "
        f"{len(final_summary):,}"
    )

    print("=" * 70)

    return final_summary


# ============================================================
# TITLE GENERATION
# ============================================================

TITLE_SYSTEM_PROMPT = """
You are a professional content analyst.

Create a short, accurate title based only on
the supplied transcript.

Rules:

1. Maximum 8 words.
2. Clear.
3. Professional.
4. Meaningful.
5. Describe the actual subject.
6. Do not invent topics.
7. Do not use quotation marks.
8. Do not use emojis.
9. Return ONLY the title.
10. Do not provide explanation.
"""


def generate_title(
    transcript: str,
    source_type: str = "meeting",
) -> str:
    """
    Generate a short title from transcript.

    Only a representative portion is sent to the LLM
    because sending an entire multi-hour transcript
    just for title generation is unnecessary.
    """

    validate_transcript(
        transcript
    )

    source_type = validate_source_type(
        source_type
    )

    # --------------------------------------------------------
    # Select representative transcript
    # --------------------------------------------------------

    # First 4000 characters usually contain
    # enough context for a title.

    title_text = transcript[:4000]

    if source_type == "youtube":

        context = """
This is a YouTube video.
Create a title describing the video's
main subject.
"""

    else:

        context = """
This is a meeting recording.
Create a title describing the
main subject discussed.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
{TITLE_SYSTEM_PROMPT}

{context}
""",
            ),
            (
                "human",
                """
Transcript:

--- START ---

{text}

--- END ---
""",
            ),
        ]
    )

    chain = (
        prompt
        | get_llm()
        | StrOutputParser()
    )

    try:

        title = chain.invoke(
            {
                "text": title_text
            }
        )

    except Exception as exc:

        raise RuntimeError(
            f"Title generation failed: "
            f"{exc}"
        ) from exc

    title = (
        title or ""
    ).strip()

    if not title:

        raise RuntimeError(
            "Title generation returned empty text."
        )

    return title


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nSummarization module loaded successfully."
    )

    print(
        f"Active model: {_active_model_banner()}"
    )

    print(
        f"Summary chunk size: "
        f"{SUMMARY_CHUNK_SIZE}"
    )

    print(
        f"Summary overlap: "
        f"{SUMMARY_CHUNK_OVERLAP}"
    )

    print(
        f"Reduce chunk size: "
        f"{REDUCE_CHUNK_SIZE}"
    )

    print(
        "\nNo transcript was processed."
    )