from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from utils.audio_processor import process_input, cleanup_chunks
from core.transcriber import transcribe_all
from core.translator import translate_text
from core.summarize import summarize, generate_title
from core.rag_engine import (
    build_rag_chain,
    ask_question,
)
from core.database import (
    get_video,
    create_video,
    mark_video_failed,
    recover_stale_processing_videos,
    save_completed_video,
    update_video,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SOURCE_TYPE = os.getenv(
    "DEFAULT_SOURCE_TYPE",
    "youtube",
).strip().lower()

DEFAULT_LANGUAGE = os.getenv(
    "WHISPER_LANGUAGE",
    "en",
).strip().lower()

DEFAULT_CHUNK_MINUTES = int(
    os.getenv(
        "AUDIO_CHUNK_MINUTES",
        "10",
    )
)

DEFAULT_RAG_TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "4",
    )
)


# ============================================================
# PIPELINE RESULT
# ============================================================

@dataclass
class PipelineResult:
    """
    Complete pipeline ka structured result.

    RAG chain ko result ke andar rakha gaya hai taake
    Streamlit current processed video par questions pooch sake.
    """

    source: str
    source_type: str
    language: str

    audio_chunks: list[str] = field(
        default_factory=list
    )

    transcript: str = ""

    translated_transcript: str = ""

    title: str = ""

    summary: str = ""

    actions: list[str] = field(
        default_factory=list
    )

    decisions: list[str] = field(
        default_factory=list
    )

    questions: list[str] = field(
        default_factory=list
    )

    rag_chain: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# DATABASE / DUPLICATE PROCESSING HELPERS
# ============================================================

def get_youtube_video_id(source: str) -> str | None:
    """Return the canonical YouTube video ID, or None for local files."""
    try:
        parsed = urlparse(source.strip())
        hostname = (parsed.hostname or "").lower().removeprefix("www.")

        if hostname == "youtu.be":
            return parsed.path.strip("/").split("/")[0] or None

        if hostname == "youtube.com":
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]
            if parsed.path.startswith("/shorts/"):
                return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0] or None
    except Exception:
        pass

    return None


def generate_meeting_id(source: str) -> str:
    """Generate a deterministic ID from local meeting file content."""
    path = Path(source)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        digest.update(source.strip().encode("utf-8"))
    return f"meeting_{digest.hexdigest()[:16]}"


def build_meeting_processing_id(upload_id: str, language: str) -> str:
    """Return the language-aware processing identity for uploaded meetings.

    Upload identity stays content-based (``meeting_<hash>``). Processing,
    caching, SQLite, and vector-store identity become language-aware via the
    normalized Whisper language suffix.
    """
    normalized_language = validate_language(language)
    base_id = str(upload_id or "").strip()
    if not base_id:
        raise ValueError("upload_id cannot be empty.")

    match = re.fullmatch(r"(meeting_[0-9a-f]{16})_(en|ur|hi)", base_id)
    if match:
        base_id = match.group(1)

    return f"{base_id}_{normalized_language}"


def build_cached_result(
    record: dict[str, Any],
    source: str,
    source_type: str,
    language: str,
) -> PipelineResult:
    """Convert a completed DB record into the normal pipeline result shape."""
    transcript = (record.get("transcript") or "").strip()
    translated = (record.get("translated_transcript") or transcript).strip()
    actions = _normalize_analysis_items(record.get("actions"))
    decisions = _normalize_analysis_items(record.get("decisions"))
    questions = _normalize_analysis_items(record.get("questions"))
    return PipelineResult(
        source=source,
        source_type=source_type,
        language=language,
        transcript=transcript,
        translated_transcript=translated,
        title=(record.get("title") or "Untitled Meeting").strip(),
        summary=(record.get("summary") or "").strip(),
        actions=actions,
        decisions=decisions,
        questions=questions,
        metadata={
            "cached": True,
            "video_id": record.get("video_id"),
            "duration": record.get("duration"),
            "audio_chunks": record.get("chunk_count") or 0,
            "transcript_characters": len(transcript),
            "actions": actions,
            "decisions": decisions,
            "questions": questions,
            "analysis_status": record.get("analysis_status"),
            "analysis_error": record.get("analysis_error"),
        },
    )


def _analysis_status(record: dict[str, Any]) -> str:
    """Normalize the persisted analysis status.

    Records created before analysis_status existed carry no value; they are
    treated as 'pending' so their analysis is computed (and then persisted)
    exactly once - never silently assumed complete from empty lists.
    """
    status = str(record.get("analysis_status") or "").strip().lower()
    return status if status in {"pending", "completed", "failed"} else "pending"


def _analysis_value_empty(value: Any) -> bool:
    """Return True when a stored analysis field carries no usable value."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True
        return text.lower() in _EMPTY_ANALYSIS_RESPONSES
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def _analysis_repair_fields(record: dict[str, Any]) -> list[str]:
    """Fields that genuinely require fresh extraction for this record.

    - analysis_status == 'completed' -> nothing to repair: empty lists are a
      valid completed analysis and must NOT be re-extracted on every reopen.
    - 'pending' / 'failed' / legacy records -> fields that hold no usable
      value. Fields that already hold a value are never overwritten.
    """
    if _analysis_status(record) == "completed":
        return []
    return [
        field_name
        for field_name in ("actions", "decisions", "questions")
        if _analysis_value_empty(record.get(field_name))
    ]


# ============================================================
# VALIDATION
# ============================================================


_EMPTY_ANALYSIS_RESPONSES = {
    "no action items found.",
    "no decisions found.",
    "no questions found.",
    "no meaningful questions found.",
}


def _normalize_analysis_items(value: Any) -> list[str]:
    """Normalize extractor output into clean lists for UI and persistence."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in _EMPTY_ANALYSIS_RESPONSES:
            return []
        raw_items = []
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("##"):
                continue
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", cleaned).strip()
            if cleaned:
                raw_items.append(cleaned)
        if not raw_items:
            raw_items = [text]

    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = str(item).strip()
        if not cleaned or cleaned.lower() in _EMPTY_ANALYSIS_RESPONSES:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)
    return items


class AnalysisExtractionError(RuntimeError):
    """Raised when structured analysis genuinely fails.

    Distinct from an empty result: [] means the extractor RAN and found
    nothing, while AnalysisExtractionError means extraction never completed.
    Callers persist analysis_status='failed' + the real error; they never
    silently record an empty list as success.
    """


_FIELD_TO_EXTRACTOR = (
    ("actions", "extract_action_items"),
    ("decisions", "extract_decisions"),
    ("questions", "extract_questions"),
)


def _extract_analysis_fields(
    transcript: str,
    fields: list[str],
) -> dict[str, list[str]]:
    """Extract only the requested analysis fields.

    An empty list is a SUCCESSFUL extraction that found nothing. Failures are
    NOT converted into empty lists: they raise AnalysisExtractionError so the
    caller can persist analysis_status='failed' + a useful analysis_error.
    """
    try:
        from core import extractor as extractor_module
        extractor_functions = {
            "actions": extractor_module.extract_action_items,
            "decisions": extractor_module.extract_decisions,
            "questions": extractor_module.extract_questions,
        }
    except Exception as exc:
        raise AnalysisExtractionError(
            f"extractor module unavailable: {exc}"
        ) from exc

    wanted = set(fields)
    extracted: dict[str, list[str]] = {}

    for field_name, _function_name in _FIELD_TO_EXTRACTOR:
        if field_name not in wanted:
            continue

        extractor_fn = extractor_functions[field_name]
        if not callable(extractor_fn):
            raise AnalysisExtractionError(
                f"extractor for {field_name} is not callable; "
                f"cannot compute {field_name}."
            )

        try:
            extracted[field_name] = _normalize_analysis_items(
                extractor_fn(transcript)
            )
        except Exception as exc:
            raise AnalysisExtractionError(
                f"{field_name} extraction failed: {exc}"
            ) from exc

    return extracted


def _extract_structured_analysis(
    transcript: str,
) -> tuple[list[str], list[str], list[str]]:
    """Extract all three analysis fields; propagates AnalysisExtractionError."""
    extracted = _extract_analysis_fields(
        transcript,
        [field_name for field_name, _ in _FIELD_TO_EXTRACTOR],
    )
    return (
        extracted["actions"],
        extracted["decisions"],
        extracted["questions"],
    )


def _validate_positive_int(value: Any, *, name: str) -> int:
    """Validate a strictly positive integer parameter.

    ``bool`` is rejected because it is an int subclass; ``None``, non-int
    values, and values <= 0 raise a clear ``ValueError`` (never ``TypeError``).
    Returns the validated integer.  Never coerces strings/floats.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer. "
            f"Got {type(value).__name__}."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0."
        )

    return value


def validate_source(
    source: str,
) -> str:
    """Input URL ya local file validate karta hai."""

    if source is None:
        raise ValueError("source cannot be None.")

    if not isinstance(source, str):
        raise ValueError(
            "source must be a string. "
            f"Got {type(source).__name__}."
        )

    source = source.strip()

    if not source:
        raise ValueError(
            "Input source cannot be empty."
        )

    return source


def validate_source_type(
    source_type: str,
) -> str:
    """Source type normalize aur validate karta hai."""

    if source_type is None:
        raise ValueError("source_type cannot be None.")

    if not isinstance(source_type, str):
        raise ValueError(
            "source_type must be a string. "
            f"Got {type(source_type).__name__}."
        )

    source_type = (
        source_type
        .strip()
        .lower()
    )

    if source_type not in {
        "youtube",
        "meeting",
    }:
        raise ValueError(
            "source_type must be "
            "'youtube' or 'meeting'."
        )

    return source_type


def validate_language(language: str) -> str:
    """
    Whisper language ko normalize aur validate karta hai.

    Accepts both language codes and readable names:

        en / english
        ur / urdu
        hi / hindi

    Returns:
        Whisper-compatible language code.
    """

    if language is None:
        raise ValueError("language cannot be None.")

    if not isinstance(language, str):
        raise ValueError(
            "language must be a string. "
            f"Got {type(language).__name__}."
        )

    if not language:
        raise ValueError(
            "language cannot be empty."
        )

    normalized = (
        language
        .strip()
        .lower()
    )

    language_map = {
        "en": "en",
        "english": "en",

        "ur": "ur",
        "urdu": "ur",

        "hi": "hi",
        "hindi": "hi",
    }

    if normalized not in language_map:

        raise ValueError(
            f"Unsupported language: {language}. "
            "Supported languages: "
            "en/english, ur/urdu, hi/hindi."
        )

    return language_map[normalized]


def normalize_translation_language(
    language: Optional[str],
) -> Optional[str]:
    """
    Translation language ko normalize karta hai.

    Supported:
        english
        hindi
        urdu
    """

    if language is None:
        return None

    if not isinstance(language, str):
        raise ValueError(
            "language must be a string or None. "
            f"Got {type(language).__name__}."
        )

    language = language.strip()

    if not language:
        return None

    language = language.lower()

    aliases = {
        "en": "english",
        "english": "english",

        "hi": "hindi",
        "hindi": "hindi",

        "ur": "urdu",
        "urdu": "urdu",
    }

    if language not in aliases:
        raise ValueError(
            f"Unsupported target language: {language}. "
            "Use english, hindi or urdu."
        )

    return aliases[language]


# ============================================================
# LANGUAGE HELPERS
# ============================================================

def language_name(
    language_code: str,
) -> str:
    """Whisper language code ko readable name mein convert karta hai."""

    mapping = {
        "en": "english",
        "ur": "urdu",
        "hi": "hindi",
    }

    return mapping.get(
        language_code,
        language_code,
    )


def emit_progress(
    progress_callback: Optional[Callable[[dict[str, Any]], None]],
    stage: str,
    status: str,
    **data: Any,
) -> None:
    """Emit a pipeline event only when a callback has been supplied."""
    if not progress_callback:
        return

    event = {"stage": stage, "status": status}
    event.update(data)
    try:
        progress_callback(event)
    except Exception:
        pass


# ============================================================
# CACHED VIDEO LOADER (single source of truth)
# ============================================================

def load_cached_video(
    video_id: str,
    source: str,
    source_type: str,
    language: str,
) -> PipelineResult | None:
    """Load a completed video record, repairing missing analysis first.

    This is the ONE centralized cached-video loader:

        Database
          -> repair missing analysis only when required
          -> return authoritative cached result

    run_pipeline() and app.py (cached YouTube submissions and Video Library
    reopen) both route through this function so no UI layer can construct a
    second cached result or bypass the missing-analysis repair. Returns None
    when the record is absent or not completed.
    """
    cached = get_video(video_id)
    if not cached or str(cached.get("status") or "") != "completed":
        return None

    print()
    print("=" * 70)
    print("              CACHED VIDEO FOUND")
    print("=" * 70)
    print(f"Video ID       : {video_id}")
    print("Status         : completed")
    print("Action         : Full processing skipped")
    print("Database       : Existing transcript/summary reused")

    cached_transcript = str(
        cached.get("transcript") or ""
    ).strip()
    status = _analysis_status(cached)

    if cached_transcript and status != "completed":
        repair_fields = _analysis_repair_fields(cached)
        if status == "failed" and not repair_fields:
            # A failed run stores nothing trustworthy; re-extract everything.
            repair_fields = ["actions", "decisions", "questions"]

        if repair_fields:
            print("Analysis       : " + status + " -> repairing")
            for field_name in repair_fields:
                print(f"Extracting {field_name}...")
            try:
                extracted = _extract_analysis_fields(
                    cached_transcript,
                    repair_fields,
                )
                repaired = {
                    field_name: extracted[field_name]
                    for field_name in repair_fields
                }
                print("Analysis items :")
                for field_name in repair_fields:
                    print(f"{field_name}={len(repaired[field_name])}")

                # Persist repaired fields plus the terminal success state.
                # Existing valid analysis values are never overwritten.
                repaired["analysis_status"] = "completed"
                repaired["analysis_error"] = None
                cached = update_video(video_id, **repaired)
            except AnalysisExtractionError as exc:
                print(f"Analysis       : repair failed - {exc}")
                cached = update_video(
                    video_id,
                    analysis_status="failed",
                    analysis_error=str(exc),
                )
        elif status == "pending":
            # Nothing empty left to repair: previous run completed the fields.
            cached = update_video(
                video_id,
                analysis_status="completed",
                analysis_error=None,
            )
    elif cached_transcript:
        print("Analysis       : cached (completed)")
    else:
        print("Analysis       : unavailable (no transcript)")

    print("=" * 70)

    return build_cached_result(
        cached, source, source_type, language
    )


# ============================================================
# CACHED RESULT — TRANSLATION-CONFIG COMPATIBILITY
# ============================================================
# A completed cached record is reusable ONLY when its stored processing
# configuration matches the CURRENT request. Matching video_id alone is
# NOT sufficient: the stored translation state (enabled/disabled + target)
# and the stored source language must also satisfy the requested
# configuration. All cache-compatibility decisions live here;
# load_cached_video() keeps working as the single source of truth for
# records that ARE fully compatible.

_TRANSLATION_TARGET_UNKNOWN = "<unknown>"


def _cached_source_language_compatible(
    record: dict[str, Any],
    language: str,
) -> bool:
    """Return True when the cached record's stored source language matches
    the requested (already normalized) language.

    Legacy records that predate language persistence carry no stored value
    and are accepted best-effort so old caches keep working. A stored
    language that CONFLICTS with the request is never silently accepted.
    """
    stored = record.get("language")
    if stored is None or not str(stored).strip():
        return True
    try:
        return validate_language(str(stored)) == language
    except ValueError:
        return False


def _cached_translation_state(
    record: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return (translation_enabled, target_language) of the cached record.

    - (False, None)       -> no translation stored
    - (True, "<target>")  -> translated to a KNOWN target
    - (True, "<unknown>") -> translated, but the target was not persisted

    The stored 'transcript' is ALWAYS the original (untranslated) source
    transcript, so an unknown target is safe: a fresh requested translation
    is regenerated from the stored transcript, never from the cached
    translated text.
    """
    transcript = str(record.get("transcript") or "").strip()
    translated = str(record.get("translated_transcript") or "").strip()
    has_translation = bool(translated and translated != transcript)

    stored_target = record.get("target_language")
    if stored_target is not None and str(stored_target).strip():
        try:
            target = normalize_translation_language(str(stored_target))
        except ValueError:
            target = None
        if target is None:
            target = _TRANSLATION_TARGET_UNKNOWN
    elif has_translation:
        target = _TRANSLATION_TARGET_UNKNOWN
    else:
        target = None

    enabled = record.get("translation_enabled")
    if isinstance(enabled, bool):
        pass
    elif enabled in (1, "1", "true", "True", "TRUE", "yes"):
        enabled = True
    elif enabled in (0, "0", "false", "False", "FALSE", "no", ""):
        enabled = False
    else:
        enabled = has_translation

    return bool(enabled), target


def _cached_translation_compatible(
    record: dict[str, Any],
    translate: bool,
    target_language: Optional[str],
) -> bool:
    """Return True when the cached result satisfies the requested
    translation configuration.

    translate=True demands a stored translation to EXACTLY the requested
    target; translate=False demands NO stored translation. An unknown
    cached target can never satisfy a concrete requested target, so the
    translation is regenerated from the stored transcript.
    """
    enabled, target = _cached_translation_state(record)
    if not translate:
        return not enabled
    return enabled and target == target_language


def _persist_cache_translation(
    video_id: str,
    translated_transcript: str,
    target_language: Optional[str],
) -> None:
    """Persist a freshly produced cached translation so identical future
    requests hit the full cache. Schema-optional translation-config
    columns degrade gracefully to persisting just the translated text."""
    kwargs: dict[str, Any] = {
        "translated_transcript": translated_transcript,
        "translation_enabled": True,
    }
    if target_language:
        kwargs["target_language"] = target_language
    try:
        update_video(video_id, **kwargs)
    except Exception:
        try:
            update_video(
                video_id,
                translated_transcript=translated_transcript,
            )
        except Exception as persist_error:
            print(
                "Warning: could not persist cached translation: "
                f"{persist_error}"
            )


def _reuse_cached_transcript(
    record: dict[str, Any],
    video_id: str,
    source: str,
    source_type: str,
    language: str,
    translate: bool,
    target_language: Optional[str],
    progress_callback: Optional[Callable[[dict], None]],
) -> PipelineResult | None:
    """Rebuild a completed result from the cached transcript, applying the
    REQUESTED translation configuration without re-downloading or
    re-transcribing the video.

    - translate=False -> the cached transcript is returned as the
      processable transcript; a previously stored translation is never
      presented as the requested translation.
    - translate=True  -> the cached transcript is translated to the
      requested target and the fresh translation is persisted so identical
      future requests become full cache hits.
    """
    cached_transcript = str(record.get("transcript") or "").strip()
    if not cached_transcript:
        return None

    print("-" * 70)
    print("CACHED VIDEO FOUND (configuration differs)")
    print("Status         : completed")
    print("Action         : cached transcript reused")
    print("Download       : skipped")
    print("Transcription  : skipped")

    result = build_cached_result(
        record, source, source_type, language
    )

    if not translate:
        print("Translation    : disabled (requested)")
        print("Stored translation not returned for a disabled request")
        emit_progress(
            progress_callback,
            "translation",
            "skipped",
            message="Translation skipped (cached transcript reused)",
        )
        result.translated_transcript = cached_transcript
    else:
        # target_language is guaranteed non-empty here: run_pipeline()
        # validates that translate=True requires a target language before
        # any cache lookup happens.
        source_language = language_name(language)
        if source_language == target_language:
            print("Translation    : source == target, skipped")
            emit_progress(
                progress_callback,
                "translation",
                "skipped",
                message="Same source and target language",
            )
            result.translated_transcript = cached_transcript
        else:
            print(
                f"Translation    : {source_language} -> {target_language}"
            )
            emit_progress(
                progress_callback,
                "translation",
                "running",
                message="Translating cached transcript",
            )
            try:
                translated_text = translate_text(
                    text=cached_transcript,
                    source_language=source_language,
                    target_language=target_language,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Translation failed: {exc}"
                ) from exc
            if not translated_text or not translated_text.strip():
                raise RuntimeError("Translation returned empty text.")
            result.translated_transcript = translated_text.strip()
            print("Translation    : completed")
            emit_progress(
                progress_callback,
                "translation",
                "completed",
                message="Translation completed (cached transcript)",
            )
            _persist_cache_translation(
                video_id,
                result.translated_transcript,
                target_language,
            )

    result.metadata["cached"] = True
    result.metadata["translation_enabled"] = translate
    result.metadata["target_language"] = (
        target_language if translate else None
    )
    print("-" * 70)
    return result


def _load_cached_video_for_request(
    video_id: str,
    source: str,
    source_type: str,
    language: str,
    translate: bool,
    target_language: Optional[str],
    progress_callback: Optional[Callable[[dict], None]],
) -> PipelineResult | None:
    """Load a completed cached record ONLY when it satisfies the CURRENT
    request configuration.

    Decision ladder:
        1. record absent / not completed      -> None (full pipeline)
        2. stored source language conflicts   -> None (full pipeline)
        3. translation config matches         -> full cached result
        4. config differs, transcript usable  -> result rebuilt from the
           cached transcript with the requested translation applied
        5. nothing reusable                   -> None (full pipeline)
    """
    record = get_video(video_id)
    if not record or str(record.get("status") or "") != "completed":
        return None

    if not _cached_source_language_compatible(record, language):
        print(
            f"Cached language mismatch: "
            f"stored={record.get('language')!r}, "
            f"requested={language!r}; full pipeline will run."
        )
        return None

    if _cached_translation_compatible(record, translate, target_language):
        # Fully compatible: the existing loader remains the single source
        # of truth for the untouched cached-result path.
        return load_cached_video(
            video_id,
            source=source,
            source_type=source_type,
            language=language,
        )

    # Translation configuration differs: reuse the cached transcript only.
    return _reuse_cached_transcript(
        record,
        video_id=video_id,
        source=source,
        source_type=source_type,
        language=language,
        translate=translate,
        target_language=target_language,
        progress_callback=progress_callback,
    )


# ============================================================
# PIPELINE
# ============================================================

# ============================================================
# PIPELINE STAGES
# ============================================================

def _execute_processing_stages(
    source: str,
    source_type: str,
    language: str,
    target_language: Optional[str],
    translate: bool,
    chunk_minutes: int,
    rag_top_k: int,
    video_id: str | None,
    progress_callback: Optional[Callable[[dict], None]],
    temporary_chunks: list[str],
) -> PipelineResult:
    """Execute the audio..persistence stages of the pipeline.

    Any failure raises. run_pipeline() persists status='failed' and a
    useful error_message, then re-raises, so a registered record never
    stays stuck in 'processing'.
    """
    # ========================================================
    # RESULT OBJECT
    # ========================================================

    result = PipelineResult(
        source=source,
        source_type=source_type,
        language=language,
    )

    # ========================================================
    # STEP 1 — AUDIO PROCESSING
    # ========================================================

    print()
    print("[1/5] Processing input...")
    emit_progress(
        progress_callback,
        "input",
        "running",
        message="Preparing input and validating file",
    )

    try:

        audio_chunks = process_input(
            source=source,
            chunk_minutes=chunk_minutes,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Audio processing failed: {exc}"
        ) from exc

    if not audio_chunks:

        raise RuntimeError(
            "Audio processing completed but "
            "no audio chunks were generated."
        )

    result.audio_chunks = audio_chunks
    temporary_chunks.extend(audio_chunks)

    print(
        f"Audio chunks created: "
        f"{len(audio_chunks)}"
    )

    emit_progress(
        progress_callback,
        "input",
        "completed",
        current=1,
        total=1,
        message="Input prepared",
    )
    emit_progress(
        progress_callback,
        "audio",
        "running",
        message="Converting audio and chunking media",
    )
    emit_progress(
        progress_callback,
        "audio",
        "completed",
        current=len(audio_chunks),
        total=len(audio_chunks),
        message="Audio converted and chunked",
    )

    # ========================================================
    # STEP 2 — WHISPER TRANSCRIPTION
    # ========================================================

    print()
    print("[2/5] Transcribing audio with Whisper...")
    emit_progress(
        progress_callback,
        "transcription",
        "running",
        message="Transcribing audio chunks",
    )

    try:

        transcript = transcribe_all(
            chunks=audio_chunks,
            translate=False,
            language=language,
            progress_callback=progress_callback,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Transcription failed: {exc}"
        ) from exc

    if not transcript or not transcript.strip():

        raise RuntimeError(
            "Whisper returned an empty transcript."
        )

    transcript = transcript.strip()

    result.transcript = transcript
    emit_progress(
        progress_callback,
        "transcription",
        "completed",
        current=len(audio_chunks),
        total=len(audio_chunks),
        message="Transcription completed",
    )

    print(
        f"Transcript characters: "
        f"{len(transcript):,}"
    )

    # ========================================================
    # STEP 3 — OPTIONAL TRANSLATION
    # ========================================================

    print()
    print("[3/5] Translation...")

    processed_transcript = transcript

    if translate:
        emit_progress(
            progress_callback,
            "translation",
            "running",
            message="Translating transcript",
        )

        source_language = language_name(
            language
        )

        if (
            source_language
            == target_language
        ):

            print(
                "Source and target languages are "
                "the same. Translation skipped."
            )

        else:

            print(
                f"Translating "
                f"{source_language} → "
                f"{target_language}..."
            )

            try:

                processed_transcript = (
                    translate_text(
                        text=transcript,
                        source_language=source_language,
                        target_language=target_language,
                    )
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Translation failed: {exc}"
                ) from exc

            if not processed_transcript:
                raise RuntimeError(
                    "Translation returned empty text."
                )

            processed_transcript = (
                processed_transcript.strip()
            )

            print(
                "Translation completed."
            )
            emit_progress(
                progress_callback,
                "translation",
                "completed",
                message="Translation completed",
            )

    else:

        print(
            "Translation disabled."
        )
        emit_progress(
            progress_callback,
            "translation",
            "skipped",
            message="Translation skipped",
        )

    result.translated_transcript = (
        processed_transcript
    )

    # ========================================================
    # STEP 4 — TITLE + SUMMARY
    # ========================================================

    print()
    print("[4/5] Generating title and summary...")
    emit_progress(
        progress_callback,
        "title",
        "running",
        message="Generating title",
    )
    emit_progress(
        progress_callback,
        "summary",
        "running",
        message="Generating summary",
    )

    try:

        title = generate_title(
            transcript=processed_transcript,
            source_type=source_type,
        )

        emit_progress(
            progress_callback,
            "title",
            "completed",
            message="Title generated",
        )

        summary = summarize(
            transcript=processed_transcript,
            source_type=source_type,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Summarization failed: {exc}"
        ) from exc

    if not title or not title.strip():
        raise RuntimeError(
            "Title generation returned empty output."
        )

    if not summary or not summary.strip():
        raise RuntimeError(
            "Summarization returned empty output."
        )

    result.title = title.strip()

    result.summary = summary.strip()

    print(
        "Extracting action items, decisions, and questions..."
    )

    # Persist 'pending' BEFORE extraction so a crash or a genuine extraction
    # failure is never misread as a completed analysis. Legacy records that
    # predate analysis_status default to the same pending behavior.
    if video_id:
        update_video(
            video_id,
            analysis_status="pending",
            analysis_error=None,
        )

    analysis_status = "completed"
    analysis_error = None
    try:
        (
            result.actions,
            result.decisions,
            result.questions,
        ) = _extract_structured_analysis(
            processed_transcript
        )
    except AnalysisExtractionError as exc:
        analysis_status = "failed"
        analysis_error = str(exc)
        result.actions = []
        result.decisions = []
        result.questions = []
        print(f"Analysis       : FAILED - {exc}")
        if video_id:
            update_video(
                video_id,
                analysis_status="failed",
                analysis_error=analysis_error,
            )
    emit_progress(
        progress_callback,
        "summary",
        "completed",
        message="Summary generated",
    )

    print(
        f"Title: {result.title}"
    )

    print(
        f"Summary characters: "
        f"{len(result.summary):,}"
    )

    print(
        f"Analysis items   : "
        f"actions={len(result.actions)}, "
        f"decisions={len(result.decisions)}, "
        f"questions={len(result.questions)}"
    )

    # ========================================================
    # STEP 5 — VECTOR STORE + RAG
    # ========================================================

    print()
    print(
        "[5/5] Building vector database and RAG..."
    )
    emit_progress(
        progress_callback,
        "rag",
        "running",
        message="Building vector database",
    )

    try:

        # IMPORTANT:
        #
        # build_rag_chain() internally:
        #
        # Transcript
        #     ↓
        # ChromaDB
        #     ↓
        # Retriever
        #     ↓
        # Mistral
        #     ↓
        # RAG Chain
        #
        # Isliye yahan separately
        # build_vector_store() call nahi karna.

        rag_chain = build_rag_chain(
            transcript=processed_transcript,
            source_type=source_type,
            top_k=rag_top_k,
            video_id=video_id,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Vector store / RAG initialization failed: "
            f"{exc}"
        ) from exc

    result.rag_chain = rag_chain
    emit_progress(
        progress_callback,
        "rag",
        "completed",
        message="Vector database and RAG ready",
    )
    emit_progress(
        progress_callback,
        "finalizing",
        "running",
        message="Finalizing results",
    )

    # ========================================================
    # METADATA
    # ========================================================

    result.metadata = {
        "video_id": video_id,
        "source_type": source_type,

        "language": language,

        "target_language": target_language,

        "translation_enabled": translate,

        "audio_chunks": len(
            audio_chunks
        ),

        "transcript_characters": len(
            transcript
        ),

        "processed_transcript_characters": len(
            processed_transcript
        ),

        "summary_characters": len(
            result.summary
        ),

        "actions": list(result.actions),

        "decisions": list(result.decisions),

        "questions": list(result.questions),

        "analysis_status": analysis_status,

        "analysis_error": analysis_error,

        "rag_top_k": rag_top_k,
    }

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("              PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Title          : {result.title}"
    )

    print(
        f"Audio chunks   : "
        f"{len(result.audio_chunks)}"
    )

    print(
        f"Transcript     : "
        f"{len(result.transcript):,} chars"
    )

    print(
        f"Summary        : "
        f"{len(result.summary):,} chars"
    )

    print(
        "Vector store   : READY"
    )

    print(
        "RAG assistant  : READY"
    )

    # Persist successful processing so the same YouTube video is not
    # downloaded/transcribed/summarized again in the future.
    #
    # The stored translation configuration (enabled/disabled + target) is
    # part of the persistence contract: the cache-compatibility logic reads
    # translation_enabled / target_language from the stored record when a
    # later request arrives. These values always mirror the ACTUAL request
    # configuration (`translate` / normalized `target_language`) - never a
    # hardcoded language and never an inference from the presence of
    # translated text.
    if video_id:
        save_completed_video(
            video_id,
            title=result.title,
            language=language,
            transcript=result.transcript,
            translated_transcript=result.translated_transcript,
            summary=result.summary,
            actions=result.actions,
            decisions=result.decisions,
            questions=result.questions,
            duration=result.metadata.get("duration"),
            chunk_count=len(result.audio_chunks),
            analysis_status=analysis_status,
            analysis_error=analysis_error,
            translation_enabled=translate,
            target_language=target_language if translate else None,
        )

    print("=" * 70)

    return result


def run_pipeline(
    source: str,
    source_type: str = DEFAULT_SOURCE_TYPE,
    language: str = DEFAULT_LANGUAGE,
    target_language: Optional[str] = None,
    translate: bool = False,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    rag_top_k: int = DEFAULT_RAG_TOP_K,
    video_id: str | None = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> PipelineResult:
    """
    Complete Video / Meeting processing pipeline.

    Flow:

        Input
          ↓
        Audio Processing
          ↓
        Whisper GPU Transcription
          ↓
        Optional Translation
          ↓
        Title + Summary
          ↓
        Chroma Vector Store
          ↓
        Retriever
          ↓
        Mistral RAG
          ↓
        PipelineResult
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    source = validate_source(
        source
    )

    source_type = validate_source_type(
        source_type
    )

    language = validate_language(
        language
    )

    target_language = normalize_translation_language(
        target_language
    )

    # The requested translation configuration must be well-formed BEFORE any
    # cached result is considered: an enabled translation without a target
    # can never be satisfied by a cached record.
    if translate and not target_language:
        raise ValueError(
            "target_language is required "
            "when translate=True."
        )

    # chunk_minutes / rag_top_k are validated here, BEFORE any stale
    # recovery, database query, cache lookup, or expensive processing, so
    # invalid values are rejected before they can reach those paths.
    chunk_minutes = _validate_positive_int(
        chunk_minutes,
        name="chunk_minutes",
    )

    rag_top_k = _validate_positive_int(
        rag_top_k,
        name="rag_top_k",
    )

    recovered_stale_records = recover_stale_processing_videos()
    if recovered_stale_records:
        print(
            "Recovered stale processing records: "
            + ", ".join(recovered_stale_records)
        )

    # ========================================================
    # DATABASE DUPLICATE CHECK (configuration-aware)
    # ========================================================
    # YouTube videos have a stable canonical ID. If the same video has
    # already completed successfully, skip download/transcription/LLM
    # processing and reuse the stored result.
    if source_type == "youtube":
        video_id = video_id or get_youtube_video_id(source)
    else:
        upload_id = video_id or generate_meeting_id(source)
        video_id = build_meeting_processing_id(upload_id, language)
    if video_id:
        # A cached record is reused ONLY when it satisfies the CURRENT
        # request: the same source language AND the same translation config
        # (enabled/disabled + exact target). A translation-config mismatch
        # reuses the cached transcript where possible instead of
        # re-downloading/re-transcribing.
        cached_result = _load_cached_video_for_request(
            video_id,
            source=source,
            source_type=source_type,
            language=language,
            translate=translate,
            target_language=target_language,
            progress_callback=progress_callback,
        )
        if cached_result is not None:
            # Cached results stay RAG-lazy: transcript/summary/analysis load
            # now, but the per-video vector store / RAG chain is only
            # built-or-reloaded when the user actually asks a question
            # (app.py get_cached_rag_chain -> load_rag_chain reuses the
            # existing Chroma store). No eager embedding work on open.
            return cached_result

    # Register a new YouTube video as processing. Existing incomplete
    # records are reused instead of creating duplicate rows.
    if video_id:
        existing = get_video(video_id)
        if existing is None:
            create_video(
                video_id,
                source,
                source_type,
                language=language,
                status="processing",
            )
        elif existing.get("status") != "completed":
            update_video(
                video_id,
                source=source,
                source_type=source_type,
                language=language,
                status="processing",
                error_message=None,
            )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print("              AI VIDEO / MEETING ASSISTANT")
    print("=" * 70)

    print(
        f"Source type     : {source_type}"
    )

    print(
        f"Video ID        : {video_id or 'N/A'}"
    )

    print(
        f"Language        : {language}"
    )

    print(
        f"Target language  : "
        f"{target_language or 'None'}"
    )

    print(
        f"Translation     : "
        f"{'Enabled' if translate else 'Disabled'}"
    )

    print(
        f"Audio chunk size : "
        f"{chunk_minutes} minutes"
    )

    print(
        f"RAG Top-K        : {rag_top_k}"
    )

    print("=" * 70)

    # ========================================================
    # FAILURE-SAFE EXECUTION + CURRENT-RUN AUDIO CLEANUP
    # ========================================================

    temporary_chunks: list[str] = []
    try:
        try:
            return _execute_processing_stages(
                source=source,
                source_type=source_type,
                language=language,
                target_language=target_language,
                translate=translate,
                chunk_minutes=chunk_minutes,
                rag_top_k=rag_top_k,
                video_id=video_id,
                progress_callback=progress_callback,
                temporary_chunks=temporary_chunks,
            )
        except Exception as exc:
            # A registered record must never stay stuck in "processing":
            # persist the real failure, then re-raise so the UI still gets it.
            if video_id:
                try:
                    mark_video_failed(video_id, str(exc))
                except Exception:
                    pass  # never mask the original pipeline error
            raise
    finally:
        if temporary_chunks:
            print(
                f"Cleaning up {len(temporary_chunks)} temporary audio chunks"
            )
            try:
                # Pass only paths returned by this invocation. The existing
                # helper owns deletion semantics; no shared-directory sweep.
                cleanup_chunks(list(temporary_chunks))
            except Exception as cleanup_error:
                # Never replace a pipeline exception with a cleanup failure.
                print(
                    f"Warning: temporary audio cleanup failed: {cleanup_error}"
                )
            else:
                print("Temporary audio cleanup completed")


# ============================================================
# ASK QUESTION
# ============================================================

def ask_meeting(
    rag_chain: Any,
    question: str,
) -> str:
    """
    Current pipeline ke RAG chain se question answer karta hai.

    Example:

        result = run_pipeline(...)

        answer = ask_meeting(
            result.rag_chain,
            "What is LangGraph?"
        )
    """

    if rag_chain is None:

        raise ValueError(
            "RAG chain is not initialized. "
            "Run run_pipeline() first."
        )

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    try:

        answer = ask_question(
            rag_chain=rag_chain,
            question=question,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Question answering failed: {exc}"
        ) from exc

    return answer.strip()


# ============================================================
# RESULT → DICTIONARY
# ============================================================

def result_to_dict(
    result: PipelineResult,
) -> dict[str, Any]:
    """
    PipelineResult ko Streamlit-friendly dictionary mein
    convert karta hai.

    RAG chain dictionary mein intentionally include nahi hoti.
    """

    if not isinstance(
        result,
        PipelineResult,
    ):

        raise TypeError(
            "result must be a PipelineResult."
        )

    return {
        "source": result.source,

        "source_type": result.source_type,

        "language": result.language,

        "audio_chunks": result.audio_chunks,

        "transcript": result.transcript,

        "translated_transcript": (
            result.translated_transcript
        ),

        "title": result.title,

        "summary": result.summary,

        "actions": result.actions,

        "decisions": result.decisions,

        "questions": result.questions,

        "metadata": result.metadata,
    }


# ============================================================
# HIGH-LEVEL PROCESS FUNCTION
# ============================================================

def process_video(
    source: str,
    source_type: str = DEFAULT_SOURCE_TYPE,
    language: str = DEFAULT_LANGUAGE,
    target_language: Optional[str] = None,
    translate: bool = False,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    rag_top_k: int = DEFAULT_RAG_TOP_K,
    video_id: str | None = None,
) -> dict[str, Any]:
    """
    Streamlit/API ke liye simple high-level function.

    Returns:
        Serializable dictionary.
    """

    result = run_pipeline(
        source=source,
        source_type=source_type,
        language=language,
        target_language=target_language,
        translate=translate,
        chunk_minutes=chunk_minutes,
        rag_top_k=rag_top_k,
        video_id=video_id,
    )

    emit_progress(
        None,
        "complete",
        "Pipeline completed successfully.",
        current=1,
        total=1,
    )

    return result_to_dict(
        result
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("                 VIDEO AGENT")
    print("=" * 70)

    print(
        f"Default source : "
        f"{DEFAULT_SOURCE_TYPE}"
    )

    print(
        f"Default language : "
        f"{DEFAULT_LANGUAGE}"
    )

    print(
        f"Audio chunks : "
        f"{DEFAULT_CHUNK_MINUTES} minutes"
    )

    print(
        f"RAG Top-K : "
        f"{DEFAULT_RAG_TOP_K}"
    )

    print()
    print(
        "Main pipeline module loaded successfully."
    )

    print("=" * 70)
