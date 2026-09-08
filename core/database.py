from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "video_agent.db"
)

DB_PATH = Path(
    os.getenv("VIDEO_DB_PATH", str(DEFAULT_DB_PATH))
).expanduser()


# ============================================================
# DATABASE SCHEMA
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    video_id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,

    title TEXT,
    language TEXT,

    transcript TEXT,
    translated_transcript TEXT,
    translation_enabled INTEGER NOT NULL DEFAULT 0,
    target_language TEXT,
    summary TEXT,
    actions TEXT DEFAULT '[]',
    decisions TEXT DEFAULT '[]',
    questions TEXT DEFAULT '[]',

    analysis_status TEXT,
    analysis_error TEXT,

    duration REAL,
    chunk_count INTEGER,

    status TEXT NOT NULL DEFAULT 'processing',
    error_message TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_videos_status
ON videos(status);

CREATE INDEX IF NOT EXISTS idx_videos_source_type
ON videos(source_type);

CREATE INDEX IF NOT EXISTS idx_videos_updated_at
ON videos(updated_at);
"""


# ============================================================
# CONNECTION
# ============================================================

def get_db_path() -> Path:
    """Return the configured SQLite database path."""
    return DB_PATH


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Open a reliable local SQLite connection."""

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    _ensure_schema(connection)

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# INITIALIZATION
# ============================================================

def init_db() -> None:
    """
    Create the persistent database and tables.

    Safe to call every time the application starts.
    Existing records are preserved.
    """

    with get_connection() as connection:
        _ensure_schema(connection)

    print(f"Database initialized: {DB_PATH}")


# ============================================================
# HELPERS
# ============================================================

def _now() -> str:
    """Return a UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


VALID_VIDEO_STATUSES = {
    "processing",
    "completed",
    "failed",
}


def _normalize_and_validate_status(status: Any) -> str:
    """Normalize and validate a video status value.

    Returns the canonical lowercase status.  Raises ``ValueError`` for
    ``None``, empty, or values outside the application's allowed lifecycle
    states (``processing`` / ``completed`` / ``failed``) so invalid states
    can never be persisted.
    """

    if status is None:
        raise ValueError(
            "status cannot be None. "
            f"Allowed statuses: {', '.join(sorted(VALID_VIDEO_STATUSES))}"
        )

    if not isinstance(status, str):
        raise ValueError(
            f"status must be a string, got {type(status).__name__}. "
            f"Allowed statuses: {', '.join(sorted(VALID_VIDEO_STATUSES))}"
        )

    normalized = status.strip().lower()

    if not normalized:
        raise ValueError(
            "status cannot be empty. "
            f"Allowed statuses: {', '.join(sorted(VALID_VIDEO_STATUSES))}"
        )

    if normalized not in VALID_VIDEO_STATUSES:
        raise ValueError(
            f"Invalid status: {status!r}. "
            f"Allowed statuses: {', '.join(sorted(VALID_VIDEO_STATUSES))}"
        )

    return normalized


def _normalize_video_id(video_id: Any) -> str:
    """Validate and normalize a video_id. None and any non-string value
    (including falsy values like 0 / False and truthy values like 123 /
    [] / {}) raise a clear ValueError; empty or whitespace-only strings
    also raise. Returns the stripped canonical video_id; never coerces.
    """

    if video_id is None:
        raise ValueError("video_id cannot be None.")

    if not isinstance(video_id, str):
        raise ValueError(
            "video_id must be a string. "
            f"Got {type(video_id).__name__}."
        )

    normalized = video_id.strip()

    if not normalized:
        raise ValueError("video_id cannot be empty.")

    return normalized


def _normalize_and_validate_source(source: Any) -> str:
    """Validate and normalize a source value. None, empty strings,
    whitespace-only strings, and non-string values raise a clear
    ValueError. Returns the stripped source; never coerces.
    """

    if source is None:
        raise ValueError("source cannot be None.")

    if not isinstance(source, str):
        raise ValueError(
            "source must be a string. "
            f"Got {type(source).__name__}."
        )

    normalized = source.strip()

    if not normalized:
        raise ValueError("source cannot be empty.")

    return normalized


def _normalize_and_validate_source_type(source_type: Any) -> str:
    """Validate and normalize a source_type. Allowed values are exactly
    youtube and meeting (case/whitespace normalized). None, empty,
    non-string, or unknown values raise a clear ValueError; never coerces.
    """

    if source_type is None:
        raise ValueError("source_type cannot be None.")

    if not isinstance(source_type, str):
        raise ValueError(
            "source_type must be a string. "
            f"Got {type(source_type).__name__}."
        )

    normalized = source_type.strip().lower()

    if normalized not in {"youtube", "meeting"}:
        raise ValueError(
            "source_type must be 'youtube' or 'meeting'. "
            f"Got {source_type!r}."
        )

    return normalized


def _validate_strict_int(value: Any, *, name: str, minimum: int) -> int:
    """Validate an integer-valued parameter strictly. bool is rejected
    because it is an int subclass; None, non-int values, and values below
    minimum raise a clear ValueError. Returns the validated integer.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer. "
            f"Got {type(value).__name__}."
        )

    if value < minimum:
        if minimum > 0:
            raise ValueError(f"{name} must be greater than {minimum - 1}.")
        raise ValueError(f"{name} cannot be negative.")

    return value


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse persisted timestamps defensively."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_path_missing(record: dict[str, Any]) -> bool:
    """Return True when a local meeting source points to a missing file."""
    if str(record.get("source_type") or "").strip().lower() != "meeting":
        return False
    source = str(record.get("source") or "").strip()
    if not source:
        return True
    return not Path(source).exists()


_ANALYSIS_COLUMNS = {
    "actions",
    "decisions",
    "questions",
}


def _normalize_analysis_value(value: Any) -> list[Any]:
    """Normalize persisted analysis content to a JSON-safe list."""
    if value in (None, ""):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return parsed if isinstance(parsed, list) else [parsed]

    return [value]


def _serialize_analysis_value(value: Any) -> str:
    return json.dumps(
        _normalize_analysis_value(value),
        ensure_ascii=False,
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the base schema and apply additive backward-compatible migrations."""
    connection.executescript(SCHEMA)

    existing_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(videos)"
        ).fetchall()
    }

    migrations = {
        "actions": "TEXT DEFAULT '[]'",
        "decisions": "TEXT DEFAULT '[]'",
        "questions": "TEXT DEFAULT '[]'",
        "analysis_status": "TEXT",
        "analysis_error": "TEXT",
        "translation_enabled": "INTEGER NOT NULL DEFAULT 0",
        "target_language": "TEXT",
    }

    for column_name, column_type in migrations.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE videos ADD COLUMN {column_name} {column_type}"
            )


def _row_to_dict(
    row: sqlite3.Row | None,
) -> dict[str, Any] | None:
    """Convert a SQLite row to a dictionary."""

    if row is None:
        return None

    record = dict(row)

    for column_name in _ANALYSIS_COLUMNS:
        record[column_name] = _normalize_analysis_value(
            record.get(column_name)
        )

    return record


# ============================================================
# LOOKUP
# ============================================================

def get_video(
    video_id: str,
) -> dict[str, Any] | None:
    """Return one video record by its canonical ID."""

    video_id = _normalize_video_id(video_id)

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM videos
            WHERE video_id = ?
            """,
            (video_id,),
        ).fetchone()

    return _row_to_dict(row)


def video_exists(
    video_id: str,
    completed_only: bool = False,
) -> bool:
    """
    Check whether a video exists.

    completed_only=True is what the pipeline will use before
    deciding whether expensive processing can be skipped.
    """

    video_id = _normalize_video_id(video_id)

    if completed_only:
        query = """
            SELECT 1
            FROM videos
            WHERE video_id = ?
              AND status = 'completed'
            LIMIT 1
        """
    else:
        query = """
            SELECT 1
            FROM videos
            WHERE video_id = ?
            LIMIT 1
        """

    with get_connection() as connection:
        row = connection.execute(
            query,
            (video_id,),
        ).fetchone()

    return row is not None


# ============================================================
# CREATE
# ============================================================

def create_video(
    video_id: str,
    source: str,
    source_type: str,
    *,
    title: str | None = None,
    language: str | None = None,
    status: str = "processing",
) -> dict[str, Any]:
    """
    Register a video before expensive processing begins.
    """

    video_id = _normalize_video_id(video_id)
    source = _normalize_and_validate_source(source)
    source_type = _normalize_and_validate_source_type(source_type)
    status = _normalize_and_validate_status(status)

    now = _now()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO videos (
                video_id,
                source,
                source_type,
                title,
                language,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                source,
                source_type,
                title,
                language,
                status,
                now,
                now,
            ),
        )

    return get_video(video_id)  # type: ignore[return-value]


# ============================================================
# UPDATE
# ============================================================

def update_video(
    video_id: str,
    **fields: Any,
) -> dict[str, Any]:
    """Update only the supplied fields of an existing video."""

    video_id = _normalize_video_id(video_id)

    allowed_fields = {
        "source",
        "source_type",
        "title",
        "language",
        "transcript",
        "translated_transcript",
        "translation_enabled",
        "target_language",
        "summary",
        "actions",
        "decisions",
        "questions",
        "duration",
        "chunk_count",
        "status",
        "error_message",
        "analysis_status",
        "analysis_error",
    }

    unknown_fields = set(fields) - allowed_fields

    if unknown_fields:
        raise ValueError(
            "Unsupported video fields: "
            + ", ".join(sorted(unknown_fields))
        )

    if not fields:
        existing = get_video(video_id)

        if existing is None:
            raise KeyError(
                f"Video not found: {video_id}"
            )

        return existing

    if "translation_enabled" in fields and fields["translation_enabled"] is not None:
        fields["translation_enabled"] = int(bool(fields["translation_enabled"]))

    if "target_language" in fields:
        value = fields["target_language"]
        fields["target_language"] = (
            str(value).strip()
            if value is not None and str(value).strip()
            else None
        )

    if "source" in fields:
        fields["source"] = _normalize_and_validate_source(
            fields["source"]
        )

    if "source_type" in fields:
        fields["source_type"] = _normalize_and_validate_source_type(
            fields["source_type"]
        )

    if "status" in fields:
        fields["status"] = _normalize_and_validate_status(
            fields["status"]
        )

    for column_name in _ANALYSIS_COLUMNS:
        if column_name in fields:
            fields[column_name] = _serialize_analysis_value(
                fields[column_name]
            )

    fields["updated_at"] = _now()

    assignments = ", ".join(
        f"{field} = ?"
        for field in fields
    )

    values = list(fields.values())
    values.append(video_id)

    with get_connection() as connection:
        cursor = connection.execute(
            f"""
            UPDATE videos
            SET {assignments}
            WHERE video_id = ?
            """,
            values,
        )

        if cursor.rowcount == 0:
            raise KeyError(
                f"Video not found: {video_id}"
            )

    return get_video(video_id)  # type: ignore[return-value]


# ============================================================
# PIPELINE SAVE FUNCTIONS
# ============================================================

def save_completed_video(
    video_id: str,
    *,
    title: str | None = None,
    language: str | None = None,
    transcript: str | None = None,
    translated_transcript: str | None = None,
    translation_enabled: bool | None = None,
    target_language: str | None = None,
    summary: str | None = None,
    actions: list[Any] | None = None,
    decisions: list[Any] | None = None,
    questions: list[Any] | None = None,
    duration: float | None = None,
    chunk_count: int | None = None,
    source: str | None = None,
    source_type: str | None = None,
    analysis_status: str | None = None,
    analysis_error: str | None = None,
) -> dict[str, Any]:
    """
    Save the final successful pipeline result atomically.

    A single ``INSERT ... ON CONFLICT(video_id) DO UPDATE`` upsert creates
    the record when it does not exist and updates it in place when it does,
    so concurrent saves of the same video_id cannot race into a UNIQUE
    constraint failure.  The record is written directly with
    status='completed'; existing 'processing' or 'failed' records are
    safely transitioned to 'completed' by the same statement.
    """

    video_id = _normalize_video_id(video_id)

    # ---------------------------------------------------------
    # Determine source type automatically if not provided
    # ---------------------------------------------------------
    if source_type is None:
        if video_id.startswith("meeting_"):
            source_type = "meeting"
        else:
            source_type = "youtube"

    source_type = _normalize_and_validate_source_type(source_type)

    # ---------------------------------------------------------
    # Determine source automatically if not provided
    # ---------------------------------------------------------
    if source is None:
        source = video_id

    source = _normalize_and_validate_source(source)

    # ---------------------------------------------------------
    # Normalize fields consistently with update_video()
    # ---------------------------------------------------------
    if translation_enabled is not None:
        translation_enabled = int(bool(translation_enabled))
    else:
        # Column is NOT NULL DEFAULT 0: a None value binds the schema
        # default (translation disabled).
        translation_enabled = 0

    if target_language is not None and str(target_language).strip():
        target_language = str(target_language).strip()
    else:
        target_language = None

    actions_json = _serialize_analysis_value(actions)
    decisions_json = _serialize_analysis_value(decisions)
    questions_json = _serialize_analysis_value(questions)

    now = _now()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO videos (
                video_id,
                source,
                source_type,
                title,
                language,
                transcript,
                translated_transcript,
                translation_enabled,
                target_language,
                summary,
                actions,
                decisions,
                questions,
                analysis_status,
                analysis_error,
                duration,
                chunk_count,
                status,
                error_message,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title = excluded.title,
                language = excluded.language,
                transcript = excluded.transcript,
                translated_transcript = excluded.translated_transcript,
                translation_enabled = excluded.translation_enabled,
                target_language = excluded.target_language,
                summary = excluded.summary,
                actions = excluded.actions,
                decisions = excluded.decisions,
                questions = excluded.questions,
                analysis_status = excluded.analysis_status,
                analysis_error = excluded.analysis_error,
                duration = excluded.duration,
                chunk_count = excluded.chunk_count,
                status = excluded.status,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                video_id,
                source,
                source_type,
                title,
                language,
                transcript,
                translated_transcript,
                translation_enabled,
                target_language,
                summary,
                actions_json,
                decisions_json,
                questions_json,
                analysis_status,
                analysis_error,
                duration,
                chunk_count,
                "completed",
                None,  # error_message
                now,   # created_at
                now,   # updated_at
            ),
        )

    return get_video(video_id)  # type: ignore[return-value]

def mark_video_failed(
    video_id: str,
    error_message: str,
) -> dict[str, Any]:
    """Mark a registered video as failed."""

    return update_video(
        video_id,
        status="failed",
        error_message=str(error_message),
    )


def recover_stale_processing_videos(
    *,
    stale_after_seconds: int = 1800,
) -> list[str]:
    """Mark abandoned processing rows as failed so they remain retryable.

    Recovery is intentionally conservative:
    - meeting records whose local source file no longer exists -> failed
    - any record still in processing beyond the stale threshold -> failed
    """
    stale_after_seconds = _validate_strict_int(
        stale_after_seconds,
        name="stale_after_seconds",
        minimum=0,
    )

    now = datetime.now(timezone.utc)
    recovered_ids: list[str] = []

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM videos
            WHERE status = 'processing'
            """
        ).fetchall()

        for row in rows:
            record = _row_to_dict(row)
            if record is None:
                continue

            video_id = str(record.get("video_id") or "").strip()
            if not video_id:
                continue

            reason: str | None = None
            if _source_path_missing(record):
                reason = (
                    "Stale processing record recovered: local meeting source file "
                    "is no longer available."
                )
            else:
                updated_at = _parse_timestamp(record.get("updated_at"))
                created_at = _parse_timestamp(record.get("created_at"))
                last_seen = updated_at or created_at
                if last_seen is not None:
                    age_seconds = (now - last_seen).total_seconds()
                    if age_seconds >= stale_after_seconds:
                        reason = (
                            "Stale processing record recovered: previous processing "
                            "did not finish successfully."
                        )

            if reason is None:
                continue

            connection.execute(
                """
                UPDATE videos
                SET status = 'failed',
                    error_message = ?,
                    updated_at = ?
                WHERE video_id = ?
                  AND status = 'processing'
                """,
                (reason, _now(), video_id),
            )
            recovered_ids.append(video_id)

    return recovered_ids


# ============================================================
# LIST
# ============================================================

def list_videos(
    *,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return a page of stored videos for the Knowledge Base UI."""

    limit = _validate_strict_int(limit, name="limit", minimum=1)
    offset = _validate_strict_int(offset, name="offset", minimum=0)

    filter_status: str | None = None
    if status is not None:
        if isinstance(status, str) and not status.strip():
            # Empty / whitespace-only status preserves the existing
            # no-filter behavior.
            filter_status = None
        else:
            filter_status = _normalize_and_validate_status(status)

    if filter_status is not None:
        query = """
            SELECT *
            FROM videos
            WHERE status = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """
        params = (
            filter_status,
            limit,
            offset,
        )
    else:
        query = """
            SELECT *
            FROM videos
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """
        params = (limit, offset)

    with get_connection() as connection:
        rows = connection.execute(
            query,
            params,
        ).fetchall()

    return [_row_to_dict(row) for row in rows if row is not None]


# ============================================================
# DELETE
# ============================================================

def delete_video(
    video_id: str,
) -> None:
    """
    Delete the SQLite registry record.

    Chroma vectors are intentionally NOT deleted here. The
    video-aware vector-store layer will own vector deletion.
    """

    video_id = _normalize_video_id(video_id)

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM videos
            WHERE video_id = ?
            """,
            (video_id,),
        )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO AGENT DATABASE")
    print("=" * 60)

    init_db()

    print(f"Database : {DB_PATH}")
    print("SQLite   : READY")
    print(f"Records  : {len(list_videos())}")
    print("=" * 60)
