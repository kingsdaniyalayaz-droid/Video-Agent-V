"""Persistent local cache for completed Roman Urdu translations."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "translations.db"


def _normalized_transcript(original_transcript: str) -> str:
    """Normalize only the hash input, preserving the stored transcript exactly."""
    normalized = (original_transcript or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines: list[str] = []
    previous_blank = False
    for line in normalized.split("\n"):
        if line.strip():
            lines.append(line)
            previous_blank = False
        elif not previous_blank:
            lines.append("")
            previous_blank = True
    return "\n".join(lines).strip()


def _database_parent() -> Path:
    return Path(DATABASE_PATH).expanduser().resolve().parent


def _create_translation_table(connection: sqlite3.Connection, table_name: str = "roman_urdu_translations") -> None:
    """Create the source-aware table shape without changing transcript hashing."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT NOT NULL,
            original_transcript TEXT NOT NULL,
            roman_urdu_translation TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            model_name TEXT,
            source_name TEXT,
            video_id TEXT,
            UNIQUE(content_hash, video_id)
        )
        """
    )


def _has_content_hash_unique_index(connection: sqlite3.Connection) -> bool:
    """Return whether the legacy single-column content_hash uniqueness remains."""
    for index in connection.execute(
        "PRAGMA index_list(roman_urdu_translations)"
    ).fetchall():
        if not index[2]:
            continue
        index_name = str(index[1]).replace("'", "''")
        columns = [
            row[2]
            for row in connection.execute(
                f"PRAGMA index_info('{index_name}')"
            ).fetchall()
        ]
        if columns == ["content_hash"]:
            return True
    return False


def _migrate_translation_table(connection: sqlite3.Connection, columns: set[str]) -> None:
    """Rebuild only the cache table when SQLite cannot alter its unique constraint."""
    backup_table = "roman_urdu_translations_legacy_backup"
    connection.execute(f"DROP TABLE IF EXISTS {backup_table}")
    connection.execute(
        f"ALTER TABLE roman_urdu_translations RENAME TO {backup_table}"
    )
    _create_translation_table(connection)
    source_columns = [
        "id",
        "content_hash",
        "original_transcript",
        "roman_urdu_translation",
        "created_at",
        "updated_at",
        "model_name",
    ]
    if "source_name" in columns:
        source_columns.append("source_name")
    else:
        source_columns.append("NULL AS source_name")
    if "video_id" in columns:
        source_columns.append("video_id")
    else:
        source_columns.append("NULL AS video_id")
    connection.execute(
        """
        INSERT INTO roman_urdu_translations
            (id, content_hash, original_transcript, roman_urdu_translation,
             created_at, updated_at, model_name, source_name, video_id)
        SELECT """ + ", ".join(source_columns) + f" FROM {backup_table}"
    )
    connection.execute(f"DROP TABLE {backup_table}")


def initialize_translation_database() -> None:
    """Create or safely upgrade the cache table without deleting translation rows."""
    _database_parent().mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'roman_urdu_translations'"
        ).fetchone()
        if not table_exists:
            _create_translation_table(connection)
            return

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(roman_urdu_translations)"
            ).fetchall()
        }
        needs_rebuild = _has_content_hash_unique_index(connection)
        if needs_rebuild:
            _migrate_translation_table(connection, columns)
        else:
            if "source_name" not in columns:
                connection.execute(
                    "ALTER TABLE roman_urdu_translations ADD COLUMN source_name TEXT"
                )
            if "video_id" not in columns:
                connection.execute(
                    "ALTER TABLE roman_urdu_translations ADD COLUMN video_id TEXT"
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_roman_urdu_translations_hash_video "
                "ON roman_urdu_translations(content_hash, video_id)"
            )


def get_transcript_hash(original_transcript: str) -> str:
    """Return the SHA-256 hash of the normalized transcript."""
    normalized = _normalized_transcript(original_transcript)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_cached_translation(
    original_transcript: str,
    video_id: Optional[str] = None,
) -> Optional[str]:
    """Return a cached translation for the transcript/source, or None on a miss."""
    if not (original_transcript or "").strip():
        return None
    initialize_translation_database()
    content_hash = get_transcript_hash(original_transcript)
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        if video_id:
            row = connection.execute(
                """
                SELECT roman_urdu_translation
                FROM roman_urdu_translations
                WHERE content_hash = ? AND video_id = ?
                UNION ALL
                SELECT roman_urdu_translation
                FROM roman_urdu_translations
                WHERE content_hash = ? AND video_id IS NULL
                LIMIT 1
                """,
                (content_hash, str(video_id), content_hash),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT roman_urdu_translation
                FROM roman_urdu_translations
                WHERE content_hash = ?
                ORDER BY CASE WHEN video_id IS NULL THEN 0 ELSE 1 END, updated_at DESC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
    return row[0] if row is not None else None


def save_translation(
    original_transcript: str,
    roman_urdu_translation: str,
    model_name: Optional[str] = None,
    source_name: Optional[str] = None,
    video_id: Optional[str] = None,
) -> None:
    """Save or update a non-empty completed translation without duplication."""
    if not (original_transcript or "").strip() or not (roman_urdu_translation or "").strip():
        return

    initialize_translation_database()
    content_hash = get_transcript_hash(original_transcript)
    now = datetime.now().astimezone().isoformat()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        if video_id:
            connection.execute(
                """
                INSERT INTO roman_urdu_translations (
                    content_hash,
                    original_transcript,
                    roman_urdu_translation,
                    created_at,
                    updated_at,
                    model_name,
                    source_name,
                    video_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash, video_id) DO UPDATE SET
                    original_transcript = excluded.original_transcript,
                    roman_urdu_translation = excluded.roman_urdu_translation,
                    updated_at = excluded.updated_at,
                    model_name = excluded.model_name,
                    source_name = excluded.source_name
                """,
                (
                    content_hash,
                    original_transcript,
                    roman_urdu_translation,
                    now,
                    now,
                    model_name,
                    source_name,
                    str(video_id),
                ),
            )
        else:
            existing = connection.execute(
                """
                SELECT id
                FROM roman_urdu_translations
                WHERE content_hash = ?
                ORDER BY CASE WHEN video_id IS NULL THEN 0 ELSE 1 END, updated_at DESC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            if existing is not None:
                connection.execute(
                    """
                    UPDATE roman_urdu_translations
                    SET original_transcript = ?,
                        roman_urdu_translation = ?,
                        updated_at = ?,
                        model_name = ?,
                        source_name = ?
                    WHERE id = ?
                    """,
                    (
                        original_transcript,
                        roman_urdu_translation,
                        now,
                        model_name,
                        source_name,
                        existing[0],
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO roman_urdu_translations (
                        content_hash,
                        original_transcript,
                        roman_urdu_translation,
                        created_at,
                        updated_at,
                        model_name,
                        source_name,
                        video_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_hash,
                        original_transcript,
                        roman_urdu_translation,
                        now,
                        now,
                        model_name,
                        source_name,
                        None,
                    ),
                )


def delete_cached_translation(original_transcript: str) -> bool:
    """Delete a cached transcript and return whether a row was removed."""
    if not (original_transcript or "").strip():
        return False
    initialize_translation_database()
    content_hash = get_transcript_hash(original_transcript)
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        cursor = connection.execute(
            "DELETE FROM roman_urdu_translations WHERE content_hash = ?",
            (content_hash,),
        )
        return cursor.rowcount > 0


def get_translation_count() -> int:
    """Return the number of cached translation records."""
    initialize_translation_database()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        row = connection.execute("SELECT COUNT(*) FROM roman_urdu_translations").fetchone()
    return int(row[0]) if row is not None else 0


def list_saved_translations(
    limit: int = 50,
    search_query: str = "",
) -> list[dict[str, object]]:
    """List recent cached translations, optionally matching either text field."""
    initialize_translation_database()
    safe_limit = max(1, min(int(limit), 500))
    query = (search_query or "").strip()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        if query:
            pattern = f"%{query}%"
            rows = connection.execute(
                """
                SELECT id, content_hash, original_transcript,
                       roman_urdu_translation, created_at, updated_at, model_name, source_name, video_id
                FROM roman_urdu_translations
                WHERE original_transcript LIKE ?
                   OR roman_urdu_translation LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, pattern, safe_limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, content_hash, original_transcript,
                       roman_urdu_translation, created_at, updated_at, model_name, source_name, video_id
                FROM roman_urdu_translations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
    return [dict(row) for row in rows]


def get_saved_translation_by_hash(
    content_hash: str,
    video_id: Optional[str] = None,
) -> dict[str, object] | None:
    """Return a source-aware translation record, with legacy fallback."""
    initialize_translation_database()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        if video_id:
            row = connection.execute(
                """
                SELECT id, content_hash, original_transcript,
                       roman_urdu_translation, created_at, updated_at, model_name, source_name, video_id
                FROM roman_urdu_translations
                WHERE content_hash = ? AND video_id = ?
                UNION ALL
                SELECT id, content_hash, original_transcript,
                       roman_urdu_translation, created_at, updated_at, model_name, source_name, video_id
                FROM roman_urdu_translations
                WHERE content_hash = ? AND video_id IS NULL
                LIMIT 1
                """,
                (content_hash, str(video_id), content_hash),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT id, content_hash, original_transcript,
                       roman_urdu_translation, created_at, updated_at, model_name, source_name, video_id
                FROM roman_urdu_translations
                WHERE content_hash = ?
                ORDER BY CASE WHEN video_id IS NULL THEN 0 ELSE 1 END, updated_at DESC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()

    return dict(row) if row is not None else None


def get_saved_translation_by_id(translation_id: int) -> dict[str, object] | None:
    """Return one complete cached translation record by database ID."""
    initialize_translation_database()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, content_hash, original_transcript,
                   roman_urdu_translation, created_at, updated_at, model_name, source_name, video_id
            FROM roman_urdu_translations
            WHERE id = ?
            """,
            (int(translation_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def delete_translation_by_id(translation_id: int) -> bool:
    """Delete one cached translation by ID and report whether it existed."""
    initialize_translation_database()
    with sqlite3.connect(str(DATABASE_PATH)) as connection:
        cursor = connection.execute(
            "DELETE FROM roman_urdu_translations WHERE id = ?",
            (int(translation_id),),
        )
        return cursor.rowcount > 0


__all__ = [
    "initialize_translation_database",
    "get_transcript_hash",
    "get_cached_translation",
    "save_translation",
    "delete_cached_translation",
    "get_translation_count",
    "list_saved_translations",
    "get_saved_translation_by_hash",
    "get_saved_translation_by_id",
    "delete_translation_by_id",
]
