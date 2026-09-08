"""Read-only analytics helpers for persisted Video Library records."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Callable, Iterable


CONTENT_COMPONENTS = (
    "transcript",
    "roman_urdu",
    "summary",
    "action_items",
    "decisions",
    "questions",
)

_TIMESTAMP_PATTERN = re.compile(
    r"(?:\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*\]|"
    r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b)"
)


def _record_value(record: dict[str, Any], *keys: str) -> Any:
    """Read a persisted field from a record or its metadata."""
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def has_content(value: Any) -> bool:
    """Return whether a value contains actual user-visible content."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _get_record_transcript(record: dict[str, Any]) -> Any:
    """Return the canonical transcript used by all analytics calculations."""
    transcript = record.get("_library_transcript")
    if not has_content(transcript):
        transcript = _record_value(record, "transcript", "original_transcript")
    return transcript


def get_video_content_status(record: dict[str, Any]) -> dict[str, bool]:
    """Return availability for the six persisted content components."""
    translation_record = record.get("_roman_urdu_record") or {}
    roman_urdu = record.get("roman_urdu_translation")
    if not has_content(roman_urdu) and isinstance(translation_record, dict):
        roman_urdu = translation_record.get("roman_urdu_translation")

    transcript = _get_record_transcript(record)

    return {
        "transcript": has_content(transcript),
        "roman_urdu": has_content(roman_urdu),
        "summary": has_content(_record_value(record, "summary", "executive_summary")),
        "action_items": has_content(_record_value(record, "actions", "action_items")),
        "decisions": has_content(_record_value(record, "decisions")),
        "questions": has_content(_record_value(record, "questions")),
    }


def calculate_completion_percentage(
    record: dict[str, Any],
    status: dict[str, bool] | None = None,
) -> float:
    """Calculate available content components divided by six."""
    content_status = status or get_video_content_status(record)
    available = sum(1 for component in CONTENT_COMPONENTS if content_status.get(component, False))
    return round((available / len(CONTENT_COMPONENTS)) * 100, 2)


def _timestamp_to_seconds(timestamp: str) -> float | None:
    text = timestamp.strip().strip("[]").strip()
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            hours = 0
        elif len(parts) == 3:
            hours, minutes, seconds = parts
        else:
            return None
        hours_value = int(hours)
        minutes_value = int(minutes)
        seconds_value = float(seconds)
    except (TypeError, ValueError):
        return None

    if hours_value < 0 or not 0 <= minutes_value < 60 or not 0 <= seconds_value < 60:
        return None
    return hours_value * 3600 + minutes_value * 60 + seconds_value


def extract_duration_from_timestamps(transcript: str) -> float | None:
    """Return the final valid transcript timestamp in seconds, if available."""
    valid_seconds = [
        seconds
        for match in _TIMESTAMP_PATTERN.finditer(transcript or "")
        if (seconds := _timestamp_to_seconds(match.group(0))) is not None
    ]
    return valid_seconds[-1] if valid_seconds else None


def _parse_duration_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) >= 0 else None
    parsed = _timestamp_to_seconds(str(value))
    if parsed is not None:
        return parsed
    try:
        numeric = float(str(value).strip())
        return numeric if numeric >= 0 else None
    except (TypeError, ValueError):
        return None


def get_reliable_duration(record: dict[str, Any]) -> float | None:
    """Prefer real duration metadata, then the final valid transcript timestamp."""
    duration = _record_value(record, "duration", "duration_seconds", "video_duration")
    parsed_duration = _parse_duration_value(duration)
    if parsed_duration is not None:
        return parsed_duration
    transcript = _get_record_transcript(record)
    return extract_duration_from_timestamps(str(transcript or ""))


def format_duration(seconds: float | None) -> str:
    """Format reliable duration as hours/minutes without inventing missing data."""
    if seconds is None:
        return "Duration unavailable"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours:02d}h {minutes:02d}m"
    return f"{minutes:02d}m"


def _source_name(record: dict[str, Any]) -> str:
    return str(
        record.get("source_name")
        or record.get("title")
        or record.get("source")
        or "Untitled Source"
    )


def _parse_date(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    for date_format in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, date_format).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _record_date(record: dict[str, Any]) -> datetime | None:
    return _parse_date(
        _record_value(
            record,
            "created_at",
            "processed_at",
            "completed_at",
            "finished_at",
            "date",
            "updated_at",
        )
    )


def calculate_library_analytics(
    records: Iterable[dict[str, Any]],
    source_name_getter: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Calculate read-only analytics from one already-loaded record collection."""
    items = list(records)
    get_name = source_name_getter or _source_name
    statuses = [get_video_content_status(record) for record in items]

    total_words = 0
    for record, status in zip(items, statuses):
        if status["transcript"]:
            transcript = _get_record_transcript(record)
            total_words += len(str(transcript or "").split())

    durations = [duration for duration in (get_reliable_duration(record) for record in items) if duration is not None]
    eligible_count = sum(status["transcript"] for status in statuses)
    translated_count = sum(
        status["transcript"] and status["roman_urdu"]
        for status in statuses
    )
    coverage = (translated_count / eligible_count * 100) if eligible_count else 0.0

    completion_entries = []
    for record, status in zip(items, statuses):
        completion_entries.append(
            {
                "source_name": get_name(record),
                "percentage": calculate_completion_percentage(record, status),
                "status": status,
                "date": _record_date(record),
            }
        )

    most_complete = None
    if completion_entries:
        highest = max(entry["percentage"] for entry in completion_entries)
        tied = [entry for entry in completion_entries if entry["percentage"] == highest]
        dated_tied = [entry for entry in tied if entry["date"] is not None]
        if dated_tied:
            most_complete = max(
                dated_tied,
                key=lambda entry: (entry["date"], entry["source_name"].casefold()),
            )
        else:
            most_complete = min(tied, key=lambda entry: entry["source_name"].casefold())

    dated_entries = [entry for entry in completion_entries if entry["date"] is not None]
    most_recent = (
        max(dated_entries, key=lambda entry: (entry["date"], entry["source_name"].casefold()))
        if dated_entries
        else None
    )

    component_totals = {
        component: sum(status[component] for status in statuses)
        for component in CONTENT_COMPONENTS
    }
    available_components = sum(component_totals.values())
    overall_completion = (
        available_components / (len(items) * len(CONTENT_COMPONENTS)) * 100
        if items
        else 0.0
    )

    return {
        "total_videos": len(items),
        "total_words": total_words,
        "duration_seconds": sum(durations) if durations else None,
        "duration_video_count": len(durations),
        "roman_urdu_coverage": round(coverage, 2),
        "roman_urdu_translated_count": translated_count,
        "roman_urdu_eligible_count": eligible_count,
        "component_totals": component_totals,
        "overall_completion": round(overall_completion, 2),
        "missing_roman_urdu": len(items) - translated_count,
        "missing_summary": len(items) - component_totals["summary"],
        "incomplete_videos": sum(entry["percentage"] < 100 for entry in completion_entries),
        "completion_entries": completion_entries,
        "most_complete": most_complete,
        "most_recent": most_recent,
    }


__all__ = [
    "CONTENT_COMPONENTS",
    "calculate_completion_percentage",
    "calculate_library_analytics",
    "extract_duration_from_timestamps",
    "format_duration",
    "get_reliable_duration",
    "get_video_content_status",
    "has_content",
]
