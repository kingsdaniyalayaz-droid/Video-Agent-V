"""
core/media_validation.py

Production-grade validation for uploaded audio/video files.

Security model:
    1. Filename extension allowlist
    2. Maximum file-size enforcement
    3. Regular-file validation
    4. ffprobe container inspection
    5. Actual format validation
    6. Audio-stream requirement
    7. Duration sanity checks
    8. Stream metadata validation

This module intentionally does not trust:
    - filename extensions
    - browser MIME types
    - client-provided metadata

FFprobe is the authoritative container/media validator.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================

class MediaValidationError(ValueError):
    """Raised when an uploaded file fails media validation."""


class MediaProbeError(MediaValidationError):
    """Raised when ffprobe cannot inspect the uploaded file."""


# ============================================================
# CONFIGURATION
# ============================================================

BYTES_PER_MB: Final[int] = 1024 * 1024
BYTES_PER_GB: Final[int] = 1024 * 1024 * 1024


HARD_MAX_UPLOAD_SIZE_GB: Final[int] = 4
DEFAULT_MAX_UPLOAD_BYTES: Final[int] = HARD_MAX_UPLOAD_SIZE_GB * BYTES_PER_GB

# The single source of truth for the upload policy. Values are normalized
# ffprobe format aliases valid for each extension.
EXTENSION_TO_FORMATS: Final[dict[str, frozenset[str]]] = {
    ".mp4": frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}),
    ".mov": frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}),
    ".avi": frozenset({"avi"}),
    # ffprobe reports the shared Matroska/WebM demuxer as
    # "matroska,webm"; both are aliases of the permitted container family.
    ".mkv": frozenset({"matroska", "webm"}),
    ".webm": frozenset({"matroska", "webm"}),
    ".mp3": frozenset({"mp3"}),
    ".wav": frozenset({"wav"}),
    ".m4a": frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}),
    ".aac": frozenset({"aac"}),
    ".ogg": frozenset({"ogg"}),
    ".opus": frozenset({"ogg", "opus"}),
}
ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset(EXTENSION_TO_FORMATS)
ALLOWED_FORMATS: Final[frozenset[str]] = frozenset(
    format_name for formats in EXTENSION_TO_FORMATS.values() for format_name in formats
)
# Compatibility alias for callers that use the global container policy. The
# per-extension check in validate_media_file remains the authoritative guard.
ALLOWED_CONTAINER_FORMATS: Final[frozenset[str]] = ALLOWED_FORMATS


MAX_DURATION_SECONDS: Final[float] = float(
    os.getenv(
        "MAX_MEDIA_DURATION_SECONDS",
        str(24 * 60 * 60),
    )
)


FFPROBE_TIMEOUT_SECONDS: Final[int] = int(
    os.getenv(
        "FFPROBE_TIMEOUT_SECONDS",
        "30",
    )
)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass(frozen=True)
class MediaValidationResult:
    """
    Immutable result of successful media validation.
    """

    path: Path
    extension: str
    size_bytes: int
    duration_seconds: float
    format_names: tuple[str, ...]
    audio_codec: str | None
    video_codec: str | None
    audio_stream_count: int
    video_stream_count: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / BYTES_PER_MB

    @property
    def has_video(self) -> bool:
        return self.video_stream_count > 0

    @property
    def has_audio(self) -> bool:
        return self.audio_stream_count > 0


# ============================================================
# EXECUTABLE DISCOVERY
# ============================================================

def get_ffprobe_path() -> str:
    """
    Resolve ffprobe from:

        1. FFPROBE_PATH environment variable
        2. PATH

    Raises:
        RuntimeError:
            If ffprobe cannot be found.
    """

    configured = (
        os.getenv("FFPROBE_PATH") or ""
    ).strip()

    if configured:
        configured_path = Path(configured).expanduser()

        if configured_path.exists():
            return str(configured_path)

        raise RuntimeError(
            "FFPROBE_PATH is configured but does not exist: "
            f"{configured_path}"
        )

    ffprobe = shutil.which("ffprobe")

    if ffprobe:
        return ffprobe

    raise RuntimeError(
        "ffprobe was not found. "
        "Install FFmpeg and ensure ffprobe is available in PATH."
    )


# ============================================================
# SIZE HELPERS
# ============================================================

def get_max_upload_bytes() -> int:
    """
    Read MAX_UPLOAD_SIZE_GB safely.

    Defaults to 4 GB.
    """

    raw_value = (
        os.getenv(
            "MAX_UPLOAD_SIZE_GB",
            "4",
        )
        .strip()
    )

    try:
        value_gb = float(raw_value)
    except ValueError:
        value_gb = float(HARD_MAX_UPLOAD_SIZE_GB)
    if not math.isfinite(value_gb) or value_gb <= 0:
        value_gb = float(HARD_MAX_UPLOAD_SIZE_GB)
    return int(min(value_gb, HARD_MAX_UPLOAD_SIZE_GB) * BYTES_PER_GB)


# ============================================================
# BASIC FILE VALIDATION
# ============================================================

def validate_basic_upload(
    path: str | Path,
    *,
    max_upload_bytes: int | None = None,
    expected_extension: str | None = None,
) -> Path:
    """
    Validate filesystem-level properties.

    This function intentionally does NOT trust the extension alone.
    Container validation happens later via ffprobe.
    """

    if (
        max_upload_bytes is not None
        and (not isinstance(max_upload_bytes, int) or isinstance(max_upload_bytes, bool) or max_upload_bytes <= 0)
    ):
        raise ValueError("max_upload_bytes must be greater than zero.")
    candidate = Path(path).expanduser().resolve()

    if not candidate.exists():
        raise MediaValidationError(
            f"Uploaded file does not exist: {candidate}"
        )

    if not candidate.is_file():
        raise MediaValidationError(
            f"Upload path is not a regular file: {candidate}"
        )

    suffix = candidate.suffix.lower()
    if expected_extension is not None:
        suffix = expected_extension.strip().lower()
        if suffix and not suffix.startswith("."):
            suffix = f".{suffix}"

    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(
            sorted(ALLOWED_EXTENSIONS)
        )

        raise MediaValidationError(
            f"Unsupported file extension '{suffix or '[none]'}'. "
            f"Allowed extensions: {allowed}"
        )

    try:
        size_bytes = candidate.stat().st_size

    except OSError as exc:
        raise MediaValidationError(
            f"Could not read uploaded file size: {candidate}"
        ) from exc

    if size_bytes <= 0:
        raise MediaValidationError(
            "Uploaded file is empty."
        )

    limit = (
        max_upload_bytes
        if max_upload_bytes is not None
        else get_max_upload_bytes()
    )

    if size_bytes > limit:
        actual_mb = size_bytes / BYTES_PER_MB
        limit_mb = limit / BYTES_PER_MB

        raise MediaValidationError(
            f"Uploaded file is too large and exceeds the allowed size "
            f"({actual_mb:.2f} MB). "
            f"Maximum allowed size is "
            f"{limit_mb:.2f} MB."
        )

    return candidate


# ============================================================
# FFPARSE
# ============================================================

def _parse_ffprobe_json(
    output: str,
) -> dict[str, Any]:
    """
    Parse ffprobe JSON safely.
    """

    try:
        data = json.loads(output)

    except json.JSONDecodeError as exc:
        raise MediaProbeError(
            "ffprobe returned invalid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise MediaProbeError(
            "ffprobe returned an invalid response structure."
        )

    return data


# ============================================================
# SAFE NUMBER PARSING
# ============================================================

def _parse_non_negative_float(
    value: Any,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    """
    Parse finite non-negative numeric metadata.
    """

    if value is None:
        return default

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(number):
        raise MediaValidationError(
            f"Invalid {field_name} metadata."
        )

    if number < 0:
        raise MediaValidationError(
            f"Invalid negative {field_name} metadata."
        )

    return number


# ============================================================
# FFPBE PROBE
# ============================================================

def probe_media(
    path: str | Path,
) -> dict[str, Any]:
    """
    Inspect actual file bytes using ffprobe.

    No shell=True is used.
    """

    media_path = Path(path).expanduser().resolve()

    try:
        ffprobe = get_ffprobe_path()
    except RuntimeError as exc:
        # A missing or misconfigured probe must fail closed just like an
        # unparseable container; callers should never continue to FFmpeg.
        raise MediaProbeError("Unable to inspect uploaded media.") from exc

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        (
            "format=format_name,duration:"
            "stream=index,codec_type,codec_name,duration"
        ),
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(media_path),
    ]

    logger.info(
        "Probing uploaded media file: %s",
        media_path.name,
    )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(
            "Media validation timed out while inspecting the file."
        ) from exc

    except OSError as exc:
        raise MediaProbeError(
            "Unable to execute ffprobe."
        ) from exc

    if result.returncode != 0:
        error_text = (
            result.stderr.strip()
            or "Unknown ffprobe error."
        )

        logger.warning(
            "ffprobe rejected upload '%s': %s",
            media_path.name,
            error_text[:500],
        )

        raise MediaProbeError(
            "Uploaded file is not a valid supported audio/video container."
        )

    if not result.stdout.strip():
        raise MediaProbeError(
            "ffprobe returned no media metadata."
        )

    return _parse_ffprobe_json(
        result.stdout
    )


# ============================================================
# CONTAINER NORMALIZATION
# ============================================================

def _extract_format_names(
    probe_data: dict[str, Any],
) -> tuple[str, ...]:
    """
    Extract normalized ffprobe format aliases.
    """

    format_data = probe_data.get("format")

    if not isinstance(format_data, dict):
        raise MediaValidationError(
            "Uploaded file has no readable container metadata."
        )

    raw_names = (
        format_data.get("format_name")
        or ""
    )

    names = tuple(
        item.strip().lower()
        for item in str(raw_names).split(",")
        if item.strip()
    )

    if not names:
        raise MediaValidationError(
            "Uploaded file has no recognized container format."
        )

    return names


# ============================================================
# STREAM EXTRACTION
# ============================================================

def _extract_streams(
    probe_data: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Return:

        audio streams
        video streams
    """

    raw_streams = probe_data.get("streams")

    if not isinstance(raw_streams, list):
        raise MediaValidationError(
            "Uploaded file has no readable media streams."
        )

    audio_streams: list[dict[str, Any]] = []
    video_streams: list[dict[str, Any]] = []

    for stream in raw_streams:
        if not isinstance(stream, dict):
            continue

        codec_type = str(
            stream.get("codec_type") or ""
        ).strip().lower()

        if codec_type == "audio":
            audio_streams.append(stream)

        elif codec_type == "video":
            video_streams.append(stream)

    return (
        audio_streams,
        video_streams,
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def validate_media_file(
    path: str | Path,
    *,
    max_upload_bytes: int | None = None,
    max_size_bytes: int | None = None,
    require_audio: bool = True,
    expected_extension: str | None = None,
) -> MediaValidationResult:
    """
    Perform complete media validation.

    Validation order:

        filesystem
        ↓
        extension allowlist
        ↓
        size
        ↓
        ffprobe container inspection
        ↓
        container allowlist
        ↓
        stream validation
        ↓
        duration validation
    """

    if max_upload_bytes is not None and max_size_bytes is not None:
        raise ValueError("Specify only one upload size limit.")
    effective_limit = max_upload_bytes if max_upload_bytes is not None else max_size_bytes
    media_path = validate_basic_upload(
        path,
        max_upload_bytes=effective_limit,
        expected_extension=expected_extension,
    )

    probe_data = probe_media(
        media_path
    )

    format_names = _extract_format_names(
        probe_data
    )

    extension = expected_extension.strip().lower() if expected_extension is not None else media_path.suffix.lower()
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    expected_formats = EXTENSION_TO_FORMATS.get(extension)
    if expected_formats is None:
        # validate_basic_upload normally rejects this first. Keep this guard
        # fail-closed if future callers change validation ordering.
        raise MediaValidationError(
            f"Unsupported file extension '{extension or '[none]'}'."
        )
    if not set(format_names).intersection(expected_formats):
        detected = ", ".join(format_names)

        raise MediaProbeError(
            "Uploaded file extension does not match its detected container. "
            f"Detected format: {detected or '[unknown]'}."
        )

    audio_streams, video_streams = (
        _extract_streams(probe_data)
    )

    if require_audio and not audio_streams:
        raise MediaValidationError(
            "Uploaded media does not contain an audio stream. "
            "This application requires audio for transcription."
        )

    if not audio_streams and not video_streams:
        raise MediaValidationError(
            "Uploaded file does not contain audio or video streams."
        )

    format_data = probe_data.get("format")

    if not isinstance(format_data, dict):
        raise MediaValidationError(
            "Uploaded file contains invalid format metadata."
        )

    duration_seconds = _parse_non_negative_float(
        format_data.get("duration"),
        field_name="duration",
        default=0.0,
    )

    if duration_seconds <= 0:
        # Some files omit format-level duration.
        # Fall back to maximum stream duration.
        stream_durations: list[float] = []

        for stream in (
            audio_streams + video_streams
        ):
            duration = _parse_non_negative_float(
                stream.get("duration"),
                field_name="stream duration",
                default=0.0,
            )

            if duration > 0:
                stream_durations.append(duration)

        duration_seconds = (
            max(stream_durations)
            if stream_durations
            else 0.0
        )

    if duration_seconds <= 0:
        raise MediaValidationError(
            "Uploaded media has an invalid or missing duration."
        )

    if duration_seconds > MAX_DURATION_SECONDS:
        raise MediaValidationError(
            "Uploaded media exceeds the maximum allowed duration of "
            f"{MAX_DURATION_SECONDS / 3600:.2f} hours."
        )

    audio_codec: str | None = None
    video_codec: str | None = None

    if audio_streams:
        codec = audio_streams[0].get(
            "codec_name"
        )

        audio_codec = (
            str(codec).strip().lower()
            if codec
            else None
        )

    if video_streams:
        codec = video_streams[0].get(
            "codec_name"
        )

        video_codec = (
            str(codec).strip().lower()
            if codec
            else None
        )

    result = MediaValidationResult(
        path=media_path,
        extension=extension,
        size_bytes=media_path.stat().st_size,
        duration_seconds=duration_seconds,
        format_names=format_names,
        audio_codec=audio_codec,
        video_codec=video_codec,
        audio_stream_count=len(audio_streams),
        video_stream_count=len(video_streams),
    )

    logger.info(
        (
            "Media validation successful: "
            "file=%s size_bytes=%s duration=%.2fs "
            "formats=%s audio_streams=%s video_streams=%s"
        ),
        result.path.name,
        result.size_bytes,
        result.duration_seconds,
        ",".join(result.format_names),
        result.audio_stream_count,
        result.video_stream_count,
    )

    return result
