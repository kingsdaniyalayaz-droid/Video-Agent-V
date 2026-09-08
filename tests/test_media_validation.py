from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.media_validation import (
    ALLOWED_FORMATS,
    EXTENSION_TO_FORMATS,
    HARD_MAX_UPLOAD_SIZE_GB,
    MediaProbeError,
    MediaValidationError,
    get_max_upload_bytes,
    validate_basic_upload,
    validate_media_file,
)


def test_empty_file_is_rejected(
    tmp_path: Path,
) -> None:

    file_path = (
        tmp_path / "empty.wav"
    )

    file_path.write_bytes(b"")

    with pytest.raises(
        MediaValidationError,
        match="empty",
    ):
        validate_basic_upload(
            file_path
        )


def test_unsupported_extension_is_rejected(
    tmp_path: Path,
) -> None:

    file_path = (
        tmp_path / "malware.exe"
    )

    file_path.write_bytes(
        b"not a real executable"
    )

    with pytest.raises(
        MediaValidationError,
        match="Unsupported file extension",
    ):
        validate_basic_upload(
            file_path
        )


def test_size_limit_is_enforced(
    tmp_path: Path,
) -> None:

    file_path = (
        tmp_path / "large.mp4"
    )

    file_path.write_bytes(
        b"x" * 1024
    )

    with pytest.raises(
        MediaValidationError,
        match="too large",
    ):
        validate_basic_upload(
            file_path,
            max_upload_bytes=100,
        )


def test_fake_media_is_rejected(
    tmp_path: Path,
) -> None:

    file_path = (
        tmp_path / "fake.mp4"
    )

    file_path.write_text(
        "this is not a video"
    )

    with pytest.raises(
        MediaProbeError,
    ):
        validate_media_file(
            file_path
        )


@patch(
    "core.media_validation.probe_media"
)
def test_media_without_audio_is_rejected(
    mock_probe,
    tmp_path: Path,
) -> None:

    file_path = (
        tmp_path / "video.mp4"
    )

    file_path.write_bytes(
        b"placeholder"
    )

    mock_probe.return_value = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "60.0",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
            }
        ],
    }

    with pytest.raises(
        MediaValidationError,
        match="does not contain an audio stream",
    ):
        validate_media_file(
            file_path
        )


@patch(
    "core.media_validation.probe_media"
)
def test_valid_media_is_accepted(
    mock_probe,
    tmp_path: Path,
) -> None:

    file_path = (
        tmp_path / "meeting.mp4"
    )

    file_path.write_bytes(
        b"placeholder media"
    )

    mock_probe.return_value = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "120.5",
        },
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
            {
                "codec_type": "video",
                "codec_name": "h264",
            },
        ],
    }

    result = validate_media_file(
        file_path
    )

    assert result.has_audio is True
    assert result.has_video is True
    assert result.audio_codec == "aac"
    assert result.video_codec == "h264"
    assert result.duration_seconds == 120.5


@patch("core.media_validation.probe_media")
def test_extension_container_mismatch_is_rejected(
    mock_probe,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "fake.mp3"
    file_path.write_bytes(b"placeholder media")
    mock_probe.return_value = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1"},
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
    }

    with pytest.raises(
        MediaProbeError,
        match="extension does not match",
    ):
        validate_media_file(file_path)


@pytest.mark.parametrize("extension", [".mp4", "mp4", ".MP4", "MP4"])
def test_basic_upload_normalizes_expected_extension(
    extension: str,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / ".upload-123.part"
    file_path.write_bytes(b"media")

    validate_basic_upload(
        file_path,
        expected_extension=extension,
    )


@pytest.mark.parametrize("value", ["abc", "NaN", "Infinity", "-5", "0"])
def test_invalid_upload_size_configuration_uses_default(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_GB", value)
    assert get_max_upload_bytes() == 4 * 1024**3


def test_upload_size_configuration_is_hard_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_GB", "999999")
    assert get_max_upload_bytes() == HARD_MAX_UPLOAD_SIZE_GB * 1024**3


@patch("core.media_validation.get_max_upload_bytes", return_value=10)
@patch("core.media_validation.probe_media")
def test_media_validator_uses_default_size_limit(
    mock_probe,
    mock_limit,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "large.mp4"
    file_path.write_bytes(b"x" * 11)

    with pytest.raises(MediaValidationError, match="exceeds"):
        validate_media_file(file_path)
    mock_limit.assert_called_once_with()
    mock_probe.assert_not_called()


@patch("core.media_validation.probe_media")
def test_media_validator_respects_explicit_size_limit(
    mock_probe,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "large.mp4"
    file_path.write_bytes(b"x" * 11)

    with pytest.raises(MediaValidationError, match="exceeds"):
        validate_media_file(file_path, max_size_bytes=10)
    mock_probe.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1])
def test_media_validator_rejects_invalid_explicit_size_limit(
    limit: int,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        validate_media_file(tmp_path / "missing.mp4", max_size_bytes=limit)


@patch("core.media_validation.get_max_upload_bytes", return_value=10)
def test_basic_validator_uses_default_size_limit(
    mock_limit,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "large.mp4"
    file_path.write_bytes(b"x" * 11)

    with pytest.raises(MediaValidationError, match="too large"):
        validate_basic_upload(file_path)
    mock_limit.assert_called_once_with()


@pytest.mark.parametrize("limit", [0, -1])
def test_basic_validator_rejects_invalid_explicit_size_limit(
    limit: int,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        validate_basic_upload(tmp_path / "missing.mp4", max_upload_bytes=limit)


def test_allowed_formats_match_extension_policy() -> None:
    mapped_formats = set().union(*EXTENSION_TO_FORMATS.values())
    assert ALLOWED_FORMATS == mapped_formats