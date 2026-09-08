"""
Audio Processing Module
=======================

Video_Agent

Pipeline:

    YouTube URL
        |
        v
    yt-dlp
        |
        v
    WebM / M4A
        |
        v
    FFmpeg
        |
        v
    16 kHz Mono PCM WAV
        |
        v
    Audio Chunks
        |
        v
    Whisper

Supported input:
    - YouTube URLs
    - Local audio files
    - Local video files

Designed for:
    - Streamlit
    - CLI
    - Whisper
    - AI Meeting Assistant
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import yt_dlp
from core.media_validation import validate_media_file


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOWNLOAD_DIR = BASE_DIR / "downloads"
CHUNK_DIR = BASE_DIR / "downloads" / "chunks"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


# YouTube cookie file
DEFAULT_COOKIE_FILE = BASE_DIR / "cookies.txt"


# Audio configuration for Whisper
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"


# Default chunk size
DEFAULT_CHUNK_MINUTES = 10
FFMPEG_TIMEOUT_SECONDS = 300


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:
    """
    Simple application logger.

    Works in:
        - Streamlit
        - Terminal
        - VS Code
    """

    print(message)


def separator(title: Optional[str] = None) -> None:
    """
    Print a console separator.
    """

    print()
    print("=" * 70)

    if title:
        print(f"{title:^70}")

    print("=" * 70)


# ============================================================
# PATH UTILITIES
# ============================================================

def get_ffmpeg_path() -> Optional[str]:
    """
    Locate FFmpeg executable.

    Priority:
        1. FFMPEG_PATH environment variable
        2. PATH
        3. common Windows locations
    """

    # --------------------------------------------------------
    # Environment variable
    # --------------------------------------------------------

    env_path = os.getenv("FFMPEG_PATH")

    if env_path:
        path = Path(env_path)

        if path.exists():
            return str(path)

    # --------------------------------------------------------
    # PATH
    # --------------------------------------------------------

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    # --------------------------------------------------------
    # Windows common locations
    # --------------------------------------------------------

    possible_paths = [
        Path(
            r"C:\ffmpeg\bin\ffmpeg.exe"
        ),

        Path(
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        ),

        Path(
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
        ),
    ]

    # Search WinGet installations
    winget_dir = (
        Path.home()
        / "AppData"
        / "Local"
        / "Microsoft"
        / "WinGet"
        / "Packages"
    )

    if winget_dir.exists():

        try:

            possible_paths.extend(
                winget_dir.glob(
                    "Gyan.FFmpeg_*/*/bin/ffmpeg.exe"
                )
            )

        except Exception:
            pass

    for path in possible_paths:

        if path.exists():
            return str(path)

    return None


def require_ffmpeg() -> str:
    """
    Return FFmpeg path or raise a clear error.
    """

    ffmpeg = get_ffmpeg_path()

    if not ffmpeg:

        raise RuntimeError(
            "\n"
            "FFmpeg was not found.\n\n"
            "Install FFmpeg and make sure ffmpeg.exe "
            "is available in PATH.\n"
        )

    return ffmpeg


# ============================================================
# YOUTUBE URL UTILITIES
# ============================================================

def is_youtube_url(url: str) -> bool:
    """
    Determine whether input is a YouTube URL.
    """

    if not url:
        return False

    url = url.strip()

    parsed = urlparse(url)

    hostname = (
        parsed.hostname or ""
    ).lower()

    youtube_hosts = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }

    return hostname in youtube_hosts


def normalize_url(url: str) -> str:
    """
    Normalize common YouTube URL formats.
    """

    url = url.strip()

    if not url:
        raise ValueError(
            "YouTube URL cannot be empty."
        )

    # youtube.com/watch...
    if url.startswith("youtube.com/"):
        return "https://" + url

    # www.youtube.com/watch...
    if url.startswith("www.youtube.com/"):
        return "https://" + url

    # youtu.be/...
    if url.startswith("youtu.be/"):
        return "https://" + url

    return url


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID.
    """

    try:

        parsed = urlparse(
            normalize_url(url)
        )

        hostname = (
            parsed.hostname or ""
        ).lower()

        # youtu.be/<id>
        if hostname in {
            "youtu.be",
            "www.youtu.be",
        }:

            return parsed.path.strip(
                "/"
            ) or None

        # youtube.com/watch?v=<id>
        query = parse_qs(
            parsed.query
        )

        video_id = query.get("v")

        if video_id:
            return video_id[0]

    except Exception:
        pass

    return None


# ============================================================
# COOKIE CONFIGURATION
# ============================================================

def get_cookie_file() -> Optional[Path]:
    """
    Locate YouTube cookies.

    Priority:

        YOUTUBE_COOKIES environment variable
        cookies.txt in project root
    """

    env_cookie = os.getenv(
        "YOUTUBE_COOKIES"
    )

    if env_cookie:

        path = Path(
            env_cookie
        ).expanduser()

        if path.exists():
            return path

    if DEFAULT_COOKIE_FILE.exists():
        return DEFAULT_COOKIE_FILE

    return None


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(
    value: str,
    max_length: int = 100,
) -> str:
    """
    Convert arbitrary title into a safe filename.
    """

    value = value.strip()

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = value.strip(
        " ."
    )

    if not value:
        value = "audio"

    return value[:max_length]


# ============================================================
# YOUTUBE OPTIONS
# ============================================================

def build_youtube_options(
    output_template: str,
) -> dict:
    """
    Build yt-dlp configuration.

    Important:
        - Uses Deno for YouTube JS challenge solving.
        - Uses cookies.txt when available.
        - Does NOT force unstable player clients.
    """

    options = {
        # Audio
        "format": "bestaudio/best",

        # Output
        "outtmpl": output_template,
        "noplaylist": True,

        # YouTube JS runtime
        "js_runtimes": {
            "deno": {}
        },

        # Network / retry
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,

        # Avoid resuming a partially failed YouTube download
        "continuedl": False,

        # Network timeout
        "socket_timeout": 30,

        # Better retry behavior
        "file_access_retries": 3,

        # Logging
        "quiet": False,
        "no_warnings": False,

        # Prevent playlist processing
        "ignoreerrors": False,
    }

    # --------------------------------------------------------
    # Cookies
    # --------------------------------------------------------

    cookie_file = get_cookie_file()

    if cookie_file:

        options["cookiefile"] = str(cookie_file)

        log(
            f"Using YouTube cookie file:\n"
            f"{cookie_file}"
        )

    else:

        log(
            "No YouTube cookie file configured."
        )

    return options

# ============================================================
# GET YOUTUBE INFO
# ============================================================

def get_youtube_info(
    url: str,
) -> dict:
    """
    Retrieve YouTube metadata without downloading.
    """

    url = normalize_url(url)

    if not is_youtube_url(url):

        raise ValueError(
            f"Not a valid YouTube URL:\n{url}"
        )

    video_id = extract_video_id(
        url
    )

    options = build_youtube_options(
        output_template=str(
            DOWNLOAD_DIR
            / "%(id)s.%(ext)s"
        )
    )

    with yt_dlp.YoutubeDL(
        options
    ) as ydl:

        info = ydl.extract_info(
            url,
            download=False,
        )

    if not info:

        raise RuntimeError(
            "Unable to retrieve YouTube information."
        )

    return info


# ============================================================
# DOWNLOAD YOUTUBE AUDIO
# ============================================================
def download_youtube_audio(
    url: str,
    force_download: bool = False,
) -> Path:
    """
    Download YouTube audio and convert it to 16 kHz mono WAV.

    Strategy:
        1. Fetch metadata separately.
        2. Create a fresh yt-dlp session for the actual download.
        3. If HTTP 403 occurs, retry once with a completely fresh extraction.
    """

    url = normalize_url(url)

    if not is_youtube_url(url):

        raise ValueError(
            "Input is not a valid YouTube URL."
        )

    video_id = extract_video_id(url)

    if not video_id:

        raise ValueError(
            "Could not extract YouTube video ID."
        )

    separator(
        "YOUTUBE AUDIO DOWNLOAD"
    )

    log(f"URL: {url}")
    log(f"Video ID: {video_id}")
    log("Format: bestaudio/best")
    log("JS Runtime: Deno")

    ffmpeg = require_ffmpeg()

    log(
        f"FFmpeg: {ffmpeg}"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    downloaded_template = (
        DOWNLOAD_DIR
        / f"{video_id}.%(ext)s"
    )

    wav_path = (
        DOWNLOAD_DIR
        / f"{video_id}_16khz.wav"
    )

    # --------------------------------------------------------
    # Reuse existing WAV
    # --------------------------------------------------------

    if (
        wav_path.exists()
        and wav_path.stat().st_size > 0
        and not force_download
    ):

        log(
            f"Using existing WAV:\n"
            f"{wav_path}"
        )

        return wav_path

    # --------------------------------------------------------
    # Clean old partial downloads
    # --------------------------------------------------------

    for old_file in DOWNLOAD_DIR.glob(
        f"{video_id}.*"
    ):

        if old_file == wav_path:
            continue

        try:
            old_file.unlink()

        except Exception:
            pass

    # --------------------------------------------------------
    # Metadata request
    # --------------------------------------------------------

    metadata_options = build_youtube_options(
        output_template=str(downloaded_template)
    )

    try:

        log(
            "\nFetching YouTube metadata..."
        )

        with yt_dlp.YoutubeDL(
            metadata_options
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=False,
            )

        if not info:

            raise RuntimeError(
                "YouTube metadata could not be retrieved."
            )

        title = info.get(
            "title",
            video_id,
        )

        log(
            f"Title: {title}"
        )

    except yt_dlp.utils.DownloadError as exc:

        raise RuntimeError(
            f"Unable to retrieve YouTube metadata:\n{exc}"
        ) from exc

    # --------------------------------------------------------
    # Download helper
    # --------------------------------------------------------

    def attempt_download() -> None:

        download_options = build_youtube_options(
            output_template=str(
                downloaded_template
            )
        )

        with yt_dlp.YoutubeDL(
            download_options
        ) as ydl:

            # Fresh extraction happens here.
            # This avoids reusing previously extracted
            # temporary format URLs.
            ydl.download(
                [url]
            )

    # --------------------------------------------------------
    # First download attempt
    # --------------------------------------------------------

    try:

        log(
            "\nDownloading audio..."
        )

        attempt_download()

    except yt_dlp.utils.DownloadError as first_exc:

        first_message = str(first_exc)

        # ----------------------------------------------------
        # HTTP 403 -> completely fresh retry
        # ----------------------------------------------------

        if "403" in first_message:

            log(
                "\nHTTP 403 received."
            )

            log(
                "Refreshing YouTube extraction "
                "and retrying download..."
            )

            # Remove partial files before retry
            for old_file in DOWNLOAD_DIR.glob(
                f"{video_id}.*"
            ):

                if old_file == wav_path:
                    continue

                try:
                    old_file.unlink()

                except Exception:
                    pass

            try:

                attempt_download()

            except yt_dlp.utils.DownloadError as second_exc:

                message = str(second_exc)

                raise RuntimeError(
                    "\nYouTube returned HTTP 403 "
                    "after a fresh retry.\n\n"
                    "Check the following:\n"
                    "1. Update yt-dlp\n"
                    "2. Re-export cookies.txt from a browser "
                    "where YouTube is logged in\n"
                    "3. Make sure cookies.txt is Netscape format\n"
                    "4. Verify that Deno is available\n\n"
                    f"Original error:\n{message}"
                ) from second_exc

        else:

            raise RuntimeError(
                f"\nYouTube download failed:\n"
                f"{first_message}"
            ) from first_exc

    except Exception as exc:

        raise RuntimeError(
            f"\nFailed to download YouTube audio:\n"
            f"{exc}"
        ) from exc

    # --------------------------------------------------------
    # Locate downloaded file
    # --------------------------------------------------------

    downloaded_path = find_downloaded_file(
        video_id
    )

    if not downloaded_path:

        raise FileNotFoundError(
            "yt-dlp completed but the downloaded "
            "audio file could not be located."
        )

    log(
        "\nDownloaded:"
    )

    log(
        str(downloaded_path)
    )

    # --------------------------------------------------------
    # Convert to WAV
    # --------------------------------------------------------

    return convert_to_wav(
        downloaded_path,
        output_path=wav_path,
        force=True,
    )

# ============================================================
# FIND DOWNLOADED FILE
# ============================================================

def find_downloaded_file(
    video_id: str,
) -> Optional[Path]:
    """
    Find audio downloaded by yt-dlp.
    """

    candidates = []

    for path in DOWNLOAD_DIR.glob(
        f"{video_id}.*"
    ):

        if path.is_file():

            # Skip WAV generated by us
            if path.suffix.lower() == ".wav":
                continue

            candidates.append(
                path
            )

    if not candidates:
        return None

    # Prefer common audio formats
    priority = {
        ".webm": 0,
        ".m4a": 1,
        ".mp4": 2,
        ".opus": 3,
        ".mp3": 4,
    }

    candidates.sort(
        key=lambda p: priority.get(
            p.suffix.lower(),
            99,
        )
    )

    return candidates[0]


# ============================================================
# LOCAL FILE → WAV
# ============================================================

def convert_to_wav(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """
    Convert audio/video into:

        PCM S16LE
        Mono
        16 kHz
        WAV

    FFmpeg is used directly instead of pydub.

    This avoids pydub's >4 GB WAV limitation.
    """

    input_path = Path(
        input_path
    ).expanduser().resolve()

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file does not exist:\n"
            f"{input_path}"
        )

    ffmpeg = require_ffmpeg()

    if output_path is None:

        output_path = (
            input_path.parent
            / f"{input_path.stem}_16khz.wav"
        )

    output_path = Path(
        output_path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator(
        "AUDIO CONVERSION"
    )

    log(
        f"Input : {input_path}"
    )

    log(
        f"Output: {output_path}"
    )

    log(
        "Format: WAV / PCM S16LE / Mono / 16 kHz"
    )

    # --------------------------------------------------------
    # Existing output
    # --------------------------------------------------------

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not force
    ):

        log(
            "\nExisting WAV found. "
            "Skipping conversion."
        )

        return output_path

    # --------------------------------------------------------
    # FFmpeg command
    # --------------------------------------------------------

    command = [

        ffmpeg,

        "-y",

        "-hide_banner",

        "-loglevel",
        "error",

        "-i",
        str(input_path),

        # Mono
        "-ac",
        str(CHANNELS),

        # 16 kHz
        "-ar",
        str(SAMPLE_RATE),

        # PCM signed 16-bit
        "-c:a",
        AUDIO_CODEC,

        str(output_path),
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:

            error = (
                result.stderr.strip()
                or "Unknown FFmpeg error."
            )

            raise RuntimeError(
                error
            )

        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):

            raise RuntimeError(
                "FFmpeg completed but WAV file "
                "was not created."
            )

        size_mb = (
            output_path.stat().st_size
            / (
                1024 * 1024
            )
        )

        log(
            "\nAudio conversion successful."
        )

        log(
            f"WAV file: {output_path}"
        )

        log(
            f"Size: {size_mb:.2f} MB"
        )

        return output_path

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("FFmpeg conversion timed out.") from exc

    except FileNotFoundError as exc:

        raise RuntimeError(
            "FFmpeg executable could not be started."
        ) from exc

    except Exception as exc:

        # Remove incomplete output
        if output_path.exists():

            try:
                output_path.unlink()
            except Exception:
                pass

        raise RuntimeError(
            f"Audio conversion failed:\n"
            f"{exc}"
        ) from exc


# ============================================================
# AUDIO CHUNKING
# ============================================================

def chunk_audio(
    wav_path: str | Path,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    output_dir: Optional[str | Path] = None,
) -> list[str]:
    """
    Split WAV into fixed-duration chunks.

    Uses FFmpeg directly.

    Example:

        60-minute audio
        chunk_minutes=10

        → 6 WAV files
    """

    wav_path = Path(
        wav_path
    ).expanduser().resolve()

    if not wav_path.exists():

        raise FileNotFoundError(
            f"WAV file not found:\n"
            f"{wav_path}"
        )

    if chunk_minutes <= 0:

        raise ValueError(
            "chunk_minutes must be greater than 0."
        )

    ffmpeg = require_ffmpeg()

    if output_dir is None:

        output_dir = (
            CHUNK_DIR
            / wav_path.stem
        )

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator(
        "AUDIO CHUNKING"
    )

    log(
        f"Input: {wav_path}"
    )

    log(
        f"Chunk duration: "
        f"{chunk_minutes} minutes"
    )

    # --------------------------------------------------------
    # FFmpeg chunk command
    # --------------------------------------------------------

    operation_dir = output_dir / f".chunk-operation-{uuid.uuid4().hex}"
    operation_dir.mkdir()
    output_pattern = (
        operation_dir
        / "chunk_%04d.wav"
    )

    command = [

        ffmpeg,

        "-y",

        "-hide_banner",

        "-loglevel",
        "error",

        "-i",
        str(wav_path),

        # Audio format
        "-ac",
        str(CHANNELS),

        "-ar",
        str(SAMPLE_RATE),

        "-c:a",
        AUDIO_CODEC,

        # Segment
        "-f",
        "segment",

        "-segment_time",
        str(
            chunk_minutes * 60
        ),

        "-reset_timestamps",
        "1",

        str(output_pattern),
    ]

    created_chunks: list[Path] = []

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:

            raise RuntimeError(
                result.stderr.strip()
            )

        generated_chunks = sorted(operation_dir.glob("chunk_*.wav"))

        if not generated_chunks:

            raise RuntimeError(
                "FFmpeg did not create any audio chunks."
            )

        chunks: list[Path] = []
        for generated_chunk in generated_chunks:
            final_chunk = output_dir / generated_chunk.name
            if final_chunk.exists():
                raise RuntimeError(f"Refusing to overwrite existing chunk: {final_chunk.name}")
            generated_chunk.replace(final_chunk)
            chunks.append(final_chunk)
            created_chunks.append(final_chunk)
        operation_dir.rmdir()

        log(
            f"\nCreated {len(chunks)} chunk(s)."
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            size_mb = (
                chunk.stat().st_size
                / (
                    1024 * 1024
                )
            )

            log(
                f"Chunk {index}: "
                f"{chunk.name} "
                f"({size_mb:.2f} MB)"
            )

        return [
            str(chunk)
            for chunk in chunks
        ]

    except Exception as exc:
        for created_chunk in created_chunks:
            created_chunk.unlink(missing_ok=True)
        if operation_dir.exists():
            for generated_chunk in operation_dir.glob("chunk_*.wav"):
                generated_chunk.unlink(missing_ok=True)
            operation_dir.rmdir()

        raise RuntimeError(
            f"Audio chunking failed:\n"
            f"{exc}"
        ) from exc


# ============================================================
# LOCAL FILE PROCESSING
# ============================================================

def process_local_file(
    source: str | Path,
) -> Path:
    """
    Convert local audio/video file to WAV.
    """

    source = Path(
        source
    ).expanduser().resolve()

    if not source.exists():

        raise FileNotFoundError(
            f"Local file does not exist:\n"
            f"{source}"
        )

    if not source.is_file():

        raise ValueError(
            f"Input is not a file:\n"
            f"{source}"
        )

    validate_media_file(source)

    log(
        "Detected local file."
    )

    log(
        "Converting to WAV..."
    )

    return convert_to_wav(
        source
    )


# ============================================================
# MAIN INPUT PROCESSOR
# ============================================================

def process_input(
    source: str,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
) -> list[str]:
    """
    Main entry point for the application.

    Supports:

        YouTube URL
        Local audio
        Local video

    Returns:

        list[str]
            WAV chunk paths
    """

    if not source:

        raise ValueError(
            "Input source cannot be empty."
        )

    source = source.strip()

    if not source:

        raise ValueError(
            "Input source cannot be empty."
        )

    # --------------------------------------------------------
    # Normalize URL
    # --------------------------------------------------------

    if (
        source.startswith(
            "youtube.com/"
        )
        or source.startswith(
            "www.youtube.com/"
        )
        or source.startswith(
            "youtu.be/"
        )
    ):

        source = normalize_url(
            source
        )

    # --------------------------------------------------------
    # YouTube
    # --------------------------------------------------------

    if is_youtube_url(source):

        log(
            "Detected YouTube URL."
        )

        wav_path = (
            download_youtube_audio(
                source
            )
        )

    # --------------------------------------------------------
    # Local file
    # --------------------------------------------------------

    else:

        wav_path = (
            process_local_file(
                source
            )
        )

    # --------------------------------------------------------
    # Chunk audio
    # --------------------------------------------------------

    log(
        f"\nChunking audio into "
        f"{chunk_minutes}-minute pieces..."
    )

    chunks = chunk_audio(
        wav_path,
        chunk_minutes=chunk_minutes,
    )

    log(
        f"\nAudio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks


# ============================================================
# CLEANUP
# ============================================================

def cleanup_file(
    path: str | Path,
) -> None:
    """
    Safely remove a file.
    """

    path = Path(path)

    if not path.exists():
        return

    try:

        path.unlink()

        log(
            f"Deleted: {path}"
        )

    except Exception as exc:

        log(
            f"Could not delete {path}: {exc}"
        )


def cleanup_chunks(
    chunks: str | Path | list[str] | tuple[str, ...] | set[str] | None = None,
) -> None:
    """
    Safely delete generated audio chunks.

    Accepted inputs:

        1. A single chunk path
        2. A list/tuple/set of chunk paths
        3. A directory containing generated chunks
        4. None -> use the default CHUNK_DIR

    This function is intentionally backward compatible with the old
    directory-based API while also supporting the pipeline contract,
    where main.py passes the actual list of temporary chunk paths.

    Cleanup failures are isolated per file and never propagated.
    """

    if chunks is None:
        target = CHUNK_DIR

    else:
        target = chunks

    # --------------------------------------------------------
    # Multiple explicit chunk paths
    # --------------------------------------------------------

    if isinstance(
        target,
        (list, tuple, set),
    ):

        for item in target:

            try:
                path = Path(item)

                if (
                    path.exists()
                    and path.is_file()
                ):

                    cleanup_file(path)

            except Exception as exc:

                log(
                    f"Could not clean chunk "
                    f"{item}: {exc}"
                )

        return

    # --------------------------------------------------------
    # Single path
    # --------------------------------------------------------

    try:
        path = Path(target)

    except Exception as exc:

        log(
            f"Could not resolve cleanup target: {exc}"
        )

        return

    if not path.exists():
        return

    # --------------------------------------------------------
    # Single file
    # --------------------------------------------------------

    if path.is_file():

        cleanup_file(path)
        return

    # --------------------------------------------------------
    # Directory
    # --------------------------------------------------------

    if path.is_dir():

        for file in path.rglob("*.wav"):

            cleanup_file(file)


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    separator(
        "AUDIO PROCESSOR TEST"
    )

    print(
        f"Project directory:\n{BASE_DIR}"
    )

    print(
        f"\nDownloads:\n{DOWNLOAD_DIR}"
    )

    print(
        f"\nChunks:\n{CHUNK_DIR}"
    )

    ffmpeg = get_ffmpeg_path()

    print(
        f"\nFFmpeg:\n{ffmpeg or 'NOT FOUND'}"
    )

    cookies = get_cookie_file()

    print(
        f"\nYouTube cookies:\n"
        f"{cookies or 'NOT FOUND'}"
    )

    print(
        "\nAudio processor is ready."
    )
