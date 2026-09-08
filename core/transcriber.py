"""
core/transcriber.py

GPU-aware Whisper transcription engine for Video Agent.

Pipeline:

    audio_processor.chunk_audio()
                â†“
        list[Path] chunks
                â†“
        transcribe_all()
                â†“
        Whisper CUDA / CPU
                â†“
        Complete transcript

Features:
    - NVIDIA CUDA detection
    - CPU fallback
    - Whisper model caching
    - FP16 on CUDA
    - FP32 on CPU
    - Urdu / Hindi / English support
    - Optional English translation
    - Chunk-by-chunk transcription
    - GPU memory cleanup
    - Strong input validation
    - Pipeline-compatible Path handling
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import whisper
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small",
).strip()

WHISPER_LANGUAGE = os.getenv(
    "WHISPER_LANGUAGE",
    "en",
).strip().lower()

WHISPER_FP16 = os.getenv(
    "WHISPER_FP16",
    "true",
).strip().lower() in {
    "true",
    "1",
    "yes",
    "on",
}


# ============================================================
# DEVICE
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


# ============================================================
# TYPE ALIASES
# ============================================================

AudioPath = Union[str, Path]


# ============================================================
# SUPPORTED WHISPER MODELS
# ============================================================

SUPPORTED_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "turbo",
}


# ============================================================
# MODEL CACHE
# ============================================================

_model: Optional[whisper.Whisper] = None


# ============================================================
# DEVICE INFORMATION
# ============================================================

def get_device_info() -> dict:
    """
    Return current Whisper device information.
    """

    info = {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model": WHISPER_MODEL,
        "language": WHISPER_LANGUAGE,
        "fp16": (
            WHISPER_FP16
            and DEVICE == "cuda"
        ),
    }

    if torch.cuda.is_available():

        info.update(
            {
                "gpu_name":
                    torch.cuda.get_device_name(0),

                "cuda_version":
                    torch.version.cuda,

                "gpu_count":
                    torch.cuda.device_count(),
            }
        )

    return info


# ============================================================
# PRINT DEVICE INFORMATION
# ============================================================

def print_device_info() -> None:
    """
    Display Whisper runtime information.
    """

    print("\n" + "=" * 70)
    print("                  WHISPER ENGINE")
    print("=" * 70)

    print(
        f"ðŸ–¥ï¸ Device       : {DEVICE}"
    )

    print(
        f"ðŸ“¦ Model        : {WHISPER_MODEL}"
    )

    print(
        f"ðŸŒ Language     : {WHISPER_LANGUAGE}"
    )

    if DEVICE == "cuda":

        print(
            f"ðŸŽ® GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"ðŸ”¥ CUDA         : "
            f"{torch.version.cuda}"
        )

        try:

            props = (
                torch.cuda.get_device_properties(0)
            )

            vram_gb = (
                props.total_memory
                / (1024 ** 3)
            )

            print(
                f"ðŸ’¾ VRAM         : "
                f"{vram_gb:.2f} GB"
            )

        except Exception:
            pass

        print(
            f"âš¡ Precision     : "
            f"{'FP16' if WHISPER_FP16 else 'FP32'}"
        )

    else:

        print(
            "âš ï¸ CUDA          : Unavailable"
        )

        print(
            "âš¡ Precision     : FP32"
        )

    print("=" * 70)


# ============================================================
# INITIAL DEVICE DISPLAY
# ============================================================

print_device_info()


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate_model_name() -> None:
    """
    Validate Whisper model configured in .env.
    """

    if WHISPER_MODEL not in SUPPORTED_MODELS:

        raise ValueError(
            f"Unsupported Whisper model: "
            f"{WHISPER_MODEL}. "
            f"Supported models: "
            f"{', '.join(sorted(SUPPORTED_MODELS))}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model() -> whisper.Whisper:
    """
    Load Whisper model once and cache it.

    Returns:
        Cached Whisper model.
    """

    global _model

    if _model is not None:
        return _model

    validate_model_name()

    print(
        f"\nðŸ“¦ Loading Whisper model: "
        f"{WHISPER_MODEL}"
    )

    print(
        f"ðŸš€ Device: {DEVICE}"
    )

    try:

        _model = whisper.load_model(
            WHISPER_MODEL,
            device=DEVICE,
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to load Whisper model. "
            f"Model={WHISPER_MODEL}, "
            f"Device={DEVICE}. "
            f"Reason: {exc}"
        ) from exc

    print(
        "âœ… Whisper model loaded successfully."
    )

    if DEVICE == "cuda":

        print(
            f"ðŸŽ® GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    return _model


# ============================================================
# RELEASE MODEL
# ============================================================

def unload_model() -> None:
    """
    Release Whisper model and GPU memory.

    Normally this is not required during the pipeline,
    because the same model should be reused for all chunks.
    """

    global _model

    if _model is None:
        return

    print(
        "\nðŸ§¹ Releasing Whisper model..."
    )

    del _model

    _model = None

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print(
        "âœ… Whisper model released."
    )


# ============================================================
# VALIDATE AUDIO FILE
# ============================================================

def validate_audio_file(
    audio_path: AudioPath,
) -> Path:
    """
    Validate a single audio file.

    Args:
        audio_path:
            str or pathlib.Path.

    Returns:
        Validated Path.
    """

    path = Path(audio_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Audio file does not exist: "
            f"{path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Audio path is not a file: "
            f"{path}"
        )

    if path.stat().st_size <= 0:

        raise ValueError(
            f"Audio file is empty: "
            f"{path}"
        )

    return path


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(
    chunks: list[AudioPath],
) -> list[Path]:
    """
    Validate and normalize chunk paths.

    This is intentionally compatible with the output of:

        audio_processor.chunk_audio()

    Example:

        [
            Path("downloads/chunks/.../chunk_0000.wav"),
            Path("downloads/chunks/.../chunk_0001.wav"),
        ]
    """

    if not chunks:

        raise ValueError(
            "No audio chunks were provided."
        )

    validated: list[Path] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        try:

            path = validate_audio_file(
                chunk
            )

        except Exception as exc:

            raise FileNotFoundError(
                f"Invalid audio chunk "
                f"{index}/{len(chunks)}: "
                f"{chunk}"
            ) from exc

        validated.append(path)

    return validated


# ============================================================
# TRANSCRIBE SINGLE CHUNK
# ============================================================

def transcribe_chunk(
    chunk_path: AudioPath,
    translate: bool = False,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe one audio chunk.

    Args:
        chunk_path:
            Actual chunk path returned by audio_processor.

        translate:
            False -> preserve source language.
            True  -> translate speech into English.

        language:
            Optional language override.

            Examples:
                ur
                hi
                en

    Returns:
        Transcript text.
    """

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    path = validate_audio_file(
        chunk_path
    )

    # --------------------------------------------------------
    # Load cached model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    source_language = (
        language.strip().lower()
        if language
        else WHISPER_LANGUAGE
    )

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    task = (
        "translate"
        if translate
        else "transcribe"
    )

    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    use_fp16 = (
        WHISPER_FP16
        and DEVICE == "cuda"
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(
        f"\nðŸŽ™ï¸ Transcribing:"
    )

    print(
        f"   File     : {path.name}"
    )

    print(
        f"   Language : {source_language}"
    )

    print(
        f"   Task     : {task}"
    )

    print(
        f"   Device   : {DEVICE}"
    )

    print(
        f"   FP16     : {use_fp16}"
    )

    # --------------------------------------------------------
    # Whisper options
    # --------------------------------------------------------

    options = {
        "task": task,

        "language": source_language,

        "fp16": use_fp16,

        "temperature": 0,

        "condition_on_previous_text": False,

        "no_speech_threshold": 0.6,

        "compression_ratio_threshold": 2.4,

        "logprob_threshold": -1.0,
    }

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    try:

        result = model.transcribe(
            str(path),
            **options,
        )

    except RuntimeError as exc:

        error_text = str(exc).lower()

        # ----------------------------------------------------
        # CUDA OOM
        # ----------------------------------------------------

        if (
            DEVICE == "cuda"
            and (
                "out of memory" in error_text
                or "cuda out of memory"
                in error_text
            )
        ):

            torch.cuda.empty_cache()

            raise RuntimeError(
                "Whisper GPU ran out of VRAM.\n"
                f"Current model: {WHISPER_MODEL}\n"
                "Recommended for GTX 1050 Ti:\n"
                "  WHISPER_MODEL=small\n"
                "or\n"
                "  WHISPER_MODEL=base"
            ) from exc

        raise RuntimeError(
            f"Whisper failed to transcribe "
            f"{path.name}: {exc}"
        ) from exc

    except Exception as exc:

        raise RuntimeError(
            f"Whisper failed to transcribe "
            f"{path.name}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Extract transcript
    # --------------------------------------------------------

    text = result.get(
        "text",
        "",
    ).strip()

    # --------------------------------------------------------
    # Empty result
    # --------------------------------------------------------

    if not text:

        print(
            "âš ï¸ No speech detected "
            f"in {path.name}"
        )

        return ""

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        f"âœ… Chunk transcription complete "
        f"({len(text):,} characters)"
    )

    return text


# ============================================================
# TRANSCRIBE ALL CHUNKS
# ============================================================

def transcribe_all(
    chunks: list[AudioPath],
    translate: bool = False,
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    Transcribe all chunks returned by audio_processor.

    IMPORTANT:
        This function does NOT construct chunk paths.

    It consumes the exact paths returned by:

        chunk_audio()

    Therefore it works with your current structure:

        downloads/
            chunks/
                ytsHs-KZHVI_16khz/
                    chunk_0000.wav
                    chunk_0001.wav
                    ...

    Args:
        chunks:
            List of actual audio chunk paths.

        translate:
            Translate speech into English.

        language:
            Optional source language override.

    Returns:
        Complete transcript.
    """

    # --------------------------------------------------------
    # Validate chunks
    # --------------------------------------------------------

    validated_chunks = validate_chunks(
        chunks
    )

    total = len(validated_chunks)


    def emit_progress(status: str, current: int = 0, **data) -> None:
        if progress_callback:
            event = {
                "stage": "transcription",
                "status": status,
                "current": current,
                "total": total,
            }
            event.update(data)
            try:
                progress_callback(event)
            except Exception:
                pass

    emit_progress("Transcription started", current=0)

# --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    load_model()

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                 TRANSCRIPTION START")
    print("=" * 70)

    print(
        f"Chunks       : {total}"
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Model        : {WHISPER_MODEL}"
    )

    print(
        f"Language     : "
        f"{language or WHISPER_LANGUAGE}"
    )

    print(
        f"Translation  : "
        f"{translate}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Transcript storage
    # --------------------------------------------------------

    transcripts: list[str] = []

    # --------------------------------------------------------
    # Process chunks sequentially
    # --------------------------------------------------------

    for index, chunk in enumerate(
        validated_chunks,
        start=1,
    ):

        print(
            f"\n[{index}/{total}]"
        )


        chunk_started = __import__("time").monotonic()

        emit_progress(
            f"Transcribing chunk {index}/{total}...",
            current=index,
            chunk_path=str(chunk),
        )

        try:

            text = transcribe_chunk(
                chunk_path=chunk,
                translate=translate,
                language=language,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Transcription failed at "
                f"chunk {index}/{total}.\n"
                f"Chunk: {chunk}"
            ) from exc

        # ----------------------------------------------------
        # Store non-empty transcript
        # ----------------------------------------------------

        if text:

            transcripts.append(
                text
            )

        # ----------------------------------------------------
        # Clear temporary CUDA cache
        # ----------------------------------------------------

        if DEVICE == "cuda":

            torch.cuda.empty_cache()


        emit_progress(
            f"Chunk {index}/{total} completed",
            current=index,
            elapsed=__import__("time").monotonic() - chunk_started,
            characters=len(text),
            chunk_path=str(chunk),
        )

    emit_progress(
        "Transcription completed",
        current=total,
    )

# --------------------------------------------------------
    # Combine transcript
    # --------------------------------------------------------

    full_transcript = "\n\n".join(
        transcripts
    ).strip()

    # --------------------------------------------------------
    # Validate final transcript
    # --------------------------------------------------------

    if not full_transcript:

        raise RuntimeError(
            "Whisper completed successfully, "
            "but no speech was detected "
            "in any audio chunk."
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                TRANSCRIPTION COMPLETE")
    print("=" * 70)

    print(
        f"Chunks processed : {total}"
    )

    print(
        f"Successful chunks: {len(transcripts)}"
    )

    print(
        f"Transcript chars : "
        f"{len(full_transcript):,}"
    )

    print("=" * 70)

    return full_transcript


# ============================================================
# GPU MEMORY STATUS
# ============================================================

def get_gpu_memory() -> dict:
    """
    Return current GPU memory statistics.
    """

    if not torch.cuda.is_available():

        return {
            "available": False,
        }

    allocated = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )

    return {
        "available": True,

        "gpu":
            torch.cuda.get_device_name(0),

        "allocated_gb":
            round(allocated, 2),

        "reserved_gb":
            round(reserved, 2),
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nWhisper transcriber module is ready."
    )

    print(
        "\nRuntime information:"
    )

    info = get_device_info()

    for key, value in info.items():

        print(
            f"  {key}: {value}"
        )

    print(
        "\nGPU memory:"
    )

    memory = get_gpu_memory()

    for key, value in memory.items():

        print(
            f"  {key}: {value}"
        )

    print(
        "\nNo audio was transcribed."
    )
