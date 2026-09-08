from __future__ import annotations

# Windows asyncio cleanup compatibility.
# Prevent noisy ProactorBasePipeTransport / WinError 10054 tracebacks
# when a peer closes an HTTP socket while Streamlit is shutting it down.
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any
from xml.sax.saxutils import escape

import streamlit as st
import transformers
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

from core.database import delete_video, get_video, list_videos, video_exists
from core.vector_store import delete_vector_store
from core.prompt_generator import generate_prompt
from core.roman_urdu_translator import translate_to_roman_urdu
from core.translation_cache import (
    delete_translation_by_id,
    get_cached_translation,
    get_saved_translation_by_hash,
    get_saved_translation_by_id,
    get_transcript_hash,
    get_translation_count,
    list_saved_translations,
    save_translation,
)
from core.translation_exporter import (
    build_translation_excel,
    build_translation_pdf,
    build_translation_txt,
)
from core.video_library_analytics import (
    calculate_completion_percentage,
    calculate_library_analytics,
    format_duration,
    get_video_content_status,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ARROW_LOG_LEVEL"] = "ERROR"
transformers.logging.set_verbosity_error()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


_cache_resource = getattr(st, "cache_resource", None)
if _cache_resource is None:
    _cache_resource = getattr(st, "experimental_singleton", None)

def _resource_cache(**kwargs):
    """Compatibility decorator for older Streamlit installations."""
    if _cache_resource is None:
        return lambda function: function
    return _cache_resource(**kwargs)


@_resource_cache(show_spinner=False)
def get_backend():
    """Load main.py only when the backend is actually needed.

    main.py currently initializes the Whisper engine during import.
    Lazy-loading prevents normal Streamlit startup/reruns from initializing
    Whisper before the user actually needs the processing backend.
    """
    import importlib

    return importlib.import_module("main")


@_resource_cache(show_spinner=False)
def get_cached_rag_chain(video_id: str, source_type: str):
    """Load an existing per-video RAG chain without importing main.py.

    This is intentionally lazy: cached videos do not load Chroma/embeddings
    until the user actually asks a question in AI Chat.
    """
    from core.rag_engine import load_rag_chain

    return load_rag_chain(
        video_id=video_id,
        source_type=source_type,
    )


def ask_rag_question(rag_chain: Any, question: str) -> str:
    """Ask a question directly through the RAG engine.

    Do not import main.py here. main.py owns the expensive Whisper pipeline,
    so importing it from the chat path can initialize Whisper unnecessarily.
    """
    from core.rag_engine import ask_question

    return ask_question(
        rag_chain=rag_chain,
        question=question,
    )


# -----------------------------------------------------------------------------
# Page configuration and styling
# -----------------------------------------------------------------------------

SUPPORTED_FILE_TYPES = ["mp4", "mov", "avi", "mkv", "wav", "mp3", "m4a"]
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Urdu": "ur",
}


def configure_page() -> None:
    """Configure Streamlit's page-level settings."""
    st.set_page_config(
        page_title="AI Meeting Intelligence",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    """Apply the premium dark workspace design system and responsive rules."""
    st.markdown("""
    <style>
    :root { --bg:#070b16; --surface:rgba(19,28,50,.84); --border:rgba(148,163,204,.17); --strong:rgba(111,139,220,.36); --text:#f1f5ff; --muted:#9aa8c5; --blue:#6bd7ff; --green:#74e3b4; }
    .stApp { background:radial-gradient(circle at 78% -10%,rgba(111,81,220,.20),transparent 30rem),radial-gradient(circle at -10% 24%,rgba(43,160,205,.12),transparent 28rem),var(--bg); color:var(--text); }
    .block-container { max-width:1440px; padding:2rem clamp(1rem,3vw,3rem) 4rem; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(11,16,31,.98),rgba(9,13,26,.96)); border-right:1px solid var(--border); }
    .app-header { align-items:flex-start; display:flex; gap:1.5rem; justify-content:space-between; margin-bottom:1.7rem; }
    .eyebrow { color:var(--blue); font-size:.70rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
    .app-title { color:var(--text); font-size:clamp(1.9rem,4vw,3.25rem); font-weight:780; letter-spacing:-.055em; line-height:1.05; margin:0; overflow-wrap:anywhere; }
    .app-subtitle { color:var(--muted); font-size:.98rem; margin:.72rem 0 0; }
    .header-context { background:rgba(19,28,51,.72); border:1px solid var(--border); border-radius:999px; color:var(--muted); max-width:min(34rem,42vw); padding:.62rem .9rem; overflow-wrap:anywhere; text-align:right; }
    .status-badge { color:var(--green); font-weight:700; white-space:nowrap; }
    .kpi-card,[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--border); border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,.24); }
    .kpi-card { background:linear-gradient(145deg,rgba(25,35,64,.92),rgba(12,18,35,.9)); min-height:104px; padding:1rem; overflow-wrap:anywhere; }
    .kpi-label { color:var(--muted); font-size:.70rem; letter-spacing:.08em; text-transform:uppercase; }
    .kpi-value { color:var(--text); font-size:clamp(1.05rem,2vw,1.45rem); font-weight:750; line-height:1.22; margin-top:.55rem; overflow-wrap:anywhere; }
    .section-title { color:var(--text); font-size:clamp(1.05rem,2vw,1.3rem); font-weight:750; margin:1.6rem 0 .72rem; }
    .result-panel,.item-card { background:var(--surface); border:1px solid var(--border); border-radius:14px; overflow-wrap:anywhere; }
    .result-panel { padding:1.25rem 1.4rem; }
    .item-card { margin:.65rem 0; padding:.9rem 1rem; }
    .item-number { color:var(--blue); font-size:.70rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .stButton > button,.stDownloadButton > button { border:1px solid var(--border); border-radius:10px; min-height:2.55rem; }
    .stButton > button:hover,.stDownloadButton > button:hover { border-color:var(--strong); }
    .stTextInput input,.stTextArea textarea,.stSelectbox [data-baseweb="select"] > div { background:rgba(10,16,31,.72); border-color:var(--border); border-radius:10px; }
    div[data-testid="stFileUploader"] { background:rgba(14,21,39,.52); border:1px dashed var(--strong); border-radius:14px; padding:.35rem; }
    div[data-testid="stChatMessage"] { background:rgba(19,28,50,.68); border:1px solid var(--border); border-radius:14px; }
    [data-testid="stTabs"] [role="tablist"] { gap:.25rem; border-bottom:1px solid var(--border); overflow-x:auto; }
    [data-testid="stTabs"] button[role="tab"] { color:var(--muted); padding:.75rem .8rem; white-space:nowrap; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--blue); }
    [data-testid="stMetric"] { background:rgba(18,27,49,.72); border:1px solid var(--border); border-radius:14px; padding:.85rem; min-height:5.2rem; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { overflow-wrap:anywhere; }
    h1,h2,h3,h4,p,label,[data-testid="stCaptionContainer"] { overflow-wrap:anywhere; }
    .library-card-title { color:var(--text); font-size:1.05rem; font-weight:760; line-height:1.25; margin:.1rem 0 .55rem; overflow-wrap:anywhere; }
    .library-card-meta { color:var(--muted); font-size:.78rem; line-height:1.55; }
    .library-active { color:var(--green); font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .library-action-row button { min-height:2.35rem; }
    .status-pill { border:1px solid var(--border); border-radius:999px; display:inline-block; font-size:.73rem; font-weight:700; padding:.25rem .55rem; }
    .status-pill.ready,.status-pill.completed { border-color:rgba(116,227,180,.35); color:var(--green); }
    .status-pill.processing { border-color:rgba(107,215,255,.35); color:var(--blue); }
    .status-pill.cached { border-color:rgba(169,148,255,.35); color:#c8bbff; }
    .status-pill.waiting,.status-pill.unavailable { color:var(--muted); }
    .status-pill.error { border-color:rgba(255,154,168,.35); color:#ff9aa8; }
    .analytics-card [data-testid="stMetric"] { background:rgba(18,27,49,.72); border:1px solid var(--border); border-radius:14px; padding:.85rem; min-height:5.2rem; }
    .analytics-card [data-testid="stMetricValue"] { overflow-wrap:anywhere; font-size:clamp(1rem,2vw,1.45rem); }
    .analytics-card [data-testid="stMetricLabel"] { overflow-wrap:anywhere; }
    .sidebar-brand { color:var(--text); font-size:1.1rem; font-weight:780; letter-spacing:-.02em; }
    .sidebar-kicker { color:var(--blue); font-size:.68rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    @media (max-width:760px) { .analytics-card [data-testid="stMetric"] { min-height:auto; } }
    @media (max-width:1050px) { .block-container{padding-left:1.15rem;padding-right:1.15rem;} .header-context{max-width:38vw;} }
    @media (max-width:760px) { .app-header{flex-direction:column;gap:.8rem;} .header-context{max-width:100%;text-align:left;width:100%;} }
    @media (max-width:520px) { .block-container{padding:1.25rem .8rem 3rem;} .kpi-card{min-height:auto;} }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# State and small presentation helpers
# -----------------------------------------------------------------------------


def initialize_state() -> None:
    defaults = {
        "result": None,
        "processed": False,
        "chat_history": [],
        "last_source": None,
        "last_language": None,
        "complete_pdf_data": None,
        "rag_video_id": None,
        "app_page": "ðŸ  Home",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "roman_urdu_transcript" not in st.session_state:
        st.session_state.roman_urdu_transcript = None

    if "roman_urdu_source_id" not in st.session_state:
        st.session_state.roman_urdu_source_id = None
    if "active_source_identity" not in st.session_state:
        st.session_state.active_source_identity = None
    if "chat_history_source_id" not in st.session_state:
        st.session_state.chat_history_source_id = None
    active_defaults = {
        "active_translation_id": None,
        "active_original_transcript": None,
        "active_roman_urdu_translation": None,
        "active_translation_model": None,
        "active_translation_source": None,
    }
    for key, value in active_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_result_source_name(result: dict[str, Any]) -> str:
    """Return the best available identity for a processed result."""
    return str(
        result.get("source_name")
        or result.get("title")
        or result.get("source")
        or "Untitled Source"
    )


def get_selected_active_video_state(
    selected: dict[str, Any] | None,
    active_result: dict[str, Any] | None,
) -> tuple[str, str, bool]:
    """Return selected ID, active label, and whether selection is truly active."""
    selected = selected or {}
    active_result = active_result or {}
    selected_id = str(selected.get("video_id") or "")
    metadata = active_result.get("metadata") or {}
    active_id = str(
        metadata.get("video_id")
        or active_result.get("video_id")
        or active_result.get("meeting_id")
        or ""
    )
    active_name = get_result_source_name(active_result) if active_result else "No active video"
    return selected_id, active_name, bool(selected_id and active_id and selected_id == active_id)


def get_transcript_source_id(transcript: str) -> str:
    """Return a deterministic session/source identity for transcript text."""
    return hashlib.sha256(str(transcript or "").encode("utf-8")).hexdigest()


def get_result_identity(result: dict[str, Any] | None) -> str | None:
    """Return a stable source identity for chat and active-content state."""
    result = result or {}
    metadata = result.get("metadata") or {}
    source_type = str(result.get("source_type") or metadata.get("source_type") or "source")
    source_id = (
        result.get("video_id")
        or metadata.get("video_id")
        or result.get("meeting_id")
        or metadata.get("meeting_id")
        or result.get("source_id")
        or metadata.get("source_id")
    )
    if source_id:
        return f"{source_type}:id:{source_id}"
    source = result.get("source")
    if source:
        return f"{source_type}:source:{source}"
    transcript = str(result.get("transcript") or "")
    if transcript:
        return f"{source_type}:transcript:{hashlib.sha256(transcript.encode('utf-8')).hexdigest()}"
    return None


class ExportStateError(ValueError):
    """Raised when export data is missing or belongs to another source."""


def validate_export_state(
    result: dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Return export data only when it matches the active source exactly.

    The active session fields are the sole export authority. Legacy
    ``roman_urdu_transcript`` state is intentionally not consulted.
    """
    result = result or {}
    metadata = result.get("metadata") or {}
    current_identity = get_result_identity(result)
    active_identity = st.session_state.get("active_source_identity") or current_identity
    original = str(st.session_state.get("active_original_transcript") or "").strip()
    translation = str(st.session_state.get("active_roman_urdu_translation") or "").strip()
    result_transcript = str(result.get("transcript") or "").strip()

    if not original:
        raise ExportStateError("Export unavailable: no active original transcript is available.")
    if not translation:
        raise ExportStateError("Export unavailable: no active Roman Urdu translation is available.")
    if not current_identity or not active_identity or current_identity != active_identity:
        raise ExportStateError(
            "Export unavailable: the current Roman Urdu translation belongs to a different source."
        )
    if result_transcript != original:
        raise ExportStateError(
            "Export unavailable: the active transcript and translation do not belong to the same source."
        )

    # If a persisted source ID exists, require it to be represented by the
    # active result identity; this prevents title/filename-only collisions.
    result_source_id = (
        result.get("video_id")
        or result.get("meeting_id")
        or result.get("source_id")
        or metadata.get("video_id")
        or metadata.get("meeting_id")
        or metadata.get("source_id")
    )
    if result_source_id and not current_identity.endswith(f":id:{result_source_id}"):
        raise ExportStateError(
            "Export unavailable: the current translation source identity could not be verified."
        )
    return original, translation, current_identity


def is_translation_only_result(result: dict[str, Any] | None) -> bool:
    """Identify explicit Saved History results that intentionally lack analysis."""
    result = result or {}
    metadata = result.get("metadata") or {}
    return str(result.get("source_type") or "") == "saved_translation" or (
        metadata.get("active_translation_id") is not None
        and not metadata.get("video_id")
    )


def save_translation_with_source_identity(
    transcript: str,
    roman_urdu_translation: str,
    model_name: str,
    source_name: str,
    source_id: str | None = None,
) -> Any:
    """Persist translation identity when the existing cache API supports it."""
    import inspect

    kwargs = {"model_name": model_name, "source_name": source_name}
    if source_id:
        parameters = inspect.signature(save_translation).parameters
        if "video_id" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            kwargs["video_id"] = source_id
        elif "source_id" in parameters:
            kwargs["source_id"] = source_id
    return save_translation(transcript, roman_urdu_translation, **kwargs)


def saved_translation_matches_result(record: dict[str, Any] | None, result: dict[str, Any] | None) -> bool:
    """Allow a saved translation only when its source identity matches the active result."""
    record = record or {}
    result = result or {}
    result_metadata = result.get("metadata") or {}
    record_metadata = record.get("metadata") or {}
    record_source_id = (
        record.get("video_id")
        or record.get("meeting_id")
        or record.get("source_id")
        or record_metadata.get("video_id")
        or record_metadata.get("meeting_id")
        or record_metadata.get("source_id")
    )
    result_source_id = (
        result.get("video_id")
        or result.get("meeting_id")
        or result_metadata.get("video_id")
        or result_metadata.get("meeting_id")
    )
    if record_source_id and result_source_id:
        return str(record_source_id) == str(result_source_id)

    record_transcript = str(record.get("original_transcript") or record.get("transcript") or "")
    result_transcript = str(result.get("transcript") or "")
    record_hash = str(record.get("transcript_hash") or record_metadata.get("transcript_hash") or "")
    result_hash = hashlib.sha256(result_transcript.encode("utf-8")).hexdigest() if result_transcript else ""
    if record_hash and result_hash:
        return record_hash == result_hash
    return bool(record_transcript and record_transcript == result_transcript)


def can_reuse_cached_rag(
    cached_video_id: Any,
    cached_transcript: Any,
    active_result: dict[str, Any] | None,
    active_rag_video_id: Any,
    active_source_identity: str | None,
) -> bool:
    """Confirm cached-RAG reuse is for the same authoritative active video."""
    active_result = active_result or {}
    active_metadata = active_result.get("metadata") or {}
    cached_id = str(cached_video_id or "")
    cached_text = str(cached_transcript or "")
    cached_identity = get_result_identity({
        "source_type": "youtube",
        "metadata": {"video_id": cached_id},
        "transcript": cached_text,
    })
    resolved_active_identity = active_source_identity or get_result_identity(active_result)
    return bool(
        cached_id
        and str(active_rag_video_id or "") == cached_id
        and resolved_active_identity == cached_identity
        and str(active_result.get("transcript") or "") == cached_text
        and str(active_metadata.get("video_id") or "") == cached_id
        and active_result.get("rag_chain") is not None
    )


def set_active_translation(
    original_transcript: str,
    roman_urdu_translation: str,
    record: dict[str, Any] | None = None,
    source: str | None = "current uploaded video",
    clear_downstream: bool = False,
) -> None:
    """Atomically update the one active transcript/translation selection."""
    st.session_state.active_translation_id = record.get("id") if record else None
    st.session_state.active_original_transcript = original_transcript
    st.session_state.active_roman_urdu_translation = roman_urdu_translation
    st.session_state.active_translation_model = record.get("model_name") if record else None
    st.session_state.active_translation_source = source
    if clear_downstream:
        st.session_state.complete_pdf_data = None
        if not st.session_state.pop("_preserve_chat_on_same_source", False):
            st.session_state.chat_history = []
        st.session_state.generated_prompt = ""
        st.session_state.pop("prompt_studio_source_id", None)


def clear_active_translation() -> None:
    """Clear active translation state when the current source changes."""
    set_active_translation("", "", source=None, clear_downstream=True)
    st.session_state.active_translation_id = None
    st.session_state.active_original_transcript = None
    st.session_state.active_roman_urdu_translation = None
    st.session_state.active_translation_model = None
    st.session_state.active_translation_source = None


def set_active_processed_result(
    normalized_result: dict[str, Any],
    display_source: str,
) -> None:
    """Persist a successful pipeline result and synchronize active content."""
    transcript = str(normalized_result.get("transcript") or "")
    previous_transcript = st.session_state.get("active_original_transcript")
    previous_result = st.session_state.get("result") or {}
    previous_identity = st.session_state.get("active_source_identity") or get_result_identity(previous_result)
    active_identity = get_result_identity(normalized_result)
    same_source = bool(previous_identity and active_identity and previous_identity == active_identity)
    st.session_state["_preserve_chat_on_same_source"] = same_source

    st.session_state.result = normalized_result
    st.session_state.processed = True
    st.session_state.last_source = display_source
    st.session_state.last_language = normalized_result.get("language")
    if not same_source:
        st.session_state.chat_history = []
        st.session_state.pop("saved_translation_selector", None)
    st.session_state.chat_history_source_id = active_identity
    st.session_state.active_source_identity = active_identity
    st.session_state.complete_pdf_data = None

    if not same_source or previous_transcript != transcript:
        st.session_state.active_translation_id = None
        st.session_state.active_roman_urdu_translation = None
        st.session_state.active_translation_model = None
        st.session_state.active_translation_source = None
        st.session_state.roman_urdu_transcript = None
        st.session_state.roman_urdu_source_id = None

    st.session_state.active_original_transcript = transcript or None
    if not st.session_state.get("active_translation_source"):
        st.session_state.active_translation_source = display_source


def active_result_view(result: dict[str, Any]) -> dict[str, Any]:
    """Overlay only explicit Saved History translations; preserve normal analysis."""
    active_original = st.session_state.get("active_original_transcript")
    active_translation = st.session_state.get("active_roman_urdu_translation")
    if not active_original or not is_translation_only_result(result):
        return result

    active_result = dict(result)
    result_transcript = str(result.get("transcript") or "")
    if result_transcript != active_original:
        for key, empty_value in {
            "summary": "",
            "executive_summary": "",
            "actions": [],
            "decisions": [],
            "questions": [],
            "rag_chain": None,
        }.items():
            active_result[key] = empty_value
        active_result["metadata"] = dict(result.get("metadata") or {})
        active_result["metadata"]["active_translation_id"] = st.session_state.get("active_translation_id")
        active_result["metadata"].pop("video_id", None)
        active_result["source_type"] = "saved_translation"
        active_result["source"] = active_original
    active_result["transcript"] = active_original
    active_result["translated_transcript"] = active_translation or ""
    return active_result


def reset_session() -> None:
    # Remove an uncommitted/pending upload when starting a new session.
    # _cleanup_upload_file() protects an active pipeline source and swallows
    # filesystem failures, so reset_session remains safe.
    pending_upload_path = st.session_state.get("pending_upload_path")
    if pending_upload_path:
        _cleanup_upload_file(pending_upload_path)

    st.session_state.result = None
    st.session_state.processed = False
    st.session_state.chat_history = []
    st.session_state.last_source = None
    st.session_state.last_language = None
    st.session_state.complete_pdf_data = None
    st.session_state.rag_video_id = None
    st.session_state.active_source_identity = None
    st.session_state.chat_history_source_id = None
    clear_active_translation()
    st.session_state.pop("pending_upload_path", None)
    st.session_state.pop("pending_upload_name", None)
    st.session_state.pop("pending_upload_identity", None)


def display_value(value: Any, fallback: str = "N/A") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def format_seconds(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        total = max(0, int(float(value)))
    except (TypeError, ValueError):
        return display_value(value)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def source_label(source_type: Any) -> str:
    labels = {
        "youtube": "YouTube",
        "meeting": "Uploaded Meeting",
    }
    return labels.get(str(source_type), display_value(source_type))


def compact_label(value: Any, limit: int = 34) -> str:
    """Create a compact display label while retaining the original value elsewhere."""
    text = display_value(value, "Untitled")
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "â€¦"


def status_markup(label: str, state: str = "waiting") -> str:
    """Return one consistent, accessible status presentation for UI-only use."""
    icons = {"ready": "â—", "processing": "â—‰", "waiting": "â—‹", "completed": "âœ“", "cached": "â™»", "unavailable": "â€”", "error": "âš "}
    css_state = state if state in icons else "waiting"
    return f'<span class="status-pill {css_state}">{icons[css_state]} {escape(label)}</span>'


def as_items(value: Any) -> list[Any]:
    """Normalize common backend output shapes for clean UI rendering."""
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line) for line in lines]
    return numbered or [text]


def render_items(value: Any, empty_message: str) -> None:
    items = as_items(value)
    if not items:
        st.info(empty_message)
        return

    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            main_text = item.get("task") or item.get("action") or item.get("text") or item.get("item")
            if not main_text:
                main_text = "; ".join(f"{key}: {value}" for key, value in item.items())
            metadata = []
            for key in ("owner", "assignee", "deadline", "due_date", "due"):
                if item.get(key) not in (None, ""):
                    metadata.append(f"{key.replace('_', ' ').title()}: {item[key]}")
            st.markdown(
                f'<div class="item-card"><div class="item-number">Item {index}</div>'
                f'<div>{escape(str(main_text))}</div>'
                f'<div class="muted">{escape(" Â· ".join(metadata))}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="item-card"><div class="item-number">Item {index}</div>'
                f'<div>{escape(str(item))}</div></div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Header, sidebar, and input
# -----------------------------------------------------------------------------


def render_header() -> None:
    """Render application identity and real active-content context."""
    result = st.session_state.get("result") or {}
    name = st.session_state.get("active_translation_source") or result.get("source_name") or result.get("title")
    full_context = str(name) if name else "No active video"
    context = escape(compact_label(full_context, 42))
    status = status_markup("Ready", "ready") if st.session_state.get("processed") else status_markup("Waiting", "waiting")
    html = '<div class="app-header"><div><div class="eyebrow">Video Agent Â· Meeting intelligence workspace</div><h1 class="app-title">AI Meeting Intelligence</h1><p class="app-subtitle">Turn meetings and videos into structured, searchable knowledge.</p></div><div class="header-context" title="' + escape(full_context) + '">' + status + '<br>' + context + '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def prepare_processed_video_selector(processed_videos: list[dict[str, Any]]) -> None:
    """Apply a queued active-video selection before the selectbox is created."""
    pending_id = st.session_state.pop("pending_processed_video_id", None)
    if pending_id is None:
        return
    for video in processed_videos:
        video_id = str(video.get("video_id") or "")
        if video_id == str(pending_id):
            title = str(video.get("title") or "Untitled").strip()
            source_type = str(video.get("source_type") or "video").upper()
            st.session_state["processed_video_selector"] = f"[{source_type}] {title} Â· {video_id[:8]}"
            return


def render_processed_videos() -> None:
    """Show completed videos from SQLite and let the user reopen them."""
    try:
        videos = _list_all_completed_videos()
    except Exception as exc:
        st.error(f"Could not load processed videos: {exc}")
        return

    with st.expander("ðŸ“š Processed Videos", expanded=False):
        if not videos:
            st.info("No completed videos are stored yet.")
            return

        # Keep only records with a persisted source ID for reopening.
        processed_videos = [
            video for video in videos
            if video.get("video_id")
        ]
        if not processed_videos:
            st.info("No completed videos or meetings available.")
            return

        options = {}
        labels = []

        for video in processed_videos:
            v_id = str(video.get("video_id") or "")
            title = str(video.get("title") or "Untitled").strip()
            s_type = str(video.get("source_type") or "video").upper()
            label = f"[{s_type}] {title} Â· {v_id[:8]}"
            options[label] = video
            labels.append(label)
        prepare_processed_video_selector(processed_videos)
        selected_label = st.selectbox(
            "Select a processed video",
            labels,
            key="processed_video_selector",
        )

        selected = options[selected_label]
        video_id = str(selected.get("video_id") or "")
        source = str(selected.get("source") or "")
        language = str(selected.get("language") or "en")
        active_result = st.session_state.get("result") or {}
        _, active_name, selected_is_active = get_selected_active_video_state(selected, active_result)

        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"**Selected Video:** {compact_label(str(selected.get('title') or 'Untitled'), 42)}")
            st.caption(f"**Selected ID:** {video_id}")
            if selected_is_active:
                st.caption("**Active Video:** This selected video")
            else:
                st.caption(f"**Active Video:** {compact_label(active_name, 42)}")

        with col2:
            st.caption(f"**Chunks:** {selected.get('chunk_count') or 'N/A'}")
            st.caption("**Status:** Completed")
            st.caption("**RAG:** Available on demand")

            if st.button(
                "Open / Ask Questions",
                use_container_width=True,
                type="primary",
                key=f"open_processed_{video_id}",
            ):
                active_result = st.session_state.get("result") or {}
                active_metadata = active_result.get("metadata") or {}
                if (
                    st.session_state.get("rag_video_id") == video_id
                    and active_metadata.get("video_id") == video_id
                    and active_result.get("rag_chain") is not None
                ):
                    st.success(
                        "âš¡ This video's RAG is already loaded in this session. "
                        "No vector store reload or embedding initialization."
                    )
                else:
                    _load_video_library_entry_with_notice(selected)

        delete_button_key = f"delete_processed_{video_id}"
        if st.button(
            "ðŸ—‘ï¸ Delete selected source",
            use_container_width=True,
            type="secondary",
            key=delete_button_key,
        ):
            if not video_id:
                st.warning("No source ID available to delete.")
            else:
                try:
                    delete_vector_store(video_id)
                    delete_video(video_id)

                    if st.session_state.get("rag_video_id") == video_id:
                        reset_session()

                    st.success(f"Deleted selected source: {video_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete selected source: {exc}")


def render_translation_history() -> None:
    """Browse, select, load, and safely delete saved Roman Urdu translations."""
    try:
        total_saved = get_translation_count()
        st.markdown('<div class="section-title">ðŸ“š Saved Roman Urdu Translations</div>', unsafe_allow_html=True)
        st.metric("Total saved translations", total_saved)

        if total_saved == 0:
            st.info("No saved Roman Urdu translations yet.")
            return

        search_query = st.text_input(
            "Search saved translations",
            placeholder="Search original or Roman Urdu text...",
            key="saved_translation_search",
        )
        records = list_saved_translations(limit=50, search_query=search_query)
        if not records:
            st.info("No saved translations match your search.")
            return

        active_result = st.session_state.get("result") or {}
        if active_result and not is_translation_only_result(active_result):
            records = [record for record in records if saved_translation_matches_result(record, active_result)]
            if not records:
                st.info("No saved Roman Urdu translation exists for the active video.")
                return
        elif is_translation_only_result(active_result):
            active_translation_id = (active_result.get("metadata") or {}).get("active_translation_id")
            records = [record for record in records if str(record.get("id")) == str(active_translation_id)]
            if not records:
                st.info("The active saved translation is no longer available.")
                return

        def display_name(record: dict[str, Any]) -> str:
            return str(record.get("source_name") or f"Saved Translation #{record.get('id')}")

        options = {
            f"ðŸŽ¬ {display_name(record)} â€” {record.get('updated_at') or record.get('created_at') or 'Unknown date'} Â· #{record['id']}": record["id"]
            for record in records
        }
        selected_label = st.selectbox(
            "ðŸŽ¬ Select Saved Video or Translation",
            list(options),
            key="saved_translation_selector",
        )
        selected_id = int(options[selected_label])
        selected = get_saved_translation_by_id(selected_id)
        if selected is None:
            st.warning("The selected saved translation is no longer available.")
            return

        selected_name = display_name(selected)

        # Keep the primary actions immediately below the selected record.
        load_col, delete_col = st.columns(2)
        with load_col:
            if st.button(
                "â™»ï¸ Load This Translation",
                key=f"load_saved_translation_{selected_id}",
                use_container_width=True,
                type="primary",
            ):
                original = str(selected.get("original_transcript") or "")
                roman_urdu = str(selected.get("roman_urdu_translation") or "")
                set_active_translation(
                    original,
                    roman_urdu,
                    record=selected,
                    source=selected_name,
                    clear_downstream=True,
                )
                st.session_state.roman_urdu_transcript = roman_urdu
                st.session_state.roman_urdu_source_id = get_transcript_source_id(original)
                st.session_state.result = {
                    "source_type": "saved_translation",
                    "language": "English",
                    "transcript": original,
                    "translated_transcript": roman_urdu,
                    "title": selected_name,
                    "source_name": selected_name,
                    "summary": "",
                    "executive_summary": "",
                    "actions": [],
                    "decisions": [],
                    "questions": [],
                    "metadata": {
                        "cached": True,
                        "active_translation_id": selected_id,
                        **{
                            key: selected.get(key)
                            for key in ("video_id", "meeting_id", "source_id")
                            if selected.get(key) is not None
                        },
                    },
                    "rag_chain": None,
                }
                saved_identity = get_result_identity(st.session_state.result)
                st.session_state.active_source_identity = saved_identity
                st.session_state.chat_history = []
                st.session_state.chat_history_source_id = saved_identity
                st.session_state.rag_video_id = None
                st.session_state.processed = True
                st.success("âš¡ Saved Roman Urdu translation loaded.")

        pending_delete_id = st.session_state.get("pending_delete_translation_id")
        with delete_col:
            if st.button(
                "ðŸ—‘ï¸ Delete Translation",
                key=f"delete_saved_translation_{selected_id}",
                use_container_width=True,
            ):
                st.session_state.pending_delete_translation_id = selected_id
                st.rerun()

        if pending_delete_id == selected_id:
            with st.container(border=True):
                st.warning(f"âš ï¸ Are you sure you want to delete: {selected_name}?")
                cancel_col, confirm_col = st.columns(2)
                with cancel_col:
                    if st.button("Cancel", key=f"cancel_delete_saved_translation_{selected_id}", use_container_width=True):
                        st.session_state.pop("pending_delete_translation_id", None)
                        st.rerun()
                with confirm_col:
                    if st.button("Confirm Delete", key=f"confirm_delete_saved_translation_{selected_id}", use_container_width=True):
                        if delete_translation_by_id(selected_id):
                            if st.session_state.get("active_translation_id") == selected_id:
                                clear_active_translation()
                                st.session_state.roman_urdu_transcript = None
                                st.session_state.roman_urdu_source_id = None
                                st.session_state.result = None
                                st.session_state.processed = False
                            st.session_state.pop("pending_delete_translation_id", None)
                            st.success("Saved translation deleted.")
                            st.rerun()
                        else:
                            st.warning("Saved translation was already deleted.")

        with st.container(border=True):
            st.markdown("### ðŸŸ¢ Selected Content")
            st.write(f"ðŸŽ¬ **Source:** {selected_name}")
            st.write(f"ðŸ“… **Last Updated:** {selected.get('updated_at') or 'Unknown'}")
            st.write(f"ðŸ¤– **Model:** {selected.get('model_name') or 'Model not recorded'}")
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("ðŸ“ Transcript Length", f"{len(str(selected.get('original_transcript') or '')):,} characters")
            stat_col2.metric("ðŸŒ Translation Length", f"{len(str(selected.get('roman_urdu_translation') or '')):,} characters")

        with st.expander("â–¶ ðŸ“ Original Transcript Preview", expanded=False):
            st.text(str(selected.get("original_transcript") or ""))
        with st.expander("â–¶ ðŸŒ Roman Urdu Translation Preview", expanded=False):
            st.text(str(selected.get("roman_urdu_translation") or ""))
    except Exception as error:
        st.error(f"Could not load saved Roman Urdu translations: {error}")


def _library_value(record: dict[str, Any], *keys: str) -> Any:
    """Read a persisted field, checking the record and its metadata."""
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


def _library_has_content(value: Any) -> bool:
    """Return whether a persisted value contains actual user-visible data."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _prepare_video_library_entry(video: dict[str, Any]) -> dict[str, Any]:
    """Build a library index entry without creating a second content store."""
    entry = dict(video)
    transcript = str(_library_value(video, "transcript", "original_transcript") or "")
    entry["_library_transcript"] = transcript
    translation_record = None

    explicit_translation = video.get("roman_urdu_translation")
    if _library_has_content(explicit_translation):
        translation_record = {
            "id": video.get("translation_id"),
            "video_id": video.get("video_id"),
            "meeting_id": video.get("meeting_id"),
            "source_id": video.get("source_id"),
            "roman_urdu_translation": str(explicit_translation),
            "model_name": video.get("roman_urdu_model") or video.get("model_name"),
            "source_name": get_result_source_name(video),
        }
    elif transcript:
        try:
            source_identity = (
                video.get("video_id")
                or video.get("meeting_id")
                or video.get("source_id")
            )
            translation_record = get_saved_translation_by_hash(
                get_transcript_hash(transcript),
                video_id=str(source_identity) if source_identity else None,
            )
        except Exception:
            translation_record = None

    entry["_roman_urdu_record"] = translation_record
    return entry


def _get_persisted_video_entry(video_id: str) -> dict[str, Any] | None:
    """Return the exact completed persisted video without loading the backend."""
    target_id = str(video_id or "").strip()
    if not target_id:
        return None
    try:
        record = get_video(target_id)
    except Exception:
        return None
    if not record or str(record.get("status") or "").lower() != "completed":
        return None
    return _prepare_video_library_entry(record)


def _list_all_completed_videos() -> list[dict[str, Any]]:
    """Return every completed video through the database pagination API."""
    page_size = 100
    offset = 0
    videos: list[dict[str, Any]] = []
    while True:
        page = list_videos(
            status="completed",
            limit=page_size,
            offset=offset,
        )
        videos.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return videos


def _load_video_library_entry(entry: dict[str, Any]) -> str:
    """Load one persisted video record into the existing active-result state."""
    video_id = str(entry.get("video_id") or "")
    if video_id:
        # Apply this on the next rerun, before the sidebar selectbox is instantiated.
        st.session_state["pending_processed_video_id"] = video_id

    # `entry` comes from the lightweight database listing and already contains
    # the persisted transcript and analysis fields needed by the UI. Do not call
    # get_backend() here: importing main.py can initialize Whisper eagerly.
    record = dict(entry)

    transcript = str(_library_value(record, "transcript", "original_transcript") or "")
    source_name = get_result_source_name(record)
    metadata = dict(record.get("metadata") or {})
    if video_id:
        metadata["video_id"] = video_id

    result = {
        "source": record.get("source") or source_name,
        "source_type": record.get("source_type") or "video",
        "language": record.get("language") or "en",
        "audio_chunks": record.get("audio_chunks") or [],
        "chunk_count": record.get("chunk_count") or 0,
        "transcript": transcript,
        "translated_transcript": record.get("translated_transcript") or "",
        "title": record.get("title") or source_name,
        "source_name": source_name,
        "summary": _library_value(record, "summary", "executive_summary") or "",
        "metadata": metadata,
        "rag_chain": None,
        "actions": _library_value(record, "actions", "action_items") or [],
        "decisions": _library_value(record, "decisions") or [],
        "questions": _library_value(record, "questions") or [],
    }

    # Reopening a video (e.g. via the Video Library "Open" button) always
    # rebuilds `result` from the persisted record above, which resets
    # "rag_chain" to None. That is correct for a *different* video, but if
    # this exact video is already the active one AND its RAG chain was
    # already lazily loaded earlier in this session, discarding it here
    # would force an unnecessary reload on the next question and would
    # silently drop the in-memory chain object. Reuse it instead â€” this is
    # the same identity/validity check already used by the sidebar
    # "Open / Ask Questions" reuse guard, just applied inside the shared
    # loader so every entry point benefits from it.
    previous_result = st.session_state.get("result") or {}
    previous_metadata = previous_result.get("metadata") or {}
    reuse_existing_rag = bool(
        video_id
        and str(previous_metadata.get("video_id") or "") == video_id
        and str(st.session_state.get("rag_video_id") or "") == video_id
        and previous_result.get("rag_chain") is not None
        and str(previous_result.get("transcript") or "") == transcript
    )
    if reuse_existing_rag:
        result["rag_chain"] = previous_result.get("rag_chain")

    set_active_processed_result(result, source_name)

    translation_record = entry.get("_roman_urdu_record")
    if translation_record is not None and not saved_translation_matches_result(translation_record, result):
        translation_record = None
    if transcript and translation_record is None:
        try:
            active_metadata = result.get("metadata") or {}
            active_source_id = (
                result.get("video_id")
                or result.get("meeting_id")
                or active_metadata.get("video_id")
                or active_metadata.get("meeting_id")
                or active_metadata.get("source_id")
            )
            translation_record = get_saved_translation_by_hash(
                get_transcript_hash(transcript),
                video_id=str(active_source_id) if active_source_id else None,
            )
        except Exception:
            translation_record = None
        if translation_record is not None and not saved_translation_matches_result(translation_record, result):
            translation_record = None

    if translation_record and str(translation_record.get("roman_urdu_translation") or "").strip():
        roman_urdu = str(translation_record.get("roman_urdu_translation") or "")
        set_active_translation(
            transcript,
            roman_urdu,
            record=translation_record,
            source=source_name,
            clear_downstream=True,
        )
        st.session_state.roman_urdu_transcript = roman_urdu
        st.session_state.roman_urdu_source_id = get_transcript_source_id(transcript)
    else:
        st.session_state.active_translation_id = None
        st.session_state.active_roman_urdu_translation = None
        st.session_state.active_translation_model = None
        st.session_state.active_translation_source = source_name
        st.session_state.roman_urdu_transcript = None
        st.session_state.roman_urdu_source_id = None

    # RAG remains lazy and is loaded only when the user explicitly asks a
    # question â€” unless this exact video's RAG chain was already valid in
    # the current session (reuse_existing_rag above), in which case the
    # active RAG identity is kept pointed at that already-loaded chain
    # instead of being reset and forced to reload lazily again.
    st.session_state.rag_video_id = video_id if reuse_existing_rag else None
    return source_name


def _video_library_date_value(record: dict[str, Any]) -> Any:
    """Return a real persisted creation/processing date, if one exists."""
    return _library_value(
        record,
        "created_at",
        "processed_at",
        "completed_at",
        "finished_at",
        "date",
        "updated_at",
    )


def _parse_video_library_date(value: Any) -> datetime | None:
    """Parse supported persisted date formats without inventing a date."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    candidates = (text, text.replace("Z", "+00:00"))
    for candidate in candidates:
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


def _video_library_availability(entry: dict[str, Any]) -> dict[str, bool]:
    """Calculate content availability only from data actually stored."""
    status = get_video_content_status(entry)
    return {
        "Transcript": status["transcript"],
        "Roman Urdu": status["roman_urdu"],
        "Summary": status["summary"],
        "Action Items": status["action_items"],
        "Decisions": status["decisions"],
        "Questions": status["questions"],
    }


def _video_library_sort_entries(
    entries: list[dict[str, Any]],
    sort_order: str,
) -> list[dict[str, Any]]:
    """Sort entries safely, keeping records with unknown dates consistently last."""
    sorted_entries = list(entries)
    if sort_order in {"Name A â†’ Z", "Name Z â†’ A"}:
        return sorted(
            sorted_entries,
            key=lambda entry: get_result_source_name(entry).casefold(),
            reverse=sort_order == "Name Z â†’ A",
        )

    dated: list[tuple[dict[str, Any], datetime]] = []
    undated: list[dict[str, Any]] = []
    for entry in sorted_entries:
        parsed = _parse_video_library_date(_video_library_date_value(entry))
        if parsed is None:
            undated.append(entry)
        else:
            dated.append((entry, parsed))
    dated.sort(key=lambda item: item[1], reverse=sort_order == "Newest First")
    undated.sort(key=lambda entry: get_result_source_name(entry).casefold())
    return [entry for entry, _ in dated] + undated


def _load_video_library_entry_with_notice(entry: dict[str, Any]) -> str:
    """Load saved content and queue a notice that survives the Streamlit rerun."""
    source_name = _load_video_library_entry(entry)
    st.session_state["reopen_notice"] = f"Loaded {source_name}. Existing saved content is now active."
    return source_name


def render_reopen_notice() -> None:
    """Render and consume a saved-content notice after a callback-triggered rerun."""
    notice = st.session_state.pop("reopen_notice", None)
    if notice:
        st.success(notice)


def _navigate_library_export(entry: dict[str, Any]) -> None:
    """Load this exact library record, then open the existing Export page."""
    _load_video_library_entry(entry)
    navigate_to("ðŸ“¥ Export")


def _render_video_library_card(
    entry: dict[str, Any],
    pending_delete_id: str | None,
) -> None:
    source_name = get_result_source_name(entry)
    video_id = str(entry.get("video_id") or "")
    library_id = video_id or source_name
    availability = _video_library_availability(entry)
    parsed_date = _parse_video_library_date(_video_library_date_value(entry))
    active_result = st.session_state.get("result") or {}
    active_metadata = active_result.get("metadata") or {}
    active_video_id = str(active_metadata.get("video_id") or "")
    active_source = st.session_state.get("active_translation_source")
    is_active = (
        bool(video_id) and active_video_id == video_id
    ) or (
        not active_video_id and active_source == source_name
    )

    with st.container(border=True):
        if is_active:
            st.markdown("<div class=\"library-active\">â— Active content</div>", unsafe_allow_html=True)
        st.markdown(f"<div class=\"library-card-title\">ðŸŽ¬ {escape(source_name)}</div>", unsafe_allow_html=True)
        if parsed_date is not None:
            st.markdown(f"<div class=\"library-card-meta\">ðŸ“… Processed: {parsed_date.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class=\"library-card-meta\">ðŸ“… Date unavailable</div>", unsafe_allow_html=True)

        source_type = entry.get("source_type")
        if source_type:
            st.markdown(f"<div class=\"library-card-meta\">Source type: {escape(str(source_type))}</div>", unsafe_allow_html=True)
        transcript = str(entry.get("_library_transcript") or "")
        if transcript:
            st.markdown(f"<div class=\"library-card-meta\">Transcript: {len(transcript):,} characters Â· {len(transcript.split()):,} words</div>", unsafe_allow_html=True)

        indicator_columns = st.columns(3)
        indicator_labels = {
            "Transcript": "ðŸŽ™ Transcript",
            "Roman Urdu": "ðŸŒ Roman Urdu",
            "Summary": "ðŸ“Š Summary",
            "Action Items": "âœ… Action Items",
            "Decisions": "ðŸ“Œ Decisions",
            "Questions": "â“ Questions",
        }
        for index, (key, available) in enumerate(availability.items()):
            with indicator_columns[index % 3]:
                label = indicator_labels[key]
                state = "completed" if available else "unavailable"
                st.markdown(status_markup(label, state), unsafe_allow_html=True)

        open_col, export_col, delete_col = st.columns(3)
        with open_col:
            if st.button(
                "Open",
                key=f"video_library_open_{library_id}",
                use_container_width=True,
                type="primary",
                on_click=_load_video_library_entry_with_notice,
                args=(entry,),
            ):
                pass
        with export_col:
            if st.button(
                "Export",
                key=f"video_library_export_{library_id}",
                use_container_width=True,
                disabled=not availability["Transcript"],
                on_click=_navigate_library_export,
                args=(entry,),
            ):
                pass
        with delete_col:
            if st.button(
                "Delete",
                key=f"video_library_delete_{library_id}",
                use_container_width=True,
                disabled=not video_id,
            ):
                st.session_state.video_library_pending_delete_id = library_id
                st.rerun()

        if pending_delete_id == library_id:
            st.warning(
                f"Delete {source_name}? This removes the selected processed video and its vector store. "
                "The transcript-hash Roman Urdu cache is retained because it may be shared by another video."
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "Confirm Delete",
                    key=f"video_library_confirm_delete_{library_id}",
                    use_container_width=True,
                ):
                    try:
                        delete_vector_store(video_id)
                        delete_video(video_id)
                        active_result = st.session_state.get("result") or {}
                        active_metadata = active_result.get("metadata") or {}
                        active_source = st.session_state.get("active_translation_source")
                        if (
                            str(active_metadata.get("video_id") or "") == video_id
                            or (not active_metadata.get("video_id") and active_source == source_name)
                        ):
                            reset_session()
                        st.session_state.pop("video_library_pending_delete_id", None)
                        st.success(f"Deleted {source_name}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not delete {source_name}: {exc}")
            with cancel_col:
                if st.button(
                    "Cancel",
                    key=f"video_library_cancel_delete_{library_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop("video_library_pending_delete_id", None)
                    st.rerun()


def _render_video_library_analytics(entries: list[dict[str, Any]]) -> None:
    """Render read-only statistics from the already-loaded library entries."""
    analytics = calculate_library_analytics(
        entries,
        source_name_getter=get_result_source_name,
    )

    st.markdown("### ðŸ“Š Library Analytics")
    with st.container():
        st.markdown("<div class=\"analytics-card\">", unsafe_allow_html=True)
        # Explicit rows avoid relying on CSS to make Streamlit columns wrap safely.
        overview_row_one = st.columns(2)
        overview_row_one[0].metric("ðŸŽ¬ Total Videos", f"{analytics['total_videos']:,}")
        overview_row_one[1].metric("ðŸ“ Total Transcript Words", f"{analytics['total_words']:,}")
        overview_row_two = st.columns(2)
        overview_row_two[0].metric("â± Total Duration", format_duration(analytics["duration_seconds"]))
        overview_row_two[1].metric("ðŸŒ Roman Urdu Coverage", f"{analytics['roman_urdu_coverage']:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    if analytics["duration_seconds"] is not None and analytics["duration_video_count"] < analytics["total_videos"]:
        st.caption("Based on videos with available duration data.")
    st.caption(
        f"Roman Urdu: {analytics['roman_urdu_translated_count']} of "
        f"{analytics['roman_urdu_eligible_count']} eligible videos translated."
    )

    st.markdown("### ðŸ“ˆ Library Insights")
    insight_row = st.columns(2)
    most_recent = analytics["most_recent"]
    if most_recent is None:
        insight_row[0].metric("ðŸ•’ Most Recently Processed", "Date unavailable")
        insight_row[0].caption("Date unavailable")
    else:
        recent_name = str(most_recent["source_name"])
        insight_row[0].metric("ðŸ•’ Most Recently Processed", compact_label(recent_name, 24))
        insight_row[0].caption(most_recent["date"].strftime("%d %b %Y"))

    most_complete = analytics["most_complete"]
    if most_complete is None:
        insight_row[1].metric("ðŸ† Most Complete", "No content")
        insight_row[1].caption("No videos available")
    else:
        complete_name = str(most_complete["source_name"])
        insight_row[1].metric("ðŸ† Most Complete", compact_label(complete_name, 24))
        insight_row[1].caption(f"{most_complete['percentage']:.2f}% complete")

    st.metric("ðŸ“ˆ Overall Library Completion", f"{analytics['overall_completion']:.2f}%")
    st.progress(
        min(max(analytics["overall_completion"] / 100, 0.0), 1.0),
        text="Available content across all six components",
    )

    st.markdown("#### Content Gap Insights")
    gap_row_one = st.columns(2)
    gap_row_one[0].metric("ðŸŒ Missing Roman Urdu", f"{analytics['missing_roman_urdu']:,}")
    gap_row_one[1].metric("ðŸ“Š Missing Summary", f"{analytics['missing_summary']:,}")
    gap_row_two = st.columns(1)
    gap_row_two[0].metric("âš  Incomplete Videos", f"{analytics['incomplete_videos']:,}")

    st.markdown("#### Completion Distribution")
    distribution_entries = analytics["completion_entries"]
    for entry in distribution_entries:
        st.progress(
            min(max(entry["percentage"] / 100, 0.0), 1.0),
            text=f"{compact_label(entry['source_name'], 42)} Â· {entry['percentage']:.2f}%",
        )


def render_video_library_page() -> None:
    """Display the read-only analytics dashboard and saved video records."""
    st.markdown('<div class="section-title">ðŸ“š Video Library</div>', unsafe_allow_html=True)
    st.caption("Browse, filter, open, export, and safely delete saved video content.")

    try:
        videos = _list_all_completed_videos()
    except Exception as exc:
        st.error(f"Could not load the Video Library: {exc}")
        return

    if not videos:
        st.markdown("### ðŸ“Š Library Analytics")
        st.info(
            "No processed videos available yet.\n\n"
            "Process your first video to start building analytics."
        )
        return

    entries = [_prepare_video_library_entry(video) for video in videos]
    _render_video_library_analytics(entries)

    st.markdown("---")
    search_query = st.text_input(
        "ðŸ”Ž Search your videos...",
        placeholder="Search by source name or title...",
        key="video_library_search",
    ).strip().casefold()
    sort_order = st.selectbox(
        "Sort videos",
        ["Newest First", "Oldest First", "Name A â†’ Z", "Name Z â†’ A"],
        key="video_library_sort",
    )
    filter_label = st.selectbox(
        "Filter by content",
        [
            "All Videos",
            "Transcript Available",
            "Roman Urdu Available",
            "Summary Available",
            "Action Items Available",
            "Decisions Available",
            "Questions Available",
        ],
        key="video_library_filter",
    )

    filtered_entries = entries
    if search_query:
        filtered_entries = [
            entry
            for entry in filtered_entries
            if search_query in " ".join(
                str(entry.get(key) or "")
                for key in ("source_name", "title", "source")
            ).casefold()
        ]

    availability_key = {
        "Transcript Available": "Transcript",
        "Roman Urdu Available": "Roman Urdu",
        "Summary Available": "Summary",
        "Action Items Available": "Action Items",
        "Decisions Available": "Decisions",
        "Questions Available": "Questions",
    }.get(filter_label)
    if availability_key:
        filtered_entries = [
            entry
            for entry in filtered_entries
            if _video_library_availability(entry).get(availability_key, False)
        ]

    if not filtered_entries:
        st.info("No videos match your current search or filters.")
        return

    filtered_entries = _video_library_sort_entries(filtered_entries, sort_order)
    pending_delete_id = st.session_state.get("video_library_pending_delete_id")
    for row_start in range(0, len(filtered_entries), 2):
        columns = st.columns(2)
        for column, entry in zip(columns, filtered_entries[row_start:row_start + 2]):
            with column:
                _render_video_library_card(entry, pending_delete_id)



def _current_active_result() -> dict[str, Any]:
    """Return the current result overlaid with the authoritative active state."""
    return active_result_view(st.session_state.get("result") or {})


def render_transcript_page() -> None:
    """Display only the active original transcript."""
    st.markdown('<div class="section-title">ðŸŽ™ Transcript</div>', unsafe_allow_html=True)
    active_original = st.session_state.get("active_original_transcript")
    if not active_original:
        st.info("No active transcript is available. Please process or load content first.")
        return
    st.info(
        "ðŸŸ¢ Active Content\n\n"
        f"Source: {st.session_state.get('active_translation_source') or 'current content'}\n\n"
        f"Transcript length: {len(str(active_original)):,} characters"
    )
    st.markdown("### ðŸŽ™ Original Transcript")
    st.text_area(
        "Original Transcript",
        value=str(active_original),
        height=620,
        disabled=True,
        key=f"transcript_page_{st.session_state.get('active_translation_id') or 'current'}",
    )


def render_roman_urdu_page() -> None:
    """Display Roman Urdu controls only on the dedicated Roman Urdu page."""
    result = _current_active_result()
    active_original = st.session_state.get("active_original_transcript")
    if not active_original:
        st.info("No active transcript is available. Please process or load content first.")
        return
    active_translation = st.session_state.get("active_roman_urdu_translation")
    if active_translation:
        st.markdown(status_markup("Roman Urdu translation is ready", "ready"), unsafe_allow_html=True)
        with st.expander("View Translation", expanded=False):
            st.text(str(active_translation))
    else:
        render_roman_urdu_transcript(result, show_content=False, show_exports=False)


def render_summary_analysis_page() -> None:
    """Display the current result in the complete AI results tab interface."""
    result = _current_active_result()
    if not st.session_state.get("result"):
        st.info("No active processed result is available. Please process or load content first.")
        return

    st.markdown('<div class="section-title">ðŸ“Š Summary & Analysis</div>', unsafe_allow_html=True)
    tabs = st.tabs([
        "Overview",
        "Summary",
        "Action Items",
        "Decisions",
        "Questions",
        "AI Chat",
        "âœ¨ Prompt Studio",
    ])

    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_summary(result, show_exports=False)
    with tabs[2]:
        render_items(result.get("actions"), "No action items found.")
    with tabs[3]:
        render_items(result.get("decisions"), "No key decisions found.")
    with tabs[4]:
        render_items(result.get("questions"), "No open questions found.")
    with tabs[5]:
        render_chat(result)
    with tabs[6]:
        render_prompt_studio(result)


def render_export_page() -> None:
    """Display export controls only on the dedicated Export page."""
    result = _current_active_result()
    st.markdown('<div class="section-title">ðŸ“¥ Export</div>', unsafe_allow_html=True)
    st.caption("Exports use the current active transcript and Roman Urdu translation only.")
    try:
        original, translation, _ = validate_export_state(result)
    except ExportStateError as error:
        st.error(str(error))
        return
    export_title = str(result.get("title") or result.get("source_name") or "Roman Urdu Translation Report")
    col_txt, col_excel, col_pdf = st.columns(3)
    with col_txt:
        try:
            st.download_button(
                "ðŸ“„ Download TXT",
                data=build_translation_txt(original, translation, title=export_title),
                file_name="roman-urdu-translation.txt",
                mime="text/plain; charset=utf-8",
                key="dedicated_export_txt",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"TXT export failed: {error}")
    with col_excel:
        try:
            st.download_button(
                "ðŸ“Š Download Excel",
                data=build_translation_excel(original, translation),
                file_name="roman-urdu-translation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dedicated_export_excel",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Excel export failed: {error}")
    with col_pdf:
        try:
            st.download_button(
                "ðŸ“• Download PDF",
                data=build_translation_pdf(original, translation, title=export_title),
                file_name="roman-urdu-translation.pdf",
                mime="application/pdf",
                key="dedicated_export_pdf",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"PDF export failed: {error}")


def navigate_to(page: str) -> None:
    """Update the sidebar navigation widget before the next Streamlit rerun."""
    st.session_state["app_page"] = page


def render_home_page() -> None:
    """Keep Home focused on processing and compact navigation/status."""
    render_input_section()
    if st.session_state.get("active_original_transcript"):
        st.success("âœ“ Transcript Ready")
        if st.session_state.get("active_roman_urdu_translation"):
            st.success("âœ“ Roman Urdu Translation Ready")
    st.markdown("### Open a feature")
    # Responsive three-column shortcut grid; avoids six cramped buttons.
    nav_columns = st.columns(3)
    pages = [
        ("ðŸŽ™ View Transcript", "ðŸŽ™ Transcript"),
        ("ðŸŒ Roman Urdu", "ðŸŒ Roman Urdu"),
        ("ðŸ“Š Summary & Analysis", "ðŸ“Š Summary & Analysis"),
        ("ðŸ“š Saved History", "ðŸ“š Saved History"),
        ("ðŸ“š Video Library", "ðŸ“š Video Library"),
        ("ðŸ“¥ Export", "ðŸ“¥ Export"),
    ]
    for index, (label, page) in enumerate(pages):
        column = nav_columns[index % 3]
        with column:
            st.button(
                label,
                key=f"home_nav_{page}",
                use_container_width=True,
                on_click=navigate_to,
                args=(page,),
            )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class=\"sidebar-kicker\">VIDEO AGENT WORKSPACE</div><div class=\"sidebar-brand\">AI Meeting Assistant</div>", unsafe_allow_html=True)
        st.caption("Professional meeting and video analysis")
        if st.button("New session", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()
        st.markdown("#### Navigate")
        selected_page = st.radio(
            "Navigate",
            [
                "ðŸ  Home",
                "ðŸŽ™ Transcript",
                "ðŸŒ Roman Urdu",
                "ðŸ“Š Summary & Analysis",
                "ðŸ“š Saved History",
                "ðŸ“š Video Library",
                "ðŸ“¥ Export",
            ],
            key="app_page",
        )

        st.divider()
        st.markdown("#### System status")
        st.markdown(status_markup("Pipeline ready", "ready"), unsafe_allow_html=True)
        if st.session_state.get("processed"):
            result = st.session_state.get("result") or {}
            metadata = result.get("metadata") or {}
            if metadata.get("cached"):
                st.markdown(status_markup("RAG available on demand Â· Cached", "cached"), unsafe_allow_html=True)
            else:
                st.markdown(status_markup("RAG available on demand", "waiting"), unsafe_allow_html=True)
        else:
            st.markdown(status_markup("Waiting for a processed meeting", "waiting"), unsafe_allow_html=True)

        if st.session_state.get("last_source"):
            st.divider()
            full_source = str(st.session_state.last_source)
            st.caption(f"Source: {compact_label(full_source, 38)}")
            st.caption(f"Language: {st.session_state.last_language or 'N/A'}")

        st.divider()
        render_processed_videos()
        return selected_page


def render_input_section() -> None:
    st.markdown('<div class="section-title">Process New Meeting</div>', unsafe_allow_html=True)
    with st.container(border=True):
        youtube_tab, upload_tab = st.tabs(["YouTube URL", "Upload Video"])

        with youtube_tab:
            youtube_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="youtube_url")
            language_name = st.selectbox("Language", list(LANGUAGE_OPTIONS), key="youtube_language")
            if st.button("Process video", type="primary", use_container_width=True, key="process_youtube"):
                if not youtube_url.strip():
                    st.warning("Please enter a YouTube URL before processing.")
                elif not is_youtube_url(youtube_url):
                    st.warning("Please provide a valid YouTube URL.")
                else:
                    process_source(
                        source=youtube_url.strip(),
                        language=LANGUAGE_OPTIONS[language_name],
                        display_source="YouTube",
                        source_type="youtube",
                    )

        with upload_tab:
            st.caption("4GB per file â€¢ MP4, AVI, MOV, MKV, MP3, WAV, M4A")
            uploaded_file = st.file_uploader(
                "Upload Video or Audio",
                type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "m4a"],
            )

            uploaded_source = None
            if uploaded_file is not None:
                upload_identity = get_upload_identity(uploaded_file)
                pending_identity = st.session_state.get("pending_upload_identity")
                pending_path = st.session_state.get("pending_upload_path")
                if (
                    pending_identity == upload_identity
                    and pending_path
                    and Path(pending_path).exists()
                ):
                    # The SAME selected upload was already persisted on an
                    # earlier rerun: reuse the existing file. No full-file
                    # re-read, no SHA-256 recomputation, no duplicate copy.
                    uploaded_source = Path(pending_path)
                else:
                    # New selection (or stale identity): persist exactly once,
                    # then retire the previous upload file when it is safe.
                    previous_path = (
                        Path(pending_path)
                        if pending_path and Path(pending_path).exists()
                        else None
                    )
                    uploaded_source = persist_uploaded_file(uploaded_file)
                    st.session_state["pending_upload_path"] = str(uploaded_source.resolve())
                    st.session_state["pending_upload_name"] = uploaded_file.name
                    st.session_state["pending_upload_identity"] = upload_identity
                    if previous_path is not None and previous_path != uploaded_source:
                        _cleanup_upload_file(previous_path)
            elif "pending_upload_path" in st.session_state:
                uploaded_source = Path(st.session_state.get("pending_upload_path") or "")
                if not uploaded_source.exists():
                    uploaded_source = None
                    st.session_state.pop("pending_upload_path", None)
                    st.session_state.pop("pending_upload_name", None)
                    st.session_state.pop("pending_upload_identity", None)

            language_name = st.selectbox(
                "Language",
                list(LANGUAGE_OPTIONS),
                key="upload_language",
            )
            if st.button(
                "Process meeting",
                type="primary",
                use_container_width=True,
                key="process_upload",
            ):
                if uploaded_file is None and "pending_upload_path" not in st.session_state:
                    st.warning("Please upload an audio or video file before processing.")
                elif uploaded_source is None:
                    st.warning("Uploaded file could not be saved to the project workspace.")
                else:
                    saved_path = Path(st.session_state.get("pending_upload_path", str(uploaded_source.resolve())))
                    display_name = st.session_state.get("pending_upload_name", uploaded_file.name if uploaded_file is not None else saved_path.name)
                    # persist_uploaded_file derives the digest while streaming; reuse its filename ID.
                    meeting_id = saved_path.stem

                    process_source(
                        source=str(saved_path.resolve()),
                        language=LANGUAGE_OPTIONS[language_name],
                        display_source=display_name,
                        source_type="meeting",
                        meeting_id=meeting_id,
                    )


def is_youtube_url(value: str) -> bool:
    """
    Validate common YouTube URL formats.

    Supported:
        - youtube.com/watch?v=...
        - www.youtube.com/watch?v=...
        - youtu.be/...
        - youtube.com/shorts/...

    Query parameters such as:
        &t=1153s
        ?si=...
    are allowed.
    """

    from urllib.parse import urlparse, parse_qs

    value = value.strip()

    if not value:
        return False

    # Scheme missing ho to automatically add HTTPS
    candidate = (
        value
        if "://" in value
        else f"https://{value}"
    )

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False

    hostname = (
        parsed.hostname or ""
    ).lower().removeprefix("www.")

    # --------------------------------------------------------
    # youtu.be/VIDEO_ID
    # --------------------------------------------------------

    if hostname == "youtu.be":

        video_id = (
            parsed.path
            .strip("/")
            .split("/")[0]
        )

        return bool(video_id)

    # --------------------------------------------------------
    # youtube.com
    # --------------------------------------------------------

    if hostname != "youtube.com":
        return False

    # --------------------------------------------------------
    # Standard YouTube URL
    #
    # /watch?v=VIDEO_ID&t=1153s
    # --------------------------------------------------------

    if parsed.path == "/watch":

        video_id = parse_qs(
            parsed.query
        ).get(
            "v",
            [None],
        )[0]

        return bool(video_id)

    # --------------------------------------------------------
    # YouTube Shorts
    # --------------------------------------------------------

    if parsed.path.startswith("/shorts/"):

        video_id = (
            parsed.path
            .split("/shorts/", 1)[1]
            .split("/", 1)[0]
        )

        return bool(video_id)

    return False

def extract_video_id(source: str) -> str | None:
    """Extract a YouTube video ID without importing the backend.

    This function is intentionally kept in app.py because cache lookup happens
    during Streamlit rendering. Importing main.py here would initialize the
    Whisper engine before the user starts processing a video.
    """
    from urllib.parse import parse_qs, urlparse

    value = str(source or "").strip()
    if not value:
        return None

    candidate = value if "://" in value else f"https://{value}"

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None

    hostname = (parsed.hostname or "").lower().removeprefix("www.")

    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        return video_id or None

    if hostname != "youtube.com":
        return None

    if parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        return video_id or None

    if parsed.path.startswith("/shorts/"):
        video_id = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        return video_id or None

    return None


def persist_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded file atomically and return its permanent path.

    The file is streamed to a temporary ``.part`` file while calculating its
    SHA-256 digest.  Only after the write is complete is the temporary file
    atomically moved to its final content-addressed filename.  This prevents
    partially-written uploads from ever being treated as valid source files.
    """
    import tempfile

    if uploaded_file is None:
        raise ValueError("No uploaded file was supplied.")

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = str(getattr(uploaded_file, "name", "upload") or "upload")
    suffix = Path(original_name).suffix.lower() or ".mp4"
    if suffix.lstrip(".") not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported upload type: {suffix or 'unknown'}. "
            f"Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
        )

    digest = hashlib.sha256()
    total_bytes = 0
    chunk_size = 8 * 1024 * 1024
    temp_path: Path | None = None

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(upload_dir),
            prefix=".upload-",
            suffix=".part",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                digest.update(chunk)
                temp_file.write(chunk)
                total_bytes += len(chunk)

            temp_file.flush()
            try:
                os.fsync(temp_file.fileno())
            except OSError:
                # fsync is best-effort on filesystems that do not expose it.
                pass

        if total_bytes <= 0:
            raise ValueError("Uploaded file is empty.")

        meeting_id = "meeting_" + digest.hexdigest()[:16]
        saved_file_path = upload_dir / f"{meeting_id}{suffix}"

        # The digest-based destination makes the same upload idempotent.
        # If it already exists, discard only our temporary copy.
        if saved_file_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            finally:
                temp_path = None
        else:
            # os.replace works atomically on the same filesystem and is safe
            # for the Windows deployment used by this project.
            os.replace(str(temp_path), str(saved_file_path))
            temp_path = None

        if not saved_file_path.is_file():
            raise RuntimeError(
                "Uploaded file persistence failed: final file does not exist."
            )

        # Verify the persisted file is non-empty before returning it.
        try:
            if saved_file_path.stat().st_size != total_bytes:
                raise RuntimeError(
                    "Uploaded file persistence failed: saved file size does not match the upload."
                )
        except OSError as exc:
            raise RuntimeError(
                f"Uploaded file persistence failed: could not verify saved file: {exc}"
            ) from exc

        return saved_file_path

    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise


def get_upload_identity(uploaded_file: Any) -> str:
    """Return a stable identity for one selected upload across reruns.

    Streamlit's ``file_id`` is preferred because it identifies the selected
    UploadedFile without reading the entire file again.  Older Streamlit
    versions may not expose it, so the fallback uses filename, size and a
    lightweight object identity for the current session.
    """
    if uploaded_file is None:
        return ""

    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return f"file_id:{file_id}"

    name = str(getattr(uploaded_file, "name", "") or "")
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return f"name:{name}|size:{size}"

    # Last-resort fallback. It is only used by Streamlit versions without
    # file_id and without size metadata.
    return f"name:{name}|object:{id(uploaded_file)}"


def _cleanup_upload_file(upload_path: Any) -> None:
    """Best-effort removal of an orphaned temporary/pending upload.

    Cleanup is deliberately defensive: it only touches files under the
    application's ``uploads`` directory and never deletes the active source.
    Any filesystem failure is swallowed because cleanup must never break a
    Streamlit rerun or session reset.
    """
    try:
        if upload_path in (None, ""):
            return

        upload_path = Path(upload_path)
        if not upload_path.exists() or not upload_path.is_file():
            return

        upload_root = Path(UPLOAD_DIR).resolve()
        upload_path_resolved = upload_path.resolve()
        try:
            upload_path_resolved.relative_to(upload_root)
        except (OSError, ValueError):
            return

        active_result = st.session_state.get("result") or {}
        active_source = str(
            active_result.get("source")
            or (active_result.get("metadata") or {}).get("source")
            or ""
        ).strip()

        if active_source:
            try:
                if Path(active_source).resolve() == upload_path_resolved:
                    return
            except (OSError, ValueError):
                pass

        upload_path.unlink(missing_ok=True)
    except Exception:
        # Cleanup is explicitly best-effort. Never propagate filesystem errors.
        pass


def process_uploaded_file(
    uploaded_file: Any,
    language: str,
) -> None:
    """Save upload to a permanent path and then start the processing pipeline."""
    saved_file_path = persist_uploaded_file(uploaded_file)
    meeting_id = saved_file_path.stem

    process_source(
        source=str(saved_file_path.resolve()),
        language=language,
        display_source=uploaded_file.name,
        source_type="meeting",
        meeting_id=meeting_id,
    )

def normalize_pipeline_result(result: Any) -> dict[str, Any]:
    """Convert PipelineResult into the dictionary shape expected by the UI."""
    if isinstance(result, dict):
        return result

    backend = get_backend()
    PipelineResult = backend.PipelineResult

    if isinstance(result, PipelineResult):
        metadata = dict(result.metadata or {})
        return {
            "source": result.source,
            "source_type": result.source_type,
            "language": result.language,
            "audio_chunks": result.audio_chunks,
            "chunk_count": len(result.audio_chunks),
            "transcript": result.transcript,
            "translated_transcript": result.translated_transcript,
            "title": result.title,
            "summary": result.summary,
            "metadata": metadata,
            "rag_chain": result.rag_chain,
            "actions": list(getattr(result, "actions", metadata.get("actions", [])) or []),
            "decisions": list(getattr(result, "decisions", metadata.get("decisions", [])) or []),
            "questions": list(getattr(result, "questions", metadata.get("questions", [])) or []),
        }

    raise RuntimeError(
        "Pipeline returned an unsupported result type: "
        f"{type(result).__name__}"
    )


PROCESSING_STAGE_ORDER = [
    "input",
    "audio",
    "transcription",
    "translation",
    "title",
    "summary",
    "rag",
    "finalizing",
]

PROCESSING_STAGE_LABELS = {
    "input": "Preparing Input",
    "audio": "Converting Audio",
    "transcription": "Transcribing Audio",
    "translation": "Translating Transcript",
    "title": "Generating Title",
    "summary": "Generating Summary",
    "rag": "Building Vector Database",
    "finalizing": "Finalizing Results",
}


def create_processing_stage_state() -> dict[str, dict[str, str]]:
    return {
        stage: {
            "state": "waiting",
            "message": "Waiting",
        }
        for stage in PROCESSING_STAGE_ORDER
    }


def apply_progress_event(
    stage_state: dict[str, dict[str, str]],
    event: dict[str, Any],
) -> None:
    stage = str(event.get("stage") or "input").strip().lower()
    if stage not in stage_state:
        return

    status = str(event.get("status") or "waiting").strip().lower()
    current = event.get("current")
    total = event.get("total")
    message = str(event.get("message") or event.get("status") or "Waiting")

    if status in {"running", "completed", "failed", "skipped", "cached", "waiting"}:
        stage_state[stage]["state"] = status
    else:
        stage_state[stage]["state"] = "waiting"

    if current is not None and total not in (None, "") and total not in (0, "0"):
        try:
            current_int = int(current)
            total_int = int(total)
            if current_int > 0 and total_int > 0:
                message = f"{current_int} / {total_int} chunks"
        except (TypeError, ValueError):
            pass

    if isinstance(message, str) and message.strip():
        stage_state[stage]["message"] = message.strip()


def compute_overall_progress(stage_state: dict[str, dict[str, str]]) -> int:
    if not stage_state:
        return 0

    total_steps = len(stage_state)
    completed = 0
    for stage in PROCESSING_STAGE_ORDER:
        state = stage_state.get(stage, {}).get("state", "waiting")
        if state in {"completed", "skipped", "cached"}:
            completed += 1
        elif state == "running":
            completed += 0.5

    percent = round((completed / total_steps) * 100)
    return max(0, min(100, percent))


def compute_active_step(stage_state: dict[str, dict[str, str]]) -> int:
    completed = 0
    for stage in PROCESSING_STAGE_ORDER:
        state = stage_state.get(stage, {}).get("state", "waiting")
        if state in {"completed", "skipped", "cached"}:
            completed += 1
        elif state == "running":
            return max(1, completed + 1)
    return max(1, completed)


def render_processing_panel(
    stage_state: dict[str, dict[str, str]],
    chunk_summary: str | None = None,
    error_message: str | None = None,
    elapsed_seconds: int | None = None,
) -> None:
    overall_percent = compute_overall_progress(stage_state)
    active_step = compute_active_step(stage_state)
    running_stage = next(
        (
            stage for stage in PROCESSING_STAGE_ORDER
            if stage_state.get(stage, {}).get("state") == "running"
        ),
        None,
    )
    failed_stage = next(
        (
            stage for stage in PROCESSING_STAGE_ORDER
            if stage_state.get(stage, {}).get("state") == "failed"
        ),
        None,
    )
    active_stage = running_stage or failed_stage or next(
        (
            stage for stage in PROCESSING_STAGE_ORDER
            if stage_state.get(stage, {}).get("state", "waiting") == "waiting"
        ),
        PROCESSING_STAGE_ORDER[-1],
    )
    entry = stage_state.get(active_stage, {"state": "waiting", "message": "Waiting"})
    state = entry.get("state", "waiting")
    label = PROCESSING_STAGE_LABELS.get(active_stage, active_stage.replace("_", " ").title())
    text = entry.get("message", "Waiting")
    detail = {
        "waiting": "Waiting",
        "running": text,
        "completed": "Completed",
        "failed": "Failed",
        "skipped": "Skipped",
        "cached": "Loaded from Cache",
    }.get(state, text)
    presentation_state = {
        "running": "processing",
        "completed": "completed",
        "failed": "error",
        "skipped": "unavailable",
        "cached": "cached",
        "waiting": "waiting",
    }.get(state, "waiting")
    completed_count = sum(
        stage_state.get(stage, {}).get("state") in {"completed", "skipped", "cached"}
        for stage in PROCESSING_STAGE_ORDER
    )

    with st.container(border=True):
        st.markdown(
            '<div class="section-title">PROCESSING VIDEO / MEETING</div>',
            unsafe_allow_html=True,
        )
        st.progress(
            overall_percent / 100,
            text=f"{overall_percent}% Â· {PROCESSING_STAGE_LABELS.get(active_stage, 'Starting')}",
        )
        st.caption(f"Step {active_step} of {len(PROCESSING_STAGE_ORDER)}")
        st.markdown(status_markup(f"{label} â€” {detail}", presentation_state), unsafe_allow_html=True)
        if completed_count:
            st.caption(f"âœ“ {completed_count} stage(s) completed")
        if chunk_summary:
            st.caption(chunk_summary)
        if elapsed_seconds is not None:
            st.caption(f"Elapsed time: {format_seconds(elapsed_seconds)}")
        if error_message:
            st.error(f"âœ• Processing failed â€” {error_message}")


def process_source(
    source: str,
    language: str,
    display_source: str,
    source_type: str,
    meeting_id: str | None = None,
) -> None:
    """Run the pipeline in a background worker and render live progress."""

    if source_type not in {"youtube", "meeting"}:
        raise ValueError("source_type must be 'youtube' or 'meeting'.")

    previous_processing_state = {
        key: st.session_state.get(key)
        for key in (
            "result",
            "processed",
            "active_translation_id",
            "active_original_transcript",
            "active_roman_urdu_translation",
            "active_translation_source",
            "active_translation_model",
            "roman_urdu_transcript",
            "roman_urdu_source_id",
            "last_source",
            "last_language",
            "rag_video_id",
            "complete_pdf_data",
            "active_source_identity",
            "chat_history",
            "chat_history_source_id",
            "_preserve_chat_on_same_source",
        )
    }
    previous_processing_state["chat_history"] = list(st.session_state.get("chat_history") or [])

    if source_type == "youtube":
        cached_video_id = extract_video_id(source)

        if cached_video_id and video_exists(cached_video_id, completed_only=True):
            try:
                # Use the same lightweight persisted-record path as Video
                # Library â†’ Open. This must not import main.py or initialize
                # Whisper merely because the YouTube video is already cached.
                entry = _get_persisted_video_entry(cached_video_id)
                if entry is None:
                    raise RuntimeError(
                        "Cached record is no longer available or not completed."
                    )

                source_name = _load_video_library_entry(entry)
                result = st.session_state.get("result") or {}
                metadata = dict(result.get("metadata") or {})
                metadata["cached"] = True
                metadata["video_id"] = cached_video_id
                metadata["rag_cache"] = "lazy_streamlit_resource"
                result["metadata"] = metadata
                result["rag_chain"] = result.get("rag_chain")

                with st.container(border=True):
                    st.markdown(
                        '<div class="section-title">â™» Cached Meeting Found</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("The transcript, summary, and vector database will be reused from the existing record.")
                    st.success("âœ“ Transcript â€” Loaded from Cache")
                    st.success("âœ“ Summary â€” Loaded from Cache")
                    st.success("âœ“ Meeting Intelligence â€” Loaded from Cache")
                    st.success("âœ“ Vector Database â€” Loaded from Existing Storage")
                    st.caption(f"Video ID: {cached_video_id}")
                    st.caption(f"Title: {display_value(result.get('title') or source_name, 'Untitled video')}")

                if st.session_state.get("rag_video_id") == cached_video_id and result.get("rag_chain") is not None:
                    st.success(
                        "âš¡ Existing session RAG reused. No vector store reload or embedding initialization."
                    )
                else:
                    st.success(
                        "Cached video loaded successfully. Transcript is ready; RAG will load when you ask a question."
                    )
                return

            except Exception as exc:
                for key, value in previous_processing_state.items():
                    st.session_state[key] = value
                st.error("Could not load the cached video.")
                with st.expander("Technical details"):
                    st.exception(exc)
                return

    processing_placeholder = st.empty()
    stage_state = create_processing_stage_state()
    result_queue: Queue = Queue()
    started_at = time.monotonic()
    completed_chunks: list[dict[str, Any]] = []
    current_chunk: dict[str, Any] | None = None
    total_chunks = 0
    current_stage = "input"
    final_result = None
    final_error = None

    def update_processing_display(
        chunk_summary: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with processing_placeholder.container():
            render_processing_panel(
                stage_state,
                chunk_summary=chunk_summary,
                error_message=error_message,
                elapsed_seconds=int(time.monotonic() - started_at),
            )

    update_processing_display()

    def worker() -> None:
        try:
            def on_progress(event: dict[str, Any]) -> None:
                result_queue.put(("progress", event))

            backend = get_backend()
            pipeline_video_id = meeting_id if source_type == "meeting" else None
            result = backend.run_pipeline(
                source,
                source_type=source_type,
                language=language,
                video_id=pipeline_video_id,
                progress_callback=on_progress,
            )
            result_queue.put(("success", result))
        except Exception as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, name="video-agent-pipeline", daemon=True)
    thread.start()

    try:
        while thread.is_alive() or not result_queue.empty():
            drained_event = False
            while True:
                try:
                    event_type, payload = result_queue.get_nowait()
                except Empty:
                    break
                drained_event = True

                if event_type == "success":
                    final_result = payload
                    continue
                if event_type == "error":
                    final_error = payload
                    continue
                if event_type == "progress":
                    event = payload or {}
                    current_stage = str(event.get("stage") or current_stage).lower()
                    apply_progress_event(stage_state, event)

                    if current_stage == "transcription":
                        current = int(event.get("current", 0) or 0)
                        total = int(event.get("total", 0) or 0)
                        if total > 0:
                            status = str(event.get("status") or "")
                            chunk_elapsed = float(event.get("elapsed", 0.0) or 0.0)
                            if "completed" in status.lower():
                                completed_chunks.append({
                                    "index": current,
                                    "total": total,
                                    "elapsed": chunk_elapsed,
                                })
                                current_chunk = None
                            elif current > 0:
                                current_chunk = {
                                    "index": current,
                                    "total": total,
                                    "elapsed": chunk_elapsed,
                                }
                            total_chunks = total

                            if stage_state.get("transcription", {}).get("state") == "running":
                                chunk_text = (
                                    f"Chunk {current} / {total}"
                                    if current and total
                                    else f"{total} chunks queued"
                                )
                            else:
                                chunk_text = f"{total} / {total} chunks completed"
                            update_processing_display(chunk_summary=chunk_text)
                            continue

                    update_processing_display()


            if thread.is_alive() and not drained_event:
                time.sleep(0.25)

        thread.join(timeout=2.0)

        if final_error is not None:
            failed_stage = current_stage if current_stage in stage_state else "input"
            if failed_stage in stage_state:
                stage_state[failed_stage]["state"] = "failed"
                stage_state[failed_stage]["message"] = "Failed"
            update_processing_display(error_message=str(final_error))
            raise final_error

        if final_result is None:
            raise RuntimeError("Pipeline worker finished without returning a result.")

        result = normalize_pipeline_result(final_result)
        result.setdefault("source_type", source_type)
        result.setdefault("language", language)
        result.setdefault("processing_time", int(time.monotonic() - started_at))
        result.setdefault("source_name", display_source)

        for stage in PROCESSING_STAGE_ORDER:
            if stage_state.get(stage, {}).get("state") in {"waiting", "running"}:
                stage_state[stage]["state"] = "completed"
                stage_state[stage]["message"] = "Completed"

        processing_placeholder.empty()

        st.success("âœ“ Processing Completed Successfully")
        set_active_processed_result(result, display_source)
        active_rag_chain = result.get("rag_chain") if isinstance(result, dict) else None
        active_video_id = (result.get("metadata") or {}).get("video_id") if isinstance(result, dict) else None
        st.session_state.rag_video_id = (
            str(active_video_id)
            if active_video_id and active_rag_chain is not None
            else None
        )

    except Exception as exc:
        failed_stage = current_stage if current_stage in stage_state else "input"
        if failed_stage in stage_state:
            stage_state[failed_stage]["state"] = "failed"
            stage_state[failed_stage]["message"] = "Failed"
        for key, value in previous_processing_state.items():
            st.session_state[key] = value
        update_processing_display(error_message=str(exc))
        with st.expander("Technical details", expanded=True):
            st.exception(exc)


# -----------------------------------------------------------------------------
# Results and chat
# -----------------------------------------------------------------------------


def render_kpis(result: dict[str, Any]) -> None:
    metadata = result.get("metadata") or {}
    cache_status = "Cached" if metadata.get("cached") else "Processed"
    metrics = [
        ("Duration", format_seconds(result.get("duration"))),
        ("Chunks", display_value(result.get("chunk_count"))),
        ("Processing time", format_seconds(result.get("processing_time"))),
        ("Source", source_label(result.get("source_type"))),
        ("Status", cache_status),
    ]
    # Explicit two-item rows keep every KPI readable even when Streamlit will not wrap columns.
    for start in range(0, len(metrics), 2):
        row_columns = st.columns(2)
        for column, (label, value) in zip(row_columns, metrics[start:start + 2]):
            with column:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{escape(str(value))}</div></div>',
                    unsafe_allow_html=True,
                )


def render_overview(result: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-panel"><div class="eyebrow">Title</div>'
        f'<h2>{escape(display_value(result.get("title"), "Untitled meeting"))}</h2>'
        f'<div class="eyebrow">Summary preview</div>'
        f'<div>{display_value(result.get("summary"), "No summary generated.")}</div></div>',
        unsafe_allow_html=True,
    )


def build_meeting_pdf(result: dict[str, Any]) -> bytes:
    """Build a robust PDF report and return valid PDF bytes."""
    import io

    buffer = io.BytesIO()

    # Prefer a real Unicode font. Helvetica can break on Hindi/Urdu text.
    pdf_font = "Helvetica"
    font_candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\NotoSans-Regular.ttf"),
        Path(r"C:\Windows\Fonts\NotoSansDevanagari-Regular.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            try:
                pdfmetrics.registerFont(
                    TTFont("VideoAgentUnicode", str(font_path))
                )
                pdf_font = "VideoAgentUnicode"
                break
            except Exception:
                continue

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=display_value(result.get("title"), "AI Meeting Report"),
        author="AI Meeting Intelligence",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PDFTitleFixed",
        parent=styles["Title"],
        fontName=pdf_font,
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "PDFHeadingFixed",
        parent=styles["Heading2"],
        fontName=pdf_font,
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "PDFBodyFixed",
        parent=styles["BodyText"],
        fontName=pdf_font,
        fontSize=9.5,
        leading=14,
        spaceAfter=7,
        wordWrap="CJK",
    )
    small_style = ParagraphStyle(
        "PDFSmallFixed",
        parent=styles["BodyText"],
        fontName=pdf_font,
        fontSize=8.5,
        leading=12,
        spaceAfter=4,
        wordWrap="CJK",
    )

    story = []

    def clean_text(value: Any, fallback: str = "N/A") -> str:
        """Make arbitrary pipeline text safe for ReportLab Paragraph."""
        text_value = display_value(value, fallback)
        # Remove characters ReportLab/PDF text handling cannot safely process.
        text_value = "".join(
            ch for ch in text_value
            if ch in "\n\r\t" or ord(ch) >= 32
        )
        return text_value.replace("\r\n", "\n").replace("\r", "\n")

    def add_heading(value: Any) -> None:
        story.append(
            Paragraph(
                escape(clean_text(value)),
                heading_style,
            )
        )

    def add_body(value: Any, fallback: str = "N/A") -> None:
        text_value = clean_text(value, fallback)

        # Do not put a massive transcript into one Paragraph.
        # Splitting it makes ReportLab much more reliable.
        parts = text_value.split("\n")
        if not parts:
            parts = [fallback]

        for part in parts:
            part = part.strip()
            if part:
                story.append(
                    Paragraph(
                        escape(part),
                        body_style,
                    )
                )

    story.append(
        Paragraph(
            escape(clean_text(result.get("title"), "AI Meeting Report")),
            title_style,
        )
    )
    story.append(
        Paragraph(
            "AI Meeting Intelligence â€” Complete Meeting Report",
            small_style,
        )
    )

    add_heading("Processing Information")

    processing_items = [
        f"Source: {source_label(result.get('source_type'))}",
        f"Language: {display_value(result.get('language'))}",
        f"Duration: {format_seconds(result.get('duration'))}",
        f"Chunks: {display_value(result.get('chunk_count'))}",
        f"Processing time: {format_seconds(result.get('processing_time'))}",
    ]

    for item in processing_items:
        add_body(item)

    add_heading("Complete Summary")
    add_body(result.get("summary"), "No summary generated.")

    def add_items_section(title: str, value: Any, empty: str) -> None:
        add_heading(title)
        items = as_items(value)

        if not items:
            add_body(empty)
            return

        for index, item in enumerate(items, 1):
            if isinstance(item, dict):
                main = (
                    item.get("task")
                    or item.get("action")
                    or item.get("text")
                    or item.get("item")
                )

                if not main:
                    main = "; ".join(
                        f"{key}: {value}"
                        for key, value in item.items()
                    )

                add_body(f"{index}. {main}")

                metadata = []
                for key in (
                    "owner",
                    "assignee",
                    "deadline",
                    "due_date",
                    "due",
                ):
                    if item.get(key) not in (None, ""):
                        metadata.append(
                            f"{key.replace('_', ' ').title()}: {item[key]}"
                        )

                if metadata:
                    story.append(
                        Paragraph(
                            escape(clean_text(" Â· ".join(metadata))),
                            small_style,
                        )
                    )
            else:
                add_body(f"{index}. {item}")

    add_items_section(
        "Action Items",
        result.get("actions"),
        "No action items found.",
    )
    add_items_section(
        "Key Decisions",
        result.get("decisions"),
        "No key decisions found.",
    )
    add_items_section(
        "Open Questions",
        result.get("questions"),
        "No open questions found.",
    )

    story.append(PageBreak())

    add_heading("Full Transcript")
    add_body(
        result.get("transcript"),
        "No transcript available.",
    )

    translated = result.get("translated_transcript")
    if translated and translated != result.get("transcript"):
        add_heading("Translated Transcript")
        add_body(
            translated,
            "No translated transcript available.",
        )

    doc.build(story)

    pdf_bytes = buffer.getvalue()

    # Hard validation: Streamlit must receive an actual PDF, not empty data.
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError(
            "PDF generation failed: invalid or empty PDF data."
        )

    return pdf_bytes

def render_summary(result: dict[str, Any], show_exports: bool = True) -> None:
    summary = display_value(result.get("summary"), "No summary generated.")
    st.markdown('<div class="section-title">Complete Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-panel">{escape(summary).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    if not show_exports:
        return

    transcript = display_value(
        result.get("transcript"),
        "No transcript generated.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download summary (.txt)",
            data=summary,
            file_name="meeting-summary.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
            key="download_summary_txt",
        )

    with col2:
        st.download_button(
            "Download transcript (.txt)",
            data=transcript,
            file_name="meeting-transcript.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
            key="download_transcript_txt",
        )

    with col3:
        try:
            # Build the PDF once per processed session and keep the exact
            # binary bytes in Session State. This avoids regenerating the
            # document during a download-triggered rerun.
            pdf_data = st.session_state.get("complete_pdf_data")

            if not isinstance(pdf_data, (bytes, bytearray)) or not bytes(pdf_data).startswith(b"%PDF-"):
                pdf_data = build_meeting_pdf(result)

                if not isinstance(pdf_data, (bytes, bytearray)):
                    raise RuntimeError(
                        "PDF generator did not return binary PDF data."
                    )

                pdf_data = bytes(pdf_data)

                if not pdf_data.startswith(b"%PDF-"):
                    raise RuntimeError(
                        "Generated PDF is invalid or empty."
                    )

                st.session_state.complete_pdf_data = pdf_data

            else:
                pdf_data = bytes(pdf_data)

            # IMPORTANT:
            # on_click="ignore" makes the download frontend-only.
            # Streamlit therefore does not rerun the app while Chrome is
            # receiving the PDF, avoiding the "File wasn't available on site"
            # race condition seen with this application.
            st.download_button(
                label="Download complete report (.pdf)",
                data=pdf_data,
                file_name="ai-meeting-complete-report.pdf",
                mime="application/pdf",
                key="download_complete_report_pdf",
                on_click="ignore",
                type="primary",
                width="stretch",
            )

            st.caption(
                f"PDF ready Â· {len(pdf_data) / 1024:.1f} KB"
            )

        except Exception as exc:
            st.error("Could not generate the complete PDF report.")

            with st.expander("PDF technical details"):
                st.exception(exc)


def render_transcript(result: dict[str, Any]) -> None:
    transcript = display_value(
        result.get("transcript"),
        "No transcript available.",
    )

    translated = display_value(
        result.get("translated_transcript"),
        "",
    )

    transcript_col, translated_col = st.columns(2)

    with transcript_col:
        st.download_button(
            "Download transcript (.txt)",
            data=transcript,
            file_name="meeting-transcript.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
            key="download_transcript_tab_txt",
        )

    with translated_col:
        if translated and translated != transcript:
            st.download_button(
                "Download translated transcript (.txt)",
                data=translated,
                file_name="meeting-transcript-translated.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key="download_translated_transcript_txt",
            )

    query = st.text_input("Transcript search", placeholder="Search transcript...", key="transcript_search")
    if query.strip():
        matches = [line for line in transcript.splitlines() if query.lower() in line.lower()]
        if matches:
            st.info(f"Found {len(matches)} matching line(s).")
            st.text("\n".join(matches))
        else:
            st.warning("No matching transcript sections found.")
    else:
        with st.expander("Full transcript", expanded=True):
            st.text(transcript)


def render_chat(result: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Ask Your Meeting</div>', unsafe_allow_html=True)
    st.caption("Ask questions based on the processed meeting transcript.")

    chat_identity = get_result_identity(result)
    if st.session_state.get("chat_history_source_id") != chat_identity:
        st.session_state.chat_history = []
        st.session_state.chat_history_source_id = chat_identity

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this meeting...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            rag_chain = result.get("rag_chain")

            # Cached YouTube videos intentionally keep RAG lazy. On the first
            # question, load only that video's existing vector store/RAG chain.
            if rag_chain is None:
                metadata = result.get("metadata") or {}
                video_id = metadata.get("video_id")

                if video_id:
                    try:
                        with st.spinner("Loading meeting knowledge..."):
                            rag_chain = get_cached_rag_chain(
                                video_id=str(video_id),
                                source_type=str(
                                    result.get("source_type") or "youtube"
                                ),
                            )

                        result["rag_chain"] = rag_chain
                        st.session_state.result = result
                        st.session_state.rag_video_id = str(video_id)
                    except Exception as exc:
                        answer = "Could not load the RAG knowledge base for this video."
                        st.error(answer)
                        with st.expander("RAG technical details"):
                            st.exception(exc)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                        return
                else:
                    answer = (
                        "RAG chain is not available for this meeting. "
                        "Please process the meeting again."
                    )
                    st.error(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    return

            try:
                with st.spinner("Searching the meeting knowledge..."):
                    # Directly call core.rag_engine. Never import main.py from
                    # the chat path because main.py owns Whisper initialization.
                    answer = ask_rag_question(rag_chain, question)

                st.markdown(answer)
            except Exception as exc:
                answer = "I could not generate an answer from the meeting knowledge."
                st.error(answer)
                with st.expander("Chat technical details"):
                    st.exception(exc)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

def render_prompt_studio(result: dict[str, Any]) -> None:
    """Render the Prompt Studio and keep all variables safely scoped to this function."""
    st.subheader("ðŸ§  AI Prompt Studio")
    st.caption(
        "Generate a high-quality AI prompt using the processed transcript, summary, and meeting insights."
    )

    current_source_id = (
        result.get("metadata", {}).get("video_id")
        or result.get("source")
        or "prompt_studio"
    )

    if st.session_state.get("prompt_studio_source_id") != current_source_id:
        st.session_state.prompt_studio_source_id = current_source_id
        st.session_state.generated_prompt = ""
        for widget_key in ("prompt_studio_goal", "prompt_studio_type"):
            st.session_state.pop(widget_key, None)

    prompt_type = st.selectbox(
        "Select Prompt Type",
        options=[
            "learning",
            "project",
            "coding",
            "research",
            "content",
            "business",
            "custom",
        ],
        format_func=lambda value: value.title(),
        key="prompt_studio_type",
    )

    user_goal = st.text_area(
        "What do you want to create?",
        placeholder=(
            "Example: Create a complete beginner-friendly roadmap based on the concepts discussed in this meeting."
        ),
        height=120,
        key="prompt_studio_goal",
    )

    def build_meeting_intelligence_payload() -> dict[str, Any]:
        return {
            "executive_summary": result.get("executive_summary"),
            "actions": result.get("actions"),
            "decisions": result.get("decisions"),
            "questions": result.get("questions"),
        }

    def generate_from_current_context() -> str:
        if not user_goal or not user_goal.strip():
            raise ValueError("Please describe what you want the AI prompt to create.")

        transcript = result.get("transcript") or ""
        summary = result.get("summary") or ""
        meeting_intelligence = build_meeting_intelligence_payload()

        return generate_prompt(
            summary=summary,
            transcript=transcript,
            meeting_intelligence=meeting_intelligence,
            prompt_type=prompt_type,
            user_goal=user_goal.strip(),
        )

    if st.button("âœ¨ Generate AI Prompt", use_container_width=True, key="generate_ai_prompt"):
        try:
            prompt_text = generate_from_current_context()
            st.session_state.generated_prompt = prompt_text
        except Exception as exc:
            st.error("Prompt generation failed.")
            with st.expander("Technical details"):
                st.exception(exc)

    generated_prompt = st.session_state.get("generated_prompt", "")

    if generated_prompt:
        st.success("Prompt generated successfully!")

        edited_prompt = st.text_area(
            "Generated Prompt",
            value=generated_prompt,
            height=500,
            key="generated_prompt_display",
        )
        st.session_state.generated_prompt = edited_prompt

        col1, col2, col3 = st.columns(3)

        with col1:
            payload = edited_prompt or generated_prompt
            safe_payload = json.dumps(payload)

            st.html(
                f"""
                <!DOCTYPE html>
                <html>
                <body style="margin:0; padding:0;">

                    <button
                        id="copyButton"
                        style="
                            width:100%;
                            height:40px;
                            border-radius:8px;
                            border:1px solid #4CAF50;
                            cursor:pointer;
                            font-size:14px;
                            background:transparent;
                            color:white;
                        "
                    >
                        ðŸ“‹ Copy Prompt
                    </button>

                    <script>
                        const button = document.getElementById("copyButton");

                        button.addEventListener("click", async () => {{
                            const text = {safe_payload};

                            try {{
                                await navigator.clipboard.writeText(text);

                                button.innerText = "âœ… Copied!";

                            }} catch (error) {{

                                const textarea = document.createElement("textarea");

                                textarea.value = text;

                                textarea.style.position = "fixed";
                                textarea.style.left = "-9999px";

                                document.body.appendChild(textarea);

                                textarea.focus();
                                textarea.select();

                                const successful = document.execCommand("copy");

                                document.body.removeChild(textarea);

                                if (successful) {{
                                    button.innerText = "âœ… Copied!";
                                }} else {{
                                    button.innerText = "âŒ Copy Failed";
                                }}
                            }}

                            setTimeout(() => {{
                                button.innerText = "ðŸ“‹ Copy Prompt";
                            }}, 2000);
                        }});
                    </script>

                </body>
                </html>
                """,
                unsafe_allow_javascript=True,
            )
        with col2:
            st.download_button(
                label="â¬‡ï¸ Download .txt",
                data=edited_prompt or generated_prompt,
                file_name="ai_generated_prompt.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key="download_generated_prompt",
            )

        with col3:
            if st.button("ðŸ”„ Regenerate", key="regenerate_ai_prompt", use_container_width=True):
                try:
                    regenerated_prompt = generate_from_current_context()
                    st.session_state.generated_prompt = regenerated_prompt
                except Exception as exc:
                    st.error("Prompt regeneration failed.")
                    with st.expander("Technical details"):
                        st.exception(exc)

def render_roman_urdu_transcript(
    result: dict[str, Any],
    show_content: bool = True,
    show_exports: bool = True,
) -> None:
    """Render Roman Urdu transcript section."""

    st.subheader("ðŸŒ Roman Urdu Transcript")

    transcript = result.get("transcript", "")

    if not transcript or not transcript.strip():
        st.warning(
            "No transcript is available for Roman Urdu conversion."
        )
        return

    source_name = get_result_source_name(result)
    active_metadata = result.get("metadata") or {}
    active_source_id = (
        result.get("video_id")
        or result.get("meeting_id")
        or active_metadata.get("video_id")
        or active_metadata.get("meeting_id")
        or active_metadata.get("source_id")
    )

    # Har processed result ke liye unique source identify karna
    source_id = active_source_id or get_transcript_source_id(transcript)

    # Agar new video/meeting load hui hai to old translation clear karo
    if (
        st.session_state.roman_urdu_source_id is not None
        and st.session_state.roman_urdu_source_id != source_id
    ):
        st.session_state.roman_urdu_transcript = None
        st.session_state.roman_urdu_source_id = None

    st.caption(
        "Convert the English transcript into natural Pakistani Roman Urdu."
    )

    if st.button(
        "ðŸŒ Convert to Roman Urdu",
        key=f"convert_roman_urdu_{source_id}",
        use_container_width=True,
    ):
        try:
            try:
                cached_translation = get_cached_translation(
                    transcript,
                    video_id=str(active_source_id) if active_source_id else None,
                )
            except Exception as cache_error:
                cached_translation = None
                st.warning(f"Translation cache unavailable; continuing without cache: {cache_error}")

            if cached_translation is not None:
                cached_record = get_saved_translation_by_hash(
                    get_transcript_hash(transcript),
                    video_id=str(active_source_id) if active_source_id else None,
                )
                if cached_record is not None and not saved_translation_matches_result(cached_record, result):
                    cached_translation = None
                    cached_record = None
            if cached_translation is not None:
                set_active_translation(
                    transcript,
                    cached_translation,
                    record=cached_record,
                    source=source_name,
                    clear_downstream=True,
                )
                st.session_state.roman_urdu_transcript = cached_translation
                st.session_state.roman_urdu_source_id = source_id
                st.success("âš¡ Roman Urdu translation loaded from saved database.")
            else:
                st.info("ðŸŒ No saved translation found. Processing Roman Urdu translation...")
                with st.spinner(
                    "Translating transcript into Roman Urdu..."
                ):
                    roman_urdu_result = translate_to_roman_urdu(
                        transcript=transcript,
                        max_retries=3,
                    )

                st.session_state.roman_urdu_transcript = roman_urdu_result
                st.session_state.roman_urdu_source_id = source_id
                saved_record = None
                try:
                    active_metadata = result.get("metadata") or {}
                    active_source_id = (
                        result.get("video_id")
                        or result.get("meeting_id")
                        or active_metadata.get("video_id")
                        or active_metadata.get("meeting_id")
                        or active_metadata.get("source_id")
                    )
                    save_translation_with_source_identity(
                        transcript,
                        roman_urdu_result,
                        model_name=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                        source_name=source_name,
                        source_id=str(active_source_id) if active_source_id else None,
                    )
                    saved_record = get_saved_translation_by_hash(
                        get_transcript_hash(transcript),
                        video_id=str(active_source_id) if active_source_id else None,
                    )
                except Exception as cache_error:
                    st.warning(f"Translation completed, but could not be saved to the cache: {cache_error}")
                else:
                    st.success("ðŸ’¾ Roman Urdu translation saved for future use.")
                set_active_translation(
                    transcript,
                    roman_urdu_result,
                    record=saved_record,
                    source=source_name,
                    clear_downstream=True,
                )

        except Exception as error:
            st.error(
                f"Roman Urdu translation failed: {error}"
            )

    # Generated result display
    if st.session_state.roman_urdu_transcript:
        if show_content:
            st.markdown("---")
            st.text_area(
                "Roman Urdu Transcript",
                value=st.session_state.roman_urdu_transcript,
                height=500,
                key=f"roman_urdu_transcript_display_{source_id}",
            )

        # Export only authoritative, identity-validated data already present
        # in session state; legacy roman_urdu_transcript is never exported.
        if show_exports:
            try:
                export_original, export_translation, _ = validate_export_state(result)
            except ExportStateError as error:
                st.error(str(error))
                return
            st.markdown("---")
            st.subheader("ðŸ’¾ Save Translation")
            export_title = str(result.get("title") or "Roman Urdu Translation Report")
            export_columns = st.columns(3)

            with export_columns[0]:
                try:
                    txt_bytes = build_translation_txt(
                        export_original,
                        export_translation,
                        title=export_title,
                    )
                    st.download_button(
                        "ðŸ“„ Download TXT",
                        data=txt_bytes,
                        file_name="roman-urdu-translation.txt",
                        mime="text/plain; charset=utf-8",
                        use_container_width=True,
                        key="download_roman_urdu_export_txt",
                    )
                except Exception as error:
                    st.error(f"TXT export failed: {error}")

            with export_columns[1]:
                try:
                    excel_bytes = build_translation_excel(
                        export_original,
                        export_translation,
                    )
                    st.download_button(
                        "ðŸ“Š Download Excel",
                        data=excel_bytes,
                        file_name="roman-urdu-translation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_roman_urdu_export_excel",
                    )
                except Exception as error:
                    st.error(f"Excel export failed: {error}")

            with export_columns[2]:
                try:
                    pdf_bytes = build_translation_pdf(
                        export_original,
                        export_translation,
                        title=export_title,
                    )
                    st.download_button(
                        "ðŸ“• Download PDF",
                        data=pdf_bytes,
                        file_name="roman-urdu-translation.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_roman_urdu_export_pdf",
                    )
                except Exception as error:
                    st.error(f"PDF export failed: {error}")

def render_footer() -> None:
    st.markdown(
        """
        <div style="
            margin-top: 3rem;
            padding: 1rem 0;
            border-top: 1px solid rgba(132,151,196,0.12);
            color: #96a3bf;
            font-size: 0.78rem;
            text-align: center;
        ">
            AI Meeting Intelligence Â· Whisper Â· Mistral Â· Chroma RAG
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------


def main() -> None:
    configure_page()
    inject_css()
    initialize_state()
    selected_page = render_sidebar()
    render_header()
    render_reopen_notice()

    if selected_page == "ðŸŽ™ Transcript":
        render_transcript_page()
    elif selected_page == "ðŸŒ Roman Urdu":
        render_roman_urdu_page()
    elif selected_page == "ðŸ“Š Summary & Analysis":
        render_summary_analysis_page()
    elif selected_page == "ðŸ“š Saved History":
        render_translation_history()
    elif selected_page == "ðŸ“š Video Library":
        render_video_library_page()
    elif selected_page == "ðŸ“¥ Export":
        render_export_page()
    else:
        render_home_page()

    render_footer()


if __name__ == "__main__":
    main()
