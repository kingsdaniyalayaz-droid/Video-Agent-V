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
import traceback
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

from core.database import (
    delete_video,
    get_video,
    list_videos,
    recover_stale_processing_videos,
    video_exists,
)
from core.vector_store import DeleteResult, delete_vector_store
from core.prompt_generator import generate_prompt
from core.roman_urdu_translator import (
    PermanentAPIError,
    RateLimitError,
    RetryableAPIError,
    TranslationPipelineError,
    translate_to_roman_urdu,
)
from core.translation_error_handling import (
    UNEXPECTED_ERROR_MESSAGE,
    classify_translation_error,
)
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


@st.cache_resource(show_spinner=False)
def get_backend():
    """Load main.py only when the backend is actually needed.

    main.py currently initializes the Whisper engine during import.
    Lazy-loading prevents normal Streamlit startup/reruns from initializing
    Whisper before the user actually needs the processing backend.
    """
    import importlib

    return importlib.import_module("main")


@st.cache_resource(show_spinner=False)
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
    """Apply the premium neon black workspace design system and responsive rules."""
    st.markdown("""
    <style>
    :root {
        --bg: #050505;
        --bg-alt: #070A07;
        --surface: #0D1410;
        --surface-2: #111A14;
        --surface-3: #141C17;
        --sidebar: #080D0A;
        --border: #1E2B23;
        --border-strong: rgba(57, 255, 136, 0.36);
        --green: #39FF88;
        --green-2: #00E676;
        --green-3: #18C96E;
        --green-glow: rgba(57, 255, 136, 0.15);
        --blue: #4F7CFF;
        --purple: #8B5CF6;
        --text: #F1F5F9;
        --text-soft: #94A3B8;
        --text-muted: #64748B;
        --danger: #FF5A5F;
        --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);
        --radius-xl: 26px;
        --radius-lg: 20px;
        --radius-md: 16px;
        --radius-sm: 12px;
    }

    html, body, [class*="css"] {
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(79, 124, 255, 0.08), transparent 26rem),
            radial-gradient(circle at 86% 10%, rgba(57, 255, 136, 0.08), transparent 28rem),
            radial-gradient(circle at 80% 28%, rgba(139, 92, 246, 0.08), transparent 22rem),
            linear-gradient(180deg, var(--bg) 0%, var(--bg-alt) 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1480px;
        padding: 1.4rem clamp(1rem, 2.8vw, 2.6rem) 3.6rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top, rgba(57, 255, 136, 0.08), transparent 18rem),
            linear-gradient(180deg, rgba(8, 13, 10, 0.98) 0%, rgba(5, 8, 6, 1) 100%);
        border-right: 1px solid rgba(30, 43, 35, 0.9);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    .top-header {
        display: flex;
        align-items: stretch;
        justify-content: space-between;
        gap: 1.1rem;
        margin-bottom: 1.35rem;
    }

    .top-header-card,
    .workspace-panel,
    .page-banner,
    .industry-hero,
    .result-panel,
    .item-card,
    .kpi-card,
    .processing-shell,
    div[data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stExpander"],
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(13, 20, 16, 0.94), rgba(17, 26, 20, 0.94));
        border: 1px solid rgba(30, 43, 35, 0.96);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow);
    }

    .top-header-card {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.35rem 1.2rem;
        min-height: 150px;
    }

    .top-header-card::before,
    .workspace-panel::before,
    .page-banner::before,
    .industry-hero::before,
    .processing-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(120deg, transparent 0%, rgba(57, 255, 136, 0.06) 48%, transparent 100%);
        opacity: 0.7;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .brand-logo {
        width: 3.3rem;
        height: 3.3rem;
        border-radius: 18px;
        display: grid;
        place-items: center;
        background:
            radial-gradient(circle at center, rgba(57, 255, 136, 0.28), rgba(57, 255, 136, 0.06) 45%, rgba(10, 15, 11, 0.95) 80%),
            linear-gradient(145deg, rgba(17, 26, 20, 0.98), rgba(6, 10, 7, 0.98));
        border: 1px solid rgba(57, 255, 136, 0.32);
        box-shadow: 0 0 0 1px rgba(57, 255, 136, 0.08), 0 0 28px rgba(57, 255, 136, 0.12);
        color: var(--green);
        font-size: 1.35rem;
    }

    .brand-kicker,
    .page-banner-kicker,
    .section-kicker,
    .sidebar-kicker,
    .feature-label,
    .orbit-eyebrow,
    .workspace-chip,
    .library-active,
    .item-number,
    .kpi-label {
        color: var(--green);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .brand-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(2rem, 3vw, 2.8rem);
        line-height: 1.02;
        letter-spacing: -0.04em;
        font-weight: 800;
    }

    .brand-subtitle,
    .workspace-subtitle,
    .page-banner-subtitle,
    .hero-core-copy,
    .feature-desc,
    .processing-note,
    .library-card-meta,
    .muted,
    .app-subtitle {
        color: var(--text-soft);
        line-height: 1.6;
    }

    .brand-subtitle {
        margin: 0.6rem 0 0;
        max-width: 52rem;
        font-size: 0.98rem;
    }

    .workspace-panel {
        position: relative;
        padding: 1.15rem 1.15rem 1rem;
        min-height: 150px;
    }

    .workspace-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.36rem 0.68rem;
        border-radius: 999px;
        border: 1px solid rgba(57, 255, 136, 0.22);
        background: rgba(57, 255, 136, 0.08);
        box-shadow: inset 0 0 0 1px rgba(57, 255, 136, 0.04);
        margin-bottom: 0.85rem;
    }

    .workspace-title {
        margin: 0;
        color: var(--text);
        font-size: 1.16rem;
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    .workspace-subtitle {
        margin: 0.35rem 0 0.95rem;
        font-size: 0.92rem;
    }

    .context-panel {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.75rem;
        margin-top: 0.7rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(30, 43, 35, 0.9);
    }

    .context-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .context-value {
        margin-top: 0.28rem;
        color: var(--text);
        font-size: 0.94rem;
        word-break: break-word;
    }

    .page-banner {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        overflow: hidden;
    }

    .page-banner-title {
        color: var(--text);
        font-size: clamp(1.25rem, 2vw, 1.55rem);
        font-weight: 780;
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }

    .page-banner-badge {
        display: inline-flex;
        align-items: center;
        justify-content: flex-end;
        min-width: 13rem;
    }

    .industry-hero {
        position: relative;
        overflow: hidden;
        padding: 1.35rem;
        margin-bottom: 1.05rem;
        min-height: 540px;
        background:
            radial-gradient(circle at center, rgba(79, 124, 255, 0.08), transparent 13rem),
            radial-gradient(circle at center, rgba(57, 255, 136, 0.07), transparent 18rem),
            linear-gradient(180deg, rgba(13, 20, 16, 0.98), rgba(10, 15, 11, 0.96));
    }

    .industry-orbit {
        position: relative;
        min-height: 500px;
    }

    .orbit-ring,
    .orbit-ring-2,
    .orbit-ring-3 {
        position: absolute;
        inset: 50% auto auto 50%;
        transform: translate(-50%, -50%);
        border-radius: 999px;
        border: 1px solid rgba(30, 43, 35, 0.95);
        pointer-events: none;
    }

    .orbit-ring { width: 24rem; height: 24rem; }
    .orbit-ring-2 { width: 32rem; height: 32rem; border-color: rgba(79, 124, 255, 0.18); }
    .orbit-ring-3 { width: 39rem; height: 39rem; border-color: rgba(57, 255, 136, 0.12); }

    .hero-core {
        position: absolute;
        inset: 50% auto auto 50%;
        transform: translate(-50%, -50%);
        width: min(24rem, 88vw);
        min-height: 15rem;
        padding: 1.4rem 1.3rem;
        border-radius: 28px;
        border: 1px solid rgba(57, 255, 136, 0.22);
        background:
            radial-gradient(circle at top, rgba(57, 255, 136, 0.12), transparent 10rem),
            linear-gradient(180deg, rgba(13, 20, 16, 0.98), rgba(17, 26, 20, 0.95));
        box-shadow: 0 0 38px rgba(57, 255, 136, 0.08), var(--shadow);
        text-align: center;
        z-index: 3;
    }

    .hero-core h2 {
        margin: 0.45rem 0 0.8rem;
        color: var(--text);
        font-size: clamp(2rem, 3.2vw, 3rem);
        line-height: 1;
        letter-spacing: -0.05em;
    }

    .hero-core-copy {
        font-size: 0.96rem;
        max-width: 18rem;
        margin: 0 auto;
    }

    .industry-card {
        position: absolute;
        width: 10.2rem;
        padding: 0.85rem 0.95rem;
        border-radius: 18px;
        background: rgba(17, 26, 20, 0.92);
        border: 1px solid rgba(30, 43, 35, 0.95);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.28);
        z-index: 2;
    }

    .industry-card::before {
        content: "";
        position: absolute;
        left: 0.9rem;
        top: 0.85rem;
        width: 0.42rem;
        height: 0.42rem;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.35);
    }

    .industry-title {
        color: var(--text);
        font-weight: 700;
        margin-left: 0.8rem;
        line-height: 1.25;
        font-size: 0.92rem;
    }

    .industry-desc {
        margin-top: 0.45rem;
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .pos-1 { top: 2%; left: 8%; }
    .pos-2 { top: 10%; left: 68%; }
    .pos-3 { top: 23%; left: -1%; }
    .pos-4 { top: 22%; left: 79%; }
    .pos-5 { top: 55%; left: -1%; }
    .pos-6 { top: 57%; left: 80%; }
    .pos-7 { top: 76%; left: 8%; }
    .pos-8 { top: 82%; left: 66%; }
    .pos-9 { top: 86%; left: 38%; transform: translateX(-50%); }

    .section-title {
        color: var(--text);
        font-size: 1.22rem;
        font-weight: 780;
        margin: 1.2rem 0 0.3rem;
        letter-spacing: -0.03em;
    }

    .section-subtitle {
        color: var(--text-soft);
        margin-bottom: 0.95rem;
        font-size: 0.94rem;
    }

    .feature-shell {
        padding: 1.1rem 1rem 0.2rem;
        min-height: 192px;
    }

    .feature-title {
        color: var(--text);
        font-size: 1.06rem;
        font-weight: 750;
        margin-top: 0.45rem;
    }

    .feature-icon {
        width: 2.65rem;
        height: 2.65rem;
        display: grid;
        place-items: center;
        border-radius: 14px;
        background: rgba(57, 255, 136, 0.08);
        border: 1px solid rgba(57, 255, 136, 0.2);
        color: var(--green);
        font-size: 1.2rem;
        box-shadow: inset 0 0 0 1px rgba(57, 255, 136, 0.04);
    }

    .process-shell {
        padding: 0.1rem 0 0;
    }

    .process-intro {
        margin-bottom: 1rem;
        padding: 0 0.1rem;
    }

    .control-group-title {
        color: var(--text);
        font-size: 0.96rem;
        font-weight: 700;
        margin: 0.2rem 0 0.2rem;
    }

    .control-group-note {
        color: var(--text-soft);
        font-size: 0.88rem;
        margin-bottom: 0.7rem;
    }

    .upload-hint {
        padding: 1rem 1rem 0.7rem;
        border-radius: 18px;
        border: 1px dashed rgba(57, 255, 136, 0.32);
        background: linear-gradient(180deg, rgba(20, 28, 23, 0.88), rgba(10, 15, 11, 0.9));
        margin-bottom: 0.8rem;
    }

    .upload-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .upload-copy {
        color: var(--text-soft);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .helper-pill-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 0.2rem;
    }

    .helper-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.34rem 0.72rem;
        border: 1px solid rgba(30, 43, 35, 0.95);
        background: rgba(17, 26, 20, 0.95);
        color: var(--text-soft);
        font-size: 0.8rem;
    }

    .result-panel {
        padding: 1.15rem 1.2rem;
        word-break: break-word;
    }

    .item-card {
        padding: 0.95rem 1rem;
        margin: 0.7rem 0;
    }

    .kpi-card {
        padding: 1rem 1.05rem;
        min-height: 108px;
        margin-bottom: 0.8rem;
    }

    .kpi-value {
        color: var(--text);
        font-size: clamp(1.06rem, 1.9vw, 1.38rem);
        font-weight: 780;
        line-height: 1.2;
        margin-top: 0.55rem;
        word-break: break-word;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        border-radius: 999px;
        padding: 0.42rem 0.76rem;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(17, 26, 20, 0.95);
        border: 1px solid rgba(30, 43, 35, 0.95);
        color: var(--text-soft);
    }

    .status-pill.ready,
    .status-pill.completed {
        color: var(--green);
        border-color: rgba(57, 255, 136, 0.3);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
    }

    .status-pill.processing {
        color: #9ec2ff;
        border-color: rgba(79, 124, 255, 0.28);
        box-shadow: 0 0 18px rgba(79, 124, 255, 0.08);
    }

    .status-pill.cached {
        color: #b9c7ff;
        border-color: rgba(139, 92, 246, 0.26);
    }

    .status-pill.error {
        color: var(--danger);
        border-color: rgba(255, 90, 95, 0.28);
    }

    .processing-shell {
        position: relative;
        padding: 1.1rem 1.15rem 1.15rem;
        overflow: hidden;
    }

    .processing-summary {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .processing-title {
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 760;
        letter-spacing: -0.02em;
    }

    .processing-subtitle {
        color: var(--text-soft);
        margin-top: 0.35rem;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .processing-stat {
        min-width: 9rem;
        text-align: right;
        color: var(--text);
        font-weight: 700;
    }

    .processing-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 1rem;
    }

    .stage-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        border: 1px solid rgba(30, 43, 35, 0.95);
        background: rgba(10, 15, 11, 0.78);
    }

    .stage-row.running {
        border-color: rgba(57, 255, 136, 0.34);
        box-shadow: 0 0 20px rgba(57, 255, 136, 0.08);
    }

    .stage-row.completed,
    .stage-row.cached,
    .stage-row.skipped {
        border-color: rgba(57, 255, 136, 0.22);
    }

    .stage-row.failed {
        border-color: rgba(255, 90, 95, 0.22);
    }

    .stage-index {
        width: 2rem;
        height: 2rem;
        display: grid;
        place-items: center;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--text);
        background: rgba(17, 26, 20, 0.95);
        border: 1px solid rgba(30, 43, 35, 0.95);
        flex-shrink: 0;
    }

    .stage-row.running .stage-index,
    .stage-row.completed .stage-index,
    .stage-row.cached .stage-index {
        border-color: rgba(57, 255, 136, 0.34);
        color: var(--green);
    }

    .stage-copy { min-width: 0; flex: 1; }

    .stage-name {
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 700;
    }

    .stage-message {
        color: var(--text-soft);
        font-size: 0.8rem;
        margin-top: 0.18rem;
        line-height: 1.5;
        word-break: break-word;
    }

    .stage-state {
        color: var(--text-muted);
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        flex-shrink: 0;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.9rem 0 1rem;
    }

    .info-grid.two-up {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .info-card {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(30, 43, 35, 0.95);
        background: rgba(17, 26, 20, 0.92);
    }

    .info-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
    }

    .info-value {
        color: var(--text);
        font-size: 1rem;
        font-weight: 720;
        margin-top: 0.45rem;
        line-height: 1.45;
        word-break: break-word;
    }

    .sidebar-section-label {
        color: var(--text-soft);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .ai-model-status-card {
        position: relative;
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(17, 26, 20, 0.96), rgba(10, 15, 11, 0.96));
        border: 1px solid rgba(57, 255, 136, 0.22);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
        margin-bottom: 0.55rem;
    }

    .ai-model-active-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--green);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .ai-model-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 10px rgba(57, 255, 136, 0.6);
    }

    .ai-model-meta {
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        color: var(--text-soft);
        font-size: 0.82rem;
        line-height: 1.5;
    }

    .ai-model-meta b {
        color: var(--text);
        font-weight: 720;
        word-break: break-word;
        text-align: right;
    }

    .sidebar-brand {
        color: var(--text);
        font-size: 1.24rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-top: 0.25rem;
    }

    .sidebar-nav-active,
    .sidebar-shortcut-card {
        position: relative;
        padding: 0.88rem 0.95rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(17, 26, 20, 0.96), rgba(10, 15, 11, 0.96));
        border: 1px solid rgba(57, 255, 136, 0.22);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
        margin-bottom: 0.55rem;
        overflow: hidden;
    }

    .sidebar-nav-active::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.7rem;
        bottom: 0.7rem;
        width: 4px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 14px rgba(57, 255, 136, 0.35);
    }

    .sidebar-nav-title,
    .sidebar-shortcut-title {
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 720;
        margin-left: 0.7rem;
    }

    .sidebar-nav-meta,
    .sidebar-shortcut-meta {
        color: var(--text-soft);
        font-size: 0.8rem;
        margin-top: 0.32rem;
        margin-left: 0.7rem;
        line-height: 1.5;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        padding: 0.78rem 0.95rem;
        border-radius: 16px;
        border: 1px solid rgba(30, 43, 35, 0.95);
        background: rgba(13, 20, 16, 0.84);
        color: var(--text-soft);
        box-shadow: none;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(57, 255, 136, 0.32);
        color: var(--text);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.8rem;
        border-radius: 14px;
        border: 1px solid rgba(30, 43, 35, 0.95);
        background: linear-gradient(180deg, rgba(17, 26, 20, 0.96), rgba(10, 15, 11, 0.96));
        color: var(--text);
        font-weight: 700;
        transition: all 0.18s ease;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(57, 255, 136, 0.34);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.09);
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        color: #031109;
        border-color: rgba(57, 255, 136, 0.4);
        background: linear-gradient(135deg, #00C853 0%, #39FF88 100%);
        box-shadow: 0 0 22px rgba(57, 255, 136, 0.18);
    }

    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        filter: brightness(1.04);
        box-shadow: 0 0 28px rgba(57, 255, 136, 0.24);
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    [data-baseweb="select"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(20, 28, 23, 0.96) !important;
        color: var(--text) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(30, 43, 35, 0.95) !important;
        box-shadow: none !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus,
    [data-baseweb="select"] *:focus,
    [data-testid="stFileUploaderDropzone"]:focus-within {
        border-color: rgba(57, 255, 136, 0.4) !important;
        box-shadow: 0 0 0 1px rgba(57, 255, 136, 0.16), 0 0 18px rgba(57, 255, 136, 0.08) !important;
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stRadio label,
    .stFileUploader label,
    .stMarkdown,
    p,
    li,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    span,
    div {
        word-break: break-word;
    }

    [data-testid="stFileUploader"] {
        padding: 0.2rem;
        border-radius: 18px;
        background: rgba(10, 15, 11, 0.74);
        border: 1px dashed rgba(57, 255, 136, 0.24);
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stCaptionContainer"],
    .stCaption {
        color: var(--text-muted) !important;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.55rem;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid rgba(30, 43, 35, 0.95);
        overflow-x: auto;
    }

    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 999px;
        padding: 0.72rem 1rem;
        color: var(--text-soft);
        border: 1px solid rgba(30, 43, 35, 0.95);
        background: rgba(17, 26, 20, 0.95);
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--green);
        border-color: rgba(57, 255, 136, 0.32);
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
    }

    [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.7rem;
        flex-wrap: wrap;
    }

    [data-testid="stRadio"] label {
        border-radius: 14px !important;
        border: 1px solid rgba(30, 43, 35, 0.95) !important;
        background: rgba(17, 26, 20, 0.95) !important;
        padding: 0.55rem 0.9rem !important;
        margin-right: 0 !important;
        color: var(--text-soft) !important;
        transition: all 0.16s ease;
    }

    [data-testid="stRadio"] label:hover {
        border-color: rgba(57, 255, 136, 0.28) !important;
        box-shadow: 0 0 16px rgba(57, 255, 136, 0.07);
    }

    [data-testid="stRadio"] label:has(input:checked) {
        border-color: rgba(57, 255, 136, 0.34) !important;
        background: rgba(57, 255, 136, 0.08) !important;
        color: var(--text) !important;
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
    }

    [data-testid="stExpander"] {
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        padding: 1rem;
        min-height: 5.3rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-soft) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    [data-testid="stChatMessage"] {
        background: rgba(13, 20, 16, 0.92);
        border: 1px solid rgba(30, 43, 35, 0.95);
        border-radius: 18px;
        padding: 0.4rem 0.8rem;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] textarea {
        background: rgba(20, 28, 23, 0.96) !important;
        color: var(--text) !important;
        border: 1px solid rgba(30, 43, 35, 0.95) !important;
    }

    .stProgress > div > div,
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #00C853, #39FF88) !important;
    }

    .footer-shell {
        margin-top: 2.7rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(30, 43, 35, 0.95);
        color: var(--text-muted);
        font-size: 0.8rem;
        text-align: center;
    }

    @media (max-width: 1180px) {
        .top-header {
            flex-direction: column;
        }
        .industry-hero,
        .industry-orbit {
            min-height: auto;
        }
        .orbit-ring,
        .orbit-ring-2,
        .orbit-ring-3,
        .hero-core,
        .industry-card {
            position: static;
            transform: none;
            width: auto;
            height: auto;
        }
        .industry-orbit {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }
        .hero-core {
            order: -1;
            margin-bottom: 0.5rem;
        }
        .processing-grid,
        .info-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 1rem 0.85rem 3rem;
        }
        .page-banner {
            flex-direction: column;
            align-items: flex-start;
        }
        .page-banner-badge {
            min-width: 0;
            width: 100%;
            justify-content: flex-start;
        }
        .industry-orbit {
            grid-template-columns: 1fr;
        }
        .hero-core h2 {
            font-size: 2rem;
        }
    }

    .dashboard-layout {
        position: relative;
        margin-top: 0.35rem;
    }

    .dashboard-layout [data-testid="stHorizontalBlock"] {
        align-items: stretch;
        gap: 1.25rem;
    }

    .dashboard-layout > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) {
        order: 2;
    }

    .dashboard-layout > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
        order: 1;
    }

    .dashboard-layout > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) {
        order: 3;
    }

    .home-side-column {
        display: flex;
        flex-direction: column;
        gap: 1.05rem;
        padding-top: 2.2rem;
    }

    .home-side-column.right-stack {
        padding-top: 4.3rem;
    }

    .home-feature-wrap {
        position: relative;
        overflow: visible;
    }

    .home-feature-card {
        position: relative;
        overflow: visible;
        padding: 1.05rem 1.05rem 0.95rem;
        min-height: 182px;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(7, 17, 13, 0.96), rgba(9, 20, 15, 0.96));
        border: 1px solid rgba(60, 255, 160, 0.16);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.34);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }

    .home-feature-card:hover {
        transform: translateY(-3px);
        border-color: rgba(57, 255, 136, 0.34);
        box-shadow: 0 24px 54px rgba(0, 0, 0, 0.38), 0 0 22px rgba(57, 255, 136, 0.08);
    }

    .home-feature-card.left::after,
    .home-feature-card.right::before {
        content: "";
        position: absolute;
        top: 50%;
        width: 6.3rem;
        height: 1px;
        background: linear-gradient(90deg, rgba(57, 255, 136, 0.42), rgba(57, 255, 136, 0.04));
        box-shadow: 0 0 10px rgba(57, 255, 136, 0.12);
        pointer-events: none;
    }

    .home-feature-card.left::after {
        right: -6.1rem;
    }

    .home-feature-card.right::before {
        left: -6.1rem;
        background: linear-gradient(270deg, rgba(57, 255, 136, 0.42), rgba(57, 255, 136, 0.04));
    }

    .home-feature-card.left .connector-node,
    .home-feature-card.right .connector-node {
        position: absolute;
        top: calc(50% - 0.38rem);
        width: 0.76rem;
        height: 0.76rem;
        border-radius: 999px;
        background: #39FF88;
        box-shadow: 0 0 0 4px rgba(57, 255, 136, 0.06), 0 0 14px rgba(57, 255, 136, 0.28);
        animation: dashboardPulse 2.8s ease-in-out infinite;
        pointer-events: none;
    }

    .home-feature-card.left .connector-node {
        right: -6.48rem;
    }

    .home-feature-card.right .connector-node {
        left: -6.48rem;
    }

    .feature-icon-shell {
        width: 3rem;
        height: 3rem;
        display: grid;
        place-items: center;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(6, 11, 8, 0.96), rgba(11, 22, 16, 0.96));
        border: 1px solid rgba(57, 255, 136, 0.28);
        color: #39FF88;
        box-shadow: inset 0 0 0 1px rgba(57, 255, 136, 0.04), 0 0 16px rgba(57, 255, 136, 0.08);
        font-size: 1.28rem;
    }

    .home-feature-title {
        color: var(--text);
        font-size: 1.12rem;
        font-weight: 760;
        letter-spacing: -0.02em;
        margin-top: 0.7rem;
    }

    .home-feature-desc {
        color: #9BA9A2;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.42rem;
        min-height: 3rem;
    }

    .dashboard-center-shell {
        position: relative;
        max-width: 520px;
        margin: 0 auto;
        overflow: visible;
    }

    .central-process-panel {
        position: relative;
        overflow: visible;
        padding: 1.2rem 1.2rem 1.05rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, rgba(7, 17, 13, 0.98), rgba(9, 20, 15, 0.98));
        border: 1px solid rgba(57, 255, 136, 0.26);
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.42), 0 0 28px rgba(57, 255, 136, 0.07);
        z-index: 2;
    }

    .central-process-panel::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 24px;
        pointer-events: none;
        background: linear-gradient(135deg, rgba(57,255,136,0.05), transparent 45%, rgba(79,124,255,0.05));
    }

    .central-rings {
        position: absolute;
        inset: -5.5rem -7.5rem;
        pointer-events: none;
        z-index: 0;
    }

    .central-rings::before,
    .central-rings::after,
    .central-rings span {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        border-radius: 999px;
        transform: translate(-50%, -50%);
        border: 1px solid rgba(57, 255, 136, 0.10);
    }

    .central-rings::before {
        width: 25rem;
        height: 25rem;
    }

    .central-rings::after {
        width: 33rem;
        height: 33rem;
        border-color: rgba(39, 199, 118, 0.08);
    }

    .central-rings span {
        width: 41rem;
        height: 41rem;
        border-color: rgba(79, 124, 255, 0.08);
    }

    .central-process-header {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.9rem;
        padding-bottom: 0.95rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(60, 255, 160, 0.12);
    }

    .central-process-eyebrow {
        color: #39FF88;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .central-process-title {
        color: #F2F5F3;
        font-size: 1.46rem;
        font-weight: 780;
        letter-spacing: -0.03em;
        margin-top: 0.35rem;
    }

    .central-process-subtitle {
        color: #9BA9A2;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-top: 0.42rem;
        max-width: 21rem;
    }

    .pipeline-ready-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.48rem;
        white-space: nowrap;
        border-radius: 999px;
        padding: 0.42rem 0.76rem;
        background: rgba(7, 17, 13, 0.95);
        border: 1px solid rgba(57, 255, 136, 0.22);
        color: #39FF88;
        font-size: 0.8rem;
        font-weight: 700;
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.08);
    }

    .pipeline-ready-badge .dot {
        width: 0.52rem;
        height: 0.52rem;
        border-radius: 999px;
        background: #39FF88;
        box-shadow: 0 0 12px rgba(57, 255, 136, 0.35);
    }

    .translation-legend {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-bottom: 0.8rem;
    }

    .translation-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(60, 255, 160, 0.12);
        background: rgba(9, 20, 15, 0.9);
        color: #9BA9A2;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .translation-chip .chip-dot {
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 999px;
        background: #64748B;
        box-shadow: 0 0 10px rgba(100, 116, 139, 0.18);
    }

    .translation-chip.active-enabled {
        color: #39FF88;
        border-color: rgba(57, 255, 136, 0.22);
        background: rgba(57, 255, 136, 0.08);
    }

    .translation-chip.active-enabled .chip-dot {
        background: #39FF88;
        box-shadow: 0 0 12px rgba(57, 255, 136, 0.25);
    }

    .translation-chip.active-disabled {
        color: #FF7A84;
        border-color: rgba(255, 90, 95, 0.20);
        background: rgba(255, 90, 95, 0.08);
    }

    .translation-chip.active-disabled .chip-dot {
        background: #FF5A5F;
        box-shadow: 0 0 12px rgba(255, 90, 95, 0.24);
    }

    .dashboard-active-note {
        margin-top: 1rem;
        padding: 0.85rem 0.95rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(7, 17, 13, 0.9), rgba(9, 20, 15, 0.9));
        border: 1px solid rgba(60, 255, 160, 0.14);
        color: #9BA9A2;
        text-align: center;
    }

    @keyframes dashboardPulse {
        0%, 100% { transform: scale(1); opacity: 0.95; }
        50% { transform: scale(1.14); opacity: 1; }
    }

    @media (max-width: 1180px) {
        .home-feature-card.left::after,
        .home-feature-card.right::before,
        .home-feature-card .connector-node,
        .central-rings {
            display: none;
        }

        .dashboard-layout > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1),
        .dashboard-layout > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2),
        .dashboard-layout > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) {
            order: initial;
        }

        .home-side-column,
        .home-side-column.right-stack {
            padding-top: 0;
        }
    }

    @media (max-width: 760px) {
        .dashboard-center-shell {
            max-width: 100%;
        }

        .central-process-header {
            flex-direction: column;
            align-items: flex-start;
        }
    }

    </style>
    """, unsafe_allow_html=True)

def initialize_state() -> None:
    try:
        recover_stale_processing_videos()
    except Exception:
        pass

    defaults = {
        "result": None,
        "processed": False,
        "chat_history": [],
        "last_source": None,
        "last_language": None,
        "complete_pdf_data": None,
        "rag_video_id": None,
        "app_page": "🏠 Home",
        "ai_model_settings": {},
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


def _exception_messages(exc: BaseException) -> list[str]:
    """Collect the visible messages from an exception chain."""
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__

    return messages


def classify_llm_error(exc: BaseException) -> str | None:
    """Return a user-facing LLM/provider error category, if any."""
    text = "\n".join(_exception_messages(exc)).lower()
    if not text:
        return None

    authentication_markers = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "authentication",
        "auth failure",
        "invalid_api_key",
        "expired_api_key",
        "invalid api key",
        "expired api key",
        "api key",
        "invalid key",
        "credential",
        "credentials",
        "permission denied",
        "access denied",
    )
    rate_limit_markers = (
        "429",
        "rate limit",
        "too many requests",
        "quota",
        "capacity",
        "credits exhausted",
        "resource exhausted",
    )
    timeout_network_markers = (
        "timeout",
        "timed out",
        "time-out",
        "connection error",
        "connection aborted",
        "connection reset",
        "connection refused",
        "network error",
        "network unreachable",
        "dns",
        "temporary failure in name resolution",
        "name resolution",
        "max retries exceeded",
        "read timeout",
        "connect timeout",
    )
    provider_unavailable_markers = (
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "provider unavailable",
        "server error",
        "upstream",
        "model unavailable",
        "overloaded",
    )
    generic_llm_markers = (
        "llm",
        "model",
        "openai",
        "anthropic",
        "mistral",
        "groq",
        "gemini",
        "summarization failed",
        "translation failed",
        "question answering failed",
        "provider response",
        "malformed response",
        "invalid response",
    )

    if any(marker in text for marker in authentication_markers):
        return "authentication"
    if any(marker in text for marker in rate_limit_markers):
        return "rate_limit"
    if any(marker in text for marker in timeout_network_markers):
        return "timeout_network"
    if any(marker in text for marker in provider_unavailable_markers):
        return "provider_unavailable"
    if any(marker in text for marker in generic_llm_markers):
        return "generic_llm"

    return None


def friendly_llm_error(exc: BaseException) -> str | None:
    """Return a short safe UI message for LLM/provider failures."""
    category = classify_llm_error(exc)
    if category == "authentication":
        return (
            "⚠️ AI service configuration issue.\n\n"
            "The AI model could not be reached because the API credentials\n"
            "may be expired or invalid. Please update the API key and try again."
        )
    if category == "rate_limit":
        return (
            "⚠️ AI service is temporarily busy.\n\n"
            "Please wait a moment and try again."
        )
    if category in {"timeout_network", "provider_unavailable"}:
        return (
            "⚠️ AI service could not be reached.\n\n"
            "Please check your connection and try again."
        )
    if category == "generic_llm":
        return (
            "⚠️ AI processing could not be completed.\n\n"
            "Please try again. If the problem continues, check your AI provider settings."
        )
    return None


GENERIC_SAFE_UI_ERROR = (
    "⚠️ Something went wrong while processing your request.\n\n"
    "Please try again. If the problem continues, check your application settings."
)


def _safe_fallback_message(
    fallback_message: str,
    exc: BaseException,
) -> str:
    """Return a caller-supplied fallback only when it is clearly safe for UI."""
    text = str(fallback_message or "").strip()
    if not text:
        return GENERIC_SAFE_UI_ERROR

    normalized = text.lower()
    exception_messages = [message.strip().lower() for message in _exception_messages(exc)]
    suspicious_markers = (
        "http://",
        "https://",
        "api key",
        "api_key",
        "secret",
        "token",
        "bearer ",
        "traceback",
        "stack trace",
        '{"',
        "{'",
        '"error"',
        "'/",
        'file "',
    )

    if any(marker in normalized for marker in suspicious_markers):
        return GENERIC_SAFE_UI_ERROR
    if re.search(r"[a-z]:\\", normalized):
        return GENERIC_SAFE_UI_ERROR
    if re.search(r"/(tmp|var|home|users?|private|etc|appdata|mnt)/", normalized):
        return GENERIC_SAFE_UI_ERROR
    if any(exception_message and exception_message == normalized for exception_message in exception_messages):
        return GENERIC_SAFE_UI_ERROR

    return text


def render_exception_feedback(
    exc: BaseException,
    *,
    fallback_message: str,
    technical_label: str = "Technical details",
) -> str:
    """Render a safe error for users while preserving technical logs."""
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    friendly_message = friendly_llm_error(exc)
    shown_message = friendly_message or _safe_fallback_message(
        fallback_message,
        exc,
    )
    st.error(shown_message)
    return shown_message


def render_safe_ui_message(message: str, *, level: str = "error") -> None:
    """Render a user-safe Streamlit status message without exception details."""
    renderer = getattr(st, level, st.error)
    renderer(message)


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


def get_result_video_id(result: dict[str, Any] | None) -> str | None:
    """Return the persisted ID that owns a result's RAG chain, if present."""
    result = result or {}
    metadata = result.get("metadata") or {}
    source_id = (
        result.get("video_id")
        or result.get("meeting_id")
        or result.get("source_id")
        or metadata.get("video_id")
        or metadata.get("meeting_id")
        or metadata.get("source_id")
    )
    return str(source_id) if source_id else None


def get_owned_rag_chain(result: dict[str, Any] | None) -> Any | None:
    """Return a RAG chain only when its session owner matches the active result."""
    result = result or {}
    result_id = get_result_video_id(result)
    rag_id = str(st.session_state.get("rag_video_id") or "")
    chain = result.get("rag_chain")
    if chain is None:
        return None
    if not result_id or not rag_id or result_id != rag_id:
        result["rag_chain"] = None
        st.session_state["rag_video_id"] = None
        return None
    return chain


def invalidate_rag_state(result: dict[str, Any] | None = None) -> None:
    """Detach any RAG chain that is no longer proven to belong to the active source."""
    if result is not None:
        result["rag_chain"] = None
    st.session_state["rag_video_id"] = None


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

    # A result from another source must never carry the previous source's RAG
    # object into the active session. New sources load RAG lazily on demand.
    if not same_source:
        invalidate_rag_state(normalized_result)

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
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def status_markup(label: str, state: str = "waiting") -> str:
    """Return one consistent, accessible status presentation for UI-only use."""
    icons = {"ready": "●", "processing": "◉", "waiting": "○", "completed": "✓", "cached": "♻", "unavailable": "—", "error": "⚠"}
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
                f'<div class="muted">{escape(" · ".join(metadata))}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="item-card"><div class="item-number">Item {index}</div>'
                f'<div>{escape(str(item))}</div></div>',
                unsafe_allow_html=True,
            )


NAV_ITEMS = [
    {
        "key": "🏠 Home",
        "icon": "🏠",
        "label": "Home",
        "description": "Workspace overview",
    },
    {
        "key": "🎙 Transcript",
        "icon": "📄",
        "label": "Transcript",
        "description": "Original meeting transcript",
    },
    {
        "key": "🌐 Roman Urdu",
        "icon": "🌐",
        "label": "Roman Urdu",
        "description": "Translation workspace",
    },
    {
        "key": "📊 Summary & Analysis",
        "icon": "📊",
        "label": "Summary & Analysis",
        "description": "AI insights and decisions",
    },
    {
        "key": "📚 Saved History",
        "icon": "🔖",
        "label": "Saved History",
        "description": "Stored Roman Urdu history",
    },
    {
        "key": "📚 Video Library",
        "icon": "🎬",
        "label": "Video Library",
        "description": "Processed videos and reopen",
    },
    {
        "key": "📥 Export",
        "icon": "⬇",
        "label": "Export",
        "description": "Download structured outputs",
    },
]

HOME_FEATURES = [
    ("🎙", "View Transcript", "Inspect the original transcript for the active source.", "🎙 Transcript"),
    ("🌐", "Roman Urdu", "Generate or review the Roman Urdu version.", "🌐 Roman Urdu"),
    ("📊", "Summary & Analysis", "Open summaries, actions, decisions, and questions.", "📊 Summary & Analysis"),
    ("🔖", "Saved History", "Load previously saved Roman Urdu translations.", "📚 Saved History"),
    ("🎬", "Video Library", "Browse processed meetings and reopen any record.", "📚 Video Library"),
    ("⬇", "Export", "Download TXT, Excel, and PDF deliverables.", "📥 Export"),
]

INDUSTRY_USE_CASES = [
    ("Fintech", "Risk reviews · board calls"),
    ("Logistics", "Operations syncs · vendor calls"),
    ("Hospitality", "Guest experience · property updates"),
    ("Real Estate", "Client briefings · sales follow-ups"),
    ("Education", "Lectures · committee meetings"),
    ("Media & Entertainment", "Production reviews · content planning"),
    ("Manufacturing", "Plant updates · safety reviews"),
    ("Retail & E-commerce", "Campaign reviews · CX analysis"),
    ("Healthcare", "Clinical briefings · admin meetings"),
]


def render_page_banner(
    kicker: str,
    title: str,
    subtitle: str,
    badge_html: str | None = None,
) -> None:
    badge = f'<div class="page-banner-badge">{badge_html}</div>' if badge_html else ""
    st.markdown(
        f'<div class="page-banner">'
        f'<div><div class="page-banner-kicker">{escape(kicker)}</div>'
        f'<div class="page-banner-title">{escape(title)}</div>'
        f'<div class="page-banner-subtitle">{escape(subtitle)}</div></div>'
        f'{badge}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_home_industry_hero() -> None:
    cards = []
    for index, (title, description) in enumerate(INDUSTRY_USE_CASES, start=1):
        cards.append(
            f'<div class="industry-card pos-{index}">'
            f'<div class="industry-title">{escape(title)}</div>'
            f'<div class="industry-desc">{escape(description)}</div>'
            f'</div>'
        )
    hero_html = (
        '<div class="industry-hero">'
        '<div class="industry-orbit">'
        '<div class="orbit-ring"></div>'
        '<div class="orbit-ring-2"></div>'
        '<div class="orbit-ring-3"></div>'
        '<div class="hero-core">'
        '<div class="orbit-eyebrow">AI industry intelligence</div>'
        '<h2>AI App Use Cases<br>Across Industries</h2>'
        '<div class="hero-core-copy">A premium AI workspace for transforming meetings and videos into structured, searchable knowledge.</div>'
        '</div>'
        + ''.join(cards) +
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header, sidebar, and input
# -----------------------------------------------------------------------------


def render_dashboard_feature_card(
    *,
    icon: str,
    title: str,
    description: str,
    page: str,
    side: str,
    index: int,
) -> None:
    st.markdown(
        f'<div class="home-feature-wrap">'
        f'<div class="home-feature-card {escape(side)}">'
        f'<div class="feature-icon-shell">{escape(icon)}</div>'
        f'<div class="feature-label">Open a feature</div>'
        f'<div class="home-feature-title">{escape(title)}</div>'
        f'<div class="home-feature-desc">{escape(description)}</div>'
        f'<span class="connector-node"></span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        f'Open {title}',
        key=f'command_center_feature_{side}_{index}_{page}',
        use_container_width=True,
    ):
        navigate_to(page)
        st.rerun()


def render_header() -> None:
    """Render the premium application header and active workspace context."""
    result = st.session_state.get("result") or {}
    source_name = st.session_state.get("active_translation_source") or result.get("source_name") or result.get("title")
    full_context = str(source_name or "No active meeting selected")
    status = status_markup("Pipeline ready", "ready") if st.session_state.get("processed") else status_markup("Waiting for input", "waiting")

    left_col, right_col = st.columns([1.7, 1.05])
    with left_col:
        st.markdown(
            '<div class="top-header-card">'
            '<div class="brand-row">'
            '<div class="brand-logo">✦</div>'
            '<div>'
            '<div class="brand-kicker">AI Meeting Intelligence</div>'
            '<h1 class="brand-title">AI Meeting Intelligence</h1>'
            '<div class="brand-subtitle">Turn meetings and videos into structured, searchable knowledge.</div>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            '<div class="workspace-panel">'
            '<div class="workspace-chip">VIDEO AGENT WORKSPACE</div>'
            '<div class="workspace-title">AI Meeting Assistant</div>'
            '<div class="workspace-subtitle">Professional meeting and video analysis</div>'
            f'<div class="context-panel"><div><div class="context-label">Active source</div><div class="context-value">{escape(compact_label(full_context, 58))}</div></div><div>{status}</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button('+ New Session', key='header_new_session', use_container_width=True):
            reset_session()
            st.rerun()

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
            st.session_state["processed_video_selector"] = f"[{source_type}] {title} · {video_id[:8]}"
            return


def _delete_video_safely(video_id: str) -> DeleteResult:
    """Delete a video's vector collection and SQLite record.

    Ordering is deliberate: the Chroma collection must be confirmed gone
    (deleted, or already absent which is safe/idempotent) BEFORE the
    SQLite record is removed. A real Chroma failure raises
    ``VectorStoreDeleteError`` and leaves the SQLite record intact, so a
    failed delete can never produce an orphan vector collection.

    Session/RAG state cleanup is intentionally NOT handled here; callers
    run their existing ownership checks (``rag_video_id`` / active
    metadata) so another video's valid RAG chain is never invalidated.
    """
    result = delete_vector_store(video_id)
    delete_video(video_id)
    return result


def render_processed_videos() -> None:
    """Show completed videos from SQLite and let the user reopen them."""
    try:
        videos = _list_all_completed_videos()
    except Exception as exc:
        render_exception_feedback(
            exc,
            fallback_message="Could not load processed videos.",
        )
        return

    with st.expander("📚 Processed Videos", expanded=False):
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
            label = f"[{s_type}] {title} · {v_id[:8]}"
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
                        "⚡ This video's RAG is already loaded in this session. "
                        "No vector store reload or embedding initialization."
                    )
                else:
                    _load_video_library_entry_with_notice(selected)

        delete_button_key = f"delete_processed_{video_id}"
        if st.button(
            "🗑️ Delete selected source",
            use_container_width=True,
            type="secondary",
            key=delete_button_key,
        ):
            if not video_id:
                st.warning("No source ID available to delete.")
            else:
                try:
                    result = _delete_video_safely(video_id)

                    if st.session_state.get("rag_video_id") == video_id:
                        reset_session()

                    detail = (
                        "vector collection removed"
                        if result.existed
                        else "no vector collection present"
                    )
                    st.success(f"Deleted selected source: {video_id} ({detail})")
                    st.rerun()
                except Exception as exc:
                    render_exception_feedback(
                        exc,
                        fallback_message="Could not delete the selected source.",
                    )


def render_translation_history() -> None:
    """Browse, select, load, and safely delete saved Roman Urdu translations."""
    try:
        total_saved = get_translation_count()
        st.markdown('<div class="section-title">📚 Saved Roman Urdu Translations</div>', unsafe_allow_html=True)
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
            f"🎬 {display_name(record)} — {record.get('updated_at') or record.get('created_at') or 'Unknown date'} · #{record['id']}": record["id"]
            for record in records
        }
        selected_label = st.selectbox(
            "🎬 Select Saved Video or Translation",
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
                "♻️ Load This Translation",
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
                st.success("⚡ Saved Roman Urdu translation loaded.")

        pending_delete_id = st.session_state.get("pending_delete_translation_id")
        with delete_col:
            if st.button(
                "🗑️ Delete Translation",
                key=f"delete_saved_translation_{selected_id}",
                use_container_width=True,
            ):
                st.session_state.pending_delete_translation_id = selected_id
                st.rerun()

        if pending_delete_id == selected_id:
            with st.container(border=True):
                st.warning(f"⚠️ Are you sure you want to delete: {selected_name}?")
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
            st.markdown("### 🟢 Selected Content")
            st.write(f"🎬 **Source:** {selected_name}")
            st.write(f"📅 **Last Updated:** {selected.get('updated_at') or 'Unknown'}")
            st.write(f"🤖 **Model:** {selected.get('model_name') or 'Model not recorded'}")
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("📝 Transcript Length", f"{len(str(selected.get('original_transcript') or '')):,} characters")
            stat_col2.metric("🌐 Translation Length", f"{len(str(selected.get('roman_urdu_translation') or '')):,} characters")

        with st.expander("▶ 📝 Original Transcript Preview", expanded=False):
            st.text(str(selected.get("original_transcript") or ""))
        with st.expander("▶ 🌐 Roman Urdu Translation Preview", expanded=False):
            st.text(str(selected.get("roman_urdu_translation") or ""))
    except Exception as error:
        render_exception_feedback(
            error,
            fallback_message="Could not load saved Roman Urdu translations.",
        )


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
    # silently drop the in-memory chain object. Reuse it instead — this is
    # the same identity/validity check already used by the sidebar
    # "Open / Ask Questions" reuse guard, just applied inside the shared
    # loader so every entry point benefits from it.
    previous_result = st.session_state.get("result") or {}
    previous_metadata = previous_result.get("metadata") or {}
    reuse_existing_rag = bool(
        video_id
        and str(previous_metadata.get("video_id") or "") == video_id
        and str(st.session_state.get("rag_video_id") or "") == video_id
        and get_owned_rag_chain(previous_result) is not None
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
    # question — unless this exact video's RAG chain was already valid in
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
    if sort_order in {"Name A → Z", "Name Z → A"}:
        return sorted(
            sorted_entries,
            key=lambda entry: get_result_source_name(entry).casefold(),
            reverse=sort_order == "Name Z → A",
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
    navigate_to("📥 Export")


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
            st.markdown("<div class=\"library-active\">● Active content</div>", unsafe_allow_html=True)
        st.markdown(f"<div class=\"library-card-title\">🎬 {escape(source_name)}</div>", unsafe_allow_html=True)
        if parsed_date is not None:
            st.markdown(f"<div class=\"library-card-meta\">📅 Processed: {parsed_date.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class=\"library-card-meta\">📅 Date unavailable</div>", unsafe_allow_html=True)

        source_type = entry.get("source_type")
        if source_type:
            st.markdown(f"<div class=\"library-card-meta\">Source type: {escape(str(source_type))}</div>", unsafe_allow_html=True)
        transcript = str(entry.get("_library_transcript") or "")
        if transcript:
            st.markdown(f"<div class=\"library-card-meta\">Transcript: {len(transcript):,} characters · {len(transcript.split()):,} words</div>", unsafe_allow_html=True)

        indicator_columns = st.columns(3)
        indicator_labels = {
            "Transcript": "🎙 Transcript",
            "Roman Urdu": "🌐 Roman Urdu",
            "Summary": "📊 Summary",
            "Action Items": "✅ Action Items",
            "Decisions": "📌 Decisions",
            "Questions": "❓ Questions",
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
                        result = _delete_video_safely(video_id)
                        active_result = st.session_state.get("result") or {}
                        active_metadata = active_result.get("metadata") or {}
                        active_source = st.session_state.get("active_translation_source")
                        if (
                            str(active_metadata.get("video_id") or "") == video_id
                            or (not active_metadata.get("video_id") and active_source == source_name)
                        ):
                            reset_session()
                        st.session_state.pop("video_library_pending_delete_id", None)
                        detail = (
                            "vector collection removed"
                            if result.existed
                            else "no vector collection present"
                        )
                        st.success(f"Deleted {source_name}. ({detail})")
                        st.rerun()
                    except Exception as exc:
                        render_exception_feedback(
                            exc,
                            fallback_message="Could not delete the selected video library item.",
                        )
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

    st.markdown("### 📊 Library Analytics")
    with st.container():
        st.markdown("<div class=\"analytics-card\">", unsafe_allow_html=True)
        # Explicit rows avoid relying on CSS to make Streamlit columns wrap safely.
        overview_row_one = st.columns(2)
        overview_row_one[0].metric("🎬 Total Videos", f"{analytics['total_videos']:,}")
        overview_row_one[1].metric("📝 Total Transcript Words", f"{analytics['total_words']:,}")
        overview_row_two = st.columns(2)
        overview_row_two[0].metric("⏱ Total Duration", format_duration(analytics["duration_seconds"]))
        overview_row_two[1].metric("🌐 Roman Urdu Coverage", f"{analytics['roman_urdu_coverage']:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    if analytics["duration_seconds"] is not None and analytics["duration_video_count"] < analytics["total_videos"]:
        st.caption("Based on videos with available duration data.")
    st.caption(
        f"Roman Urdu: {analytics['roman_urdu_translated_count']} of "
        f"{analytics['roman_urdu_eligible_count']} eligible videos translated."
    )

    st.markdown("### 📈 Library Insights")
    insight_row = st.columns(2)
    most_recent = analytics["most_recent"]
    if most_recent is None:
        insight_row[0].metric("🕒 Most Recently Processed", "Date unavailable")
        insight_row[0].caption("Date unavailable")
    else:
        recent_name = str(most_recent["source_name"])
        insight_row[0].metric("🕒 Most Recently Processed", compact_label(recent_name, 24))
        insight_row[0].caption(most_recent["date"].strftime("%d %b %Y"))

    most_complete = analytics["most_complete"]
    if most_complete is None:
        insight_row[1].metric("🏆 Most Complete", "No content")
        insight_row[1].caption("No videos available")
    else:
        complete_name = str(most_complete["source_name"])
        insight_row[1].metric("🏆 Most Complete", compact_label(complete_name, 24))
        insight_row[1].caption(f"{most_complete['percentage']:.2f}% complete")

    st.metric("📈 Overall Library Completion", f"{analytics['overall_completion']:.2f}%")
    st.progress(
        min(max(analytics["overall_completion"] / 100, 0.0), 1.0),
        text="Available content across all six components",
    )

    st.markdown("#### Content Gap Insights")
    gap_row_one = st.columns(2)
    gap_row_one[0].metric("🌐 Missing Roman Urdu", f"{analytics['missing_roman_urdu']:,}")
    gap_row_one[1].metric("📊 Missing Summary", f"{analytics['missing_summary']:,}")
    gap_row_two = st.columns(1)
    gap_row_two[0].metric("⚠ Incomplete Videos", f"{analytics['incomplete_videos']:,}")

    st.markdown("#### Completion Distribution")
    distribution_entries = analytics["completion_entries"]
    for entry in distribution_entries:
        st.progress(
            min(max(entry["percentage"] / 100, 0.0), 1.0),
            text=f"{compact_label(entry['source_name'], 42)} · {entry['percentage']:.2f}%",
        )


def render_video_library_page() -> None:
    """Display the read-only analytics dashboard and saved video records."""
    st.markdown('<div class="section-title">📚 Video Library</div>', unsafe_allow_html=True)
    st.caption("Browse, filter, open, export, and safely delete saved video content.")

    try:
        videos = _list_all_completed_videos()
    except Exception as exc:
        render_exception_feedback(
            exc,
            fallback_message="Could not load the Video Library.",
        )
        return

    if not videos:
        st.markdown("### 📊 Library Analytics")
        st.info(
            "No processed videos available yet.\n\n"
            "Process your first video to start building analytics."
        )
        return

    entries = [_prepare_video_library_entry(video) for video in videos]
    _render_video_library_analytics(entries)

    st.markdown("---")
    search_query = st.text_input(
        "🔎 Search your videos...",
        placeholder="Search by source name or title...",
        key="video_library_search",
    ).strip().casefold()
    sort_order = st.selectbox(
        "Sort videos",
        ["Newest First", "Oldest First", "Name A → Z", "Name Z → A"],
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
    active_original = st.session_state.get("active_original_transcript")
    badge = status_markup("Transcript ready", "ready") if active_original else status_markup("Waiting for a processed meeting", "waiting")
    render_page_banner(
        "Document view",
        "Transcript",
        "Review the original meeting transcript for the active source.",
        badge_html=badge,
    )
    if not active_original:
        st.info("No active transcript is available. Please process or load content first.")
        return

    source_name = st.session_state.get("active_translation_source") or "Current content"
    st.markdown(
        '<div class="info-grid two-up">'
        f'<div class="info-card"><div class="info-label">Source</div><div class="info-value">{escape(source_name)}</div></div>'
        f'<div class="info-card"><div class="info-label">Transcript length</div><div class="info-value">{len(str(active_original)):,} characters</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.text_area(
        "Original Transcript",
        value=str(active_original),
        height=640,
        disabled=True,
        key=f"transcript_page_{st.session_state.get('active_translation_id') or 'current'}",
    )

def render_roman_urdu_page() -> None:
    """Display Roman Urdu controls only on the dedicated Roman Urdu page."""
    result = _current_active_result()
    active_original = st.session_state.get("active_original_transcript")
    active_translation = st.session_state.get("active_roman_urdu_translation")
    badge = status_markup("Roman Urdu ready", "ready") if active_translation else status_markup("Translation available on demand", "waiting")
    render_page_banner(
        "Translation workspace",
        "Roman Urdu",
        "Generate, review, and export the Roman Urdu version of the active transcript.",
        badge_html=badge,
    )
    if not active_original:
        st.info("No active transcript is available. Please process or load content first.")
        return

    st.markdown(
        '<div class="info-grid two-up">'
        f'<div class="info-card"><div class="info-label">Active source</div><div class="info-value">{escape(st.session_state.get("active_translation_source") or "Current content")}</div></div>'
        f'<div class="info-card"><div class="info-label">Translation state</div><div class="info-value">{"Ready" if active_translation else "Not generated yet"}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if active_translation:
        with st.expander("View Translation", expanded=True):
            st.text(str(active_translation))
        render_roman_urdu_transcript(result, show_content=False, show_exports=True)
    else:
        render_roman_urdu_transcript(result, show_content=False, show_exports=False)

def render_summary_analysis_page() -> None:
    """Display the current result in the complete AI results tab interface."""
    result = _current_active_result()
    badge = status_markup("Analysis ready", "ready") if st.session_state.get("result") else status_markup("No processed result", "waiting")
    render_page_banner(
        "AI analysis",
        "Summary & Analysis",
        "Review the structured meeting intelligence, actions, decisions, questions, and AI chat.",
        badge_html=badge,
    )
    if not st.session_state.get("result"):
        st.info("No active processed result is available. Please process or load content first.")
        return

    tabs = st.tabs([
        "Overview",
        "Summary",
        "Action Items",
        "Decisions",
        "Questions",
        "AI Chat",
        "✨ Prompt Studio",
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

_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _safe_roman_urdu_pdf_filename(title: str) -> str:
    """Build a clean, readable download filename for the Roman Urdu PDF.

    The active video/source title becomes the filename stem: invalid
    filesystem characters are removed, whitespace and repeated separators
    are collapsed into single hyphens, and the stem is length-capped so the
    file remains valid on Windows and other common OSes.  When no usable
    title is available, ``Roman-Urdu-Translation.pdf`` is returned so the
    download never ships with an empty or invalid filename.
    """
    stem = str(title or "").strip()
    for char in _INVALID_FILENAME_CHARS:
        stem = stem.replace(char, " ")
    stem = "-".join(stem.split())
    stem = stem[:80].strip("-")
    if not stem:
        return "Roman-Urdu-Translation.pdf"
    return f"{stem}-Roman-Urdu.pdf"


def render_export_page() -> None:
    """Display export controls only on the dedicated Export page."""
    result = _current_active_result()
    badge = status_markup("Export ready", "ready") if st.session_state.get("active_roman_urdu_translation") else status_markup("Roman Urdu required", "waiting")
    render_page_banner(
        "Deliverables",
        "Export",
        "Download TXT, Excel, and PDF files from the active transcript and Roman Urdu translation.",
        badge_html=badge,
    )
    st.caption("Exports use the current active transcript and Roman Urdu translation only.")
    try:
        original, translation, _ = validate_export_state(result)
    except ExportStateError as error:
        render_safe_ui_message(str(error))
        return
    export_title = str(result.get("title") or result.get("source_name") or "Roman Urdu Translation Report")
    col_txt, col_excel, col_pdf = st.columns(3)
    with col_txt:
        try:
            st.download_button(
                "📄 Download TXT",
                data=build_translation_txt(original, translation, title=export_title),
                file_name="roman-urdu-translation.txt",
                mime="text/plain; charset=utf-8",
                key="dedicated_export_txt",
                use_container_width=True,
            )
        except Exception as error:
            render_exception_feedback(
                error,
                fallback_message="TXT export failed.",
            )
    with col_excel:
        try:
            st.download_button(
                "📊 Download Excel",
                data=build_translation_excel(original, translation),
                file_name="roman-urdu-translation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dedicated_export_excel",
                use_container_width=True,
            )
        except Exception as error:
            render_exception_feedback(
                error,
                fallback_message="Excel export failed.",
            )
    with col_pdf:
        try:
            st.download_button(
                "📕 Download PDF",
                data=build_translation_pdf(original, translation, title=export_title),
                file_name=_safe_roman_urdu_pdf_filename(export_title),
                mime="application/pdf",
                key="dedicated_export_pdf",
                use_container_width=True,
            )
        except Exception as error:
            render_exception_feedback(
                error,
                fallback_message="PDF export failed.",
            )

def navigate_to(page: str) -> None:
    """Update the sidebar navigation widget before the next Streamlit rerun."""
    st.session_state["app_page"] = page


def render_home_page() -> None:
    """Render the command-center home dashboard with preserved functionality."""
    left_features = [
        ("🎙️", "View Transcript", "Inspect the original transcript for the active source.", "🎙 Transcript"),
        ("🌐", "Roman Urdu", "Generate or review the Roman Urdu version.", "🌐 Roman Urdu"),
        ("🔖", "Saved History", "Load previously saved Roman Urdu translations.", "📚 Saved History"),
    ]
    right_features = [
        ("📊", "Summary & Analysis", "Open summaries, actions, decisions, and questions.", "📊 Summary & Analysis"),
        ("🎬", "Video Library", "Browse processed meetings and reopen any record.", "📚 Video Library"),
        ("⬇️", "Export", "Download TXT, Excel, and PDF deliverables.", "📥 Export"),
    ]

    st.markdown('<div class="dashboard-layout">', unsafe_allow_html=True)
    center_col, left_col, right_col = st.columns([1.32, 0.92, 0.92], gap="large")

    with center_col:
        render_input_section()
        if st.session_state.get("active_original_transcript"):
            st.markdown(
                '<div class="dashboard-active-note">'
                'Meeting intelligence is live. Your active transcript, Roman Urdu translation, summary, library state, and exports remain synchronized across the workspace.'
                '</div>',
                unsafe_allow_html=True,
            )

    with left_col:
        st.markdown('<div class="home-side-column left-stack">', unsafe_allow_html=True)
        for index, (icon, title, description, page) in enumerate(left_features, start=1):
            render_dashboard_feature_card(
                icon=icon,
                title=title,
                description=description,
                page=page,
                side="left",
                index=index,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="home-side-column right-stack">', unsafe_allow_html=True)
        for index, (icon, title, description, page) in enumerate(right_features, start=1):
            render_dashboard_feature_card(
                icon=icon,
                title=title,
                description=description,
                page=page,
                side="right",
                index=index,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def _safe_model_message(exc: BaseException) -> str:
    """Return a user-facing error message with this session's API key
    scrubbed, so a key can never surface in the UI."""
    message = str(exc) or "Unknown error."
    secret = (st.session_state.get("ai_model_settings") or {}).get("api_key")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message


def _render_ai_config_notification() -> None:
    """Render the latest safe provider-agnostic configuration result."""
    notification = st.session_state.get("ai_config_notification")
    created = float(st.session_state.get("ai_config_notification_time") or 0.0)
    if not notification:
        return
    if created and time.time() - created >= 6.0:
        st.session_state.pop("ai_config_notification", None)
        st.session_state.pop("ai_config_notification_time", None)
        return

    def _display(value: object, fallback: str = "Not provided") -> str:
        text = str(value or "").strip()
        return escape(text) if text else fallback

    success = notification.get("type") == "success"
    title = _display(notification.get("title"), "AI Model Configuration Result")
    rows = [
        f"<div><b>Configuration:</b> {_display(notification.get('config_name'))}</div>",
        f"<div><b>Provider:</b> {_display(notification.get('provider'))}</div>",
        f"<div><b>Model:</b> {_display(notification.get('model'))}</div>",
    ]
    if success:
        if notification.get("base_url"):
            rows.append(f"<div><b>Base URL:</b> {_display(notification.get('base_url'))}</div>")
        rows.extend([
            f"<div><b>API Key:</b> {'Configured ✓' if notification.get('api_key_configured') else 'Not Configured ✗'}</div>",
            f"<div><b>Status:</b> {_display(notification.get('status'), 'Active ✓')}</div>",
        ])
    else:
        rows.extend([
            f"<div><b>Status:</b> {_display(notification.get('status'), 'Activation Failed')}</div>",
            f"<div><b>Error:</b> {_display(notification.get('error'), 'Unknown activation error.')}</div>",
        ])
    accent = "#2f9e62" if success else "#d64545"
    st.markdown(
        f'''<div role="status" style="border-left:4px solid {accent}; padding:0.8rem 1rem; margin:0 0 1rem; border-radius:0.5rem; animation:aiConfigNoticeFade 6s forwards;">
        <strong>{"✓" if success else "✗"} {title}</strong>{"".join(rows)}</div>
        <style>@keyframes aiConfigNoticeFade {{ 0%, 88% {{ opacity:1; max-height:600px; }} 100% {{ opacity:0; max-height:0; overflow:hidden; margin:0; padding-top:0; padding-bottom:0; }} }}</style>''',
        unsafe_allow_html=True,
    )


def _apply_ai_model_settings() -> None:
    """Re-apply the active AI configuration on every rerun (provider-driven).

    Restores ALL fields: config_name, provider, model, api_key, base_url,
    temperature.  Legacy session state that predates provider/base_url is
    migrated safely (provider=None, base_url=None).  When this session has
    no dict-based activation, the persisted store's active configuration is
    restored instead, so the sidebar UI, the config store and the runtime
    LLM always agree.
    """
    settings = st.session_state.get("ai_model_settings") or {}
    if settings.get("active"):
        try:
            from core.llm_provider import configure_llm

            configure_llm(
                provider=settings.get("provider"),
                model=str(settings.get("model") or ""),
                api_key=str(settings.get("api_key") or ""),
                base_url=settings.get("base_url"),
                temperature=float(settings.get("temperature") or 0.2),
            )
        except Exception as exc:  # defensive: never break the app on rerun
            st.session_state["ai_model_settings"] = dict(settings, active=False)
            st.sidebar.warning(f"AI model deactivated: {_safe_model_message(exc)}")
        return

    from core.llm_provider import (
        get_config_store,
        get_runtime_config,
        restore_active_configuration,
    )

    store = get_config_store()
    name = store.active_name()
    if not name:
        return
    entry = store.get(name)
    if entry is None:
        return
    runtime = get_runtime_config()
    if runtime is not None and (
        runtime.provider,
        runtime.model,
        runtime.base_url,
    ) == (entry.provider, entry.model, entry.base_url):
        return  # already active -- avoid clearing the client cache
    restore_active_configuration()


def _activate_ai_model() -> None:
    """Activate the AI configuration from the UI form values (provider-driven).

    Reads Configuration Name / Provider / Model / API Key / Base URL /
    Temperature and passes EVERY field explicitly to the centralized
    provider layer.  A configuration label such as "GroqCloud" is only ever
    the Configuration Name -- it is NEVER sent as the model name, so it can
    never trigger the old "Could not determine a provider for model ..."
    auto-detection bug.
    """
    provider_label = str(st.session_state.get("ai_cfg_provider", "Auto Detect"))
    model_name = str(st.session_state.get("ai_cfg_model", "")).strip()
    api_key = str(st.session_state.get("ai_cfg_key", "")).strip()
    base_url = str(st.session_state.get("ai_cfg_base_url", "")).strip() or None
    config_name = str(st.session_state.get("ai_cfg_name", "")).strip() or None
    try:
        temperature = float(st.session_state.get("ai_cfg_temperature", 0.2))
    except (TypeError, ValueError):
        st.error("Temperature must be a number between 0.0 and 2.0.")
        return

    if not model_name:
        st.error("Model Name cannot be empty.")
        return
    if not (0.0 <= temperature <= 2.0):
        st.error("Temperature must be between 0.0 and 2.0.")
        return

    try:
        from core.llm_config_ui import build_config_from_form

        config = build_config_from_form({
            "config_name": config_name,
            "provider_label": provider_label,
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
        })
    except Exception as exc:  # provider errors (missing key, bad URL, ...)
        st.error(_safe_model_message(exc))
        return

    st.session_state["ai_model_settings"] = {
        "active": True,
        "config_name": config.config_name,
        "provider": config.provider,
        "model": config.model,
        "api_key": api_key or config.api_key,
        "base_url": config.base_url,
        "temperature": float(config.temperature),
    }
    st.rerun()


def render_ai_model_settings() -> None:
    """Sidebar card: AI Model Settings UI -- SINGLE source of truth.

    Delegates the complete provider-driven configuration UI (Configuration
    Name, Provider dropdown, Model, masked API Key, Base URL,
    Temperature) to core.llm_config_ui, which validates and activates
    through core.llm_provider.configure_llm().  No duplicate UI or LLM
    logic lives in app.py.
    """
    from core.llm_config_ui import render_ai_model_settings as _render

    _render()


def render_sidebar() -> str:
    selected_page = st.session_state.get("app_page", "🏠 Home")
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-kicker">VIDEO AGENT WORKSPACE</div>'
            '<div class="sidebar-brand">AI Meeting Assistant</div>'
            '<div class="workspace-subtitle">Professional meeting and video analysis</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('<div class="sidebar-section-label">Navigate</div>', unsafe_allow_html=True)
        for item in NAV_ITEMS:
            if selected_page == item["key"]:
                st.markdown(
                    f'<div class="sidebar-nav-active">'
                    f'<div class="sidebar-nav-title">{escape(item["icon"])} {escape(item["label"])}</div>'
                    f'<div class="sidebar-nav-meta">{escape(item["description"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f'{item["icon"]} {item["label"]}',
                    key=f'sidebar_nav_{item["key"]}',
                    use_container_width=True,
                ):
                    navigate_to(item["key"])
                    st.rerun()

        st.divider()
        st.markdown('<div class="sidebar-section-label">System status</div>', unsafe_allow_html=True)
        st.markdown(status_markup("Pipeline ready", "ready"), unsafe_allow_html=True)
        if st.session_state.get("processed"):
            result = st.session_state.get("result") or {}
            metadata = result.get("metadata") or {}
            if metadata.get("cached"):
                st.markdown(status_markup("RAG available on demand · Cached", "cached"), unsafe_allow_html=True)
            else:
                st.markdown(status_markup("RAG available on demand", "waiting"), unsafe_allow_html=True)
        else:
            st.markdown(status_markup("Waiting for a processed meeting", "waiting"), unsafe_allow_html=True)

        render_ai_model_settings()

        if st.session_state.get("last_source"):
            st.divider()
            full_source = str(st.session_state.last_source)
            st.caption(f"Source: {compact_label(full_source, 38)}")
            st.caption(f"Language: {st.session_state.last_language or 'N/A'}")

        st.divider()
        st.markdown(
            '<div class="sidebar-shortcut-card">'
            '<div class="sidebar-shortcut-title">› 📚 Processed Videos</div>'
            '<div class="sidebar-shortcut-meta">Open processed video records and switch the active workspace.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button('Go to Video Library', key='sidebar_video_library_shortcut', use_container_width=True):
            navigate_to('📚 Video Library')
            st.rerun()

        render_processed_videos()
    return selected_page

def render_input_section() -> None:
    translation_mode = st.session_state.get("translation_mode", "Disabled")
    disabled_class = "translation-chip active-disabled" if translation_mode == "Disabled" else "translation-chip"
    enabled_class = "translation-chip active-enabled" if translation_mode == "Enabled" else "translation-chip"

    st.markdown(
        '<div class="dashboard-center-shell">'
        '<div class="central-process-panel">'
        '<div class="central-rings"><span></span></div>'
        '<div class="central-process-header">'
        '<div>'
        '<div class="central-process-eyebrow">PROCESS NEW MEETING</div>'
        '<div class="central-process-title">Process New Meeting</div>'
        '<div class="central-process-subtitle">Upload a video or provide a YouTube URL to get started.</div>'
        '</div>'
        '<div class="pipeline-ready-badge"><span class="dot"></span>Pipeline ready</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="process-intro">'
        '<div class="control-group-title">Translation</div>'
        '<div class="control-group-note">Enable Roman Urdu translation as part of the existing processing pipeline.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="translation-legend">'
        f'<span class="{disabled_class}"><span class="chip-dot"></span>Disabled</span>'
        f'<span class="{enabled_class}"><span class="chip-dot"></span>Enabled</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    translation_mode = st.radio(
        "Translation",
        ["Disabled", "Enabled"],
        horizontal=True,
        key="translation_mode",
    )
    translate = translation_mode == "Enabled"
    if translate:
        target_language_name = st.selectbox(
            "Target language",
            list(LANGUAGE_OPTIONS),
            key="translation_target_language",
        )
        target_language = target_language_name.lower()
    else:
        target_language = None

    youtube_tab, upload_tab = st.tabs(["YouTube URL", "Upload Video"])

    with youtube_tab:
        st.markdown(
            '<div class="control-group-title">YouTube URL</div>'
            '<div class="control-group-note">Paste a YouTube link to process a meeting, webinar, or video.</div>',
            unsafe_allow_html=True,
        )
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=xxxxxxxxxxx",
            key="youtube_url",
        )
        language_name = st.selectbox("Language", list(LANGUAGE_OPTIONS), key="youtube_language")
        if st.button("⚡ Process Meeting", type="primary", use_container_width=True, key="process_youtube"):
            if not youtube_url.strip():
                st.warning("Please enter a YouTube URL before processing.")
            elif not is_youtube_url(youtube_url):
                st.warning("Please provide a valid YouTube URL.")
            else:
                process_source(
                    source=youtube_url.strip(),
                    language=LANGUAGE_OPTIONS[language_name],
                    target_language=target_language,
                    translate=translate,
                    display_source="YouTube",
                    source_type="youtube",
                )

    with upload_tab:
        st.markdown(
            '<div class="upload-hint">'
            '<div class="upload-title">Upload Video or Audio</div>'
            '<div class="upload-copy">Drag &amp; drop your file here<br>or click to browse</div>'
            '<div class="helper-pill-row">'
            '<span class="helper-pill">📁 Browse Files</span>'
            '<span class="helper-pill">4GB per file</span>'
            '<span class="helper-pill">MP4, AVI, MOV, MKV, MP3, WAV, M4A</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
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
                uploaded_source = Path(pending_path)
            else:
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
            "⚡ Process Meeting",
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
                meeting_id = get_backend().build_meeting_processing_id(
                    saved_path.stem,
                    LANGUAGE_OPTIONS[language_name],
                )

                process_source(
                    source=str(saved_path.resolve()),
                    language=LANGUAGE_OPTIONS[language_name],
                    target_language=target_language,
                    translate=translate,
                    display_source=display_name,
                    source_type="meeting",
                    meeting_id=meeting_id,
                )

    st.markdown('</div></div>', unsafe_allow_html=True)

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
    """Persist a validated upload after bounded streaming to a temp file."""
    import tempfile
    from core.media_validation import get_max_upload_bytes, validate_media_file

    suffix = Path(getattr(uploaded_file, "name", "upload")).suffix.lower() or ".mp4"

    # IMPORTANT:
    # Always use the module-level UPLOAD_DIR so the upload lifecycle has
    # exactly one authoritative storage location. This also allows tests and
    # deployments to override UPLOAD_DIR safely without the persistence
    # function silently writing to a hard-coded "uploads" directory.
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    max_upload_bytes = get_max_upload_bytes()
    bytes_written = 0
    temp_path: Path | None = None

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=upload_dir, prefix=".upload-", suffix=".part", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = uploaded_file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_upload_bytes:
                    raise ValueError("Uploaded file exceeds the maximum allowed size.")
                digest.update(chunk)
                temp_file.write(chunk)
            temp_file.flush()

        if bytes_written == 0:
            raise ValueError("Uploaded file is empty.")
        validate_media_file(
            temp_path,
            max_upload_bytes=max_upload_bytes,
            expected_extension=suffix,
        )

        meeting_id = "meeting_" + digest.hexdigest()[:16]
        saved_file_path = upload_dir / f"{meeting_id}{suffix}"
        upload_root = upload_dir.resolve()
        if saved_file_path.resolve().parent != upload_root:
            raise RuntimeError("Upload destination is outside configured UPLOAD_DIR.")
        if not saved_file_path.exists():
            temp_path.replace(saved_file_path)
        else:
            temp_path.unlink(missing_ok=True)
        if not saved_file_path.exists():
            raise RuntimeError("Uploaded file could not be saved.")
        return saved_file_path
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def get_upload_identity(uploaded_file: Any) -> str:
    """Return a stable identity for one selected upload across reruns.

    Streamlit's UploadedFile exposes ``file_id``: it stays constant for the
    same selected file during the whole session and changes only when a
    different file is chosen, so it is the reliable signal for "same upload".
    A name+size fallback covers older Streamlit versions.
    """
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return str(file_id)
    name = str(getattr(uploaded_file, "name", "") or "")
    size = getattr(uploaded_file, "size", None)
    return f"{name}|{size}"


def _cleanup_upload_file(upload_path: Any) -> None:
    """Best-effort removal of an orphaned persisted upload.

    A file is kept whenever it is still the active source for the session
    (an active result may reference it and a background pipeline may still
    be reading it). Cleanup never raises and never removes active sources.
    """
    try:
        upload_path = Path(upload_path)
        if not upload_path.exists():
            return
        active_result = st.session_state.get("result") or {}
        active_source = str(
            active_result.get("source")
            or (active_result.get("metadata") or {}).get("source")
            or ""
        )
        if active_source and Path(active_source).resolve() == Path(upload_path).resolve():
            return
        upload_path.unlink(missing_ok=True)
    except Exception:
        pass


def process_uploaded_file(
    uploaded_file: Any,
    language: str,
    target_language: str | None = None,
    translate: bool = False,
) -> None:
    """Save upload to a permanent path and then start the processing pipeline."""
    saved_file_path = persist_uploaded_file(uploaded_file)
    meeting_id = get_backend().build_meeting_processing_id(
        saved_file_path.stem,
        language,
    )

    process_source(
        source=str(saved_file_path.resolve()),
        language=language,
        target_language=target_language,
        translate=translate,
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
    "input": "Preparing input",
    "audio": "Processing audio",
    "transcription": "Transcribing with Whisper",
    "translation": "Translating transcript",
    "title": "Structuring meeting context",
    "summary": "Generating summary & analysis",
    "rag": "Building AI knowledge base",
    "finalizing": "Ready",
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
        "waiting": "Queued",
        "running": text,
        "completed": "Completed",
        "failed": "Failed",
        "skipped": "Skipped",
        "cached": "Loaded from cache",
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

    stage_rows: list[str] = []
    for index, stage in enumerate(PROCESSING_STAGE_ORDER, start=1):
        item = stage_state.get(stage, {"state": "waiting", "message": "Waiting"})
        item_state = str(item.get("state") or "waiting")
        item_message = str(item.get("message") or "Waiting")
        stage_rows.append(
            f'<div class="stage-row {escape(item_state)}">'
            f'<div class="stage-index">{index:02d}</div>'
            f'<div class="stage-copy">'
            f'<div class="stage-name">{escape(PROCESSING_STAGE_LABELS.get(stage, stage.title()))}</div>'
            f'<div class="stage-message">{escape(item_message)}</div>'
            f'</div>'
            f'<div class="stage-state">{escape(item_state)}</div>'
            f'</div>'
        )

    st.markdown('<div class="processing-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="processing-summary">'
        '<div>'
        '<div class="section-kicker">Pipeline status</div>'
        '<div class="processing-title">Meeting intelligence is being prepared</div>'
        f'<div class="processing-subtitle">Current stage: {escape(label)} · {escape(detail)}</div>'
        '</div>'
        f'<div class="processing-stat">Step {active_step} / {len(PROCESSING_STAGE_ORDER)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.progress(
        overall_percent / 100,
        text=f"{overall_percent}% complete",
    )
    st.markdown(status_markup(f"{label} — {detail}", presentation_state), unsafe_allow_html=True)
    if completed_count:
        st.caption(f"{completed_count} stage(s) completed")
    if chunk_summary:
        st.caption(chunk_summary)
    if elapsed_seconds is not None:
        st.caption(f"Elapsed time: {format_seconds(elapsed_seconds)}")
    if error_message:
        st.error(f"✕ Processing failed — {error_message}")
    st.markdown(f'<div class="processing-grid">{"".join(stage_rows)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def process_source(
    source: str,
    language: str,
    display_source: str,
    source_type: str,
    meeting_id: str | None = None,
    target_language: str | None = None,
    translate: bool = False,
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
                # Route every cached YouTube open through main.py's
                # authoritative loader.  The loader repairs incomplete
                # Meeting Intelligence fields before returning the result.
                backend = get_backend()
                pipeline_result = backend.load_cached_video(
                    video_id=cached_video_id,
                    source=source,
                    source_type=source_type,
                    language=language,
                )
                if pipeline_result is None:
                    # The record may have changed between video_exists() and
                    # the authoritative lookup; continue through normal
                    # processing rather than constructing a partial result.
                    raise RuntimeError(
                        "Cached record is no longer available or not completed."
                    )

                result = normalize_pipeline_result(pipeline_result)
                result.setdefault("source_type", source_type)
                result.setdefault("language", language)
                result.setdefault("source_name", display_source)
                metadata = dict(result.get("metadata") or {})
                metadata["cached"] = True
                metadata["video_id"] = cached_video_id
                metadata["rag_cache"] = "lazy_streamlit_resource"
                result["metadata"] = metadata
                # Cached RAG remains lazy. Reuse an in-memory chain only when
                # every identity check proves it belongs to this exact video;
                # otherwise detach it and let the next question load lazily.
                previous_result = st.session_state.get("result") or {}
                previous_rag_video_id = st.session_state.get("rag_video_id")
                reusable_chain = can_reuse_cached_rag(
                    cached_video_id,
                    result.get("transcript"),
                    previous_result,
                    previous_rag_video_id,
                    st.session_state.get("active_source_identity"),
                )
                result["rag_chain"] = (
                    previous_result.get("rag_chain") if reusable_chain else None
                )
                st.session_state.rag_video_id = cached_video_id if reusable_chain else None
                set_active_processed_result(result, display_source)

                with st.container(border=True):
                    st.markdown(
                        '<div class="section-title">♻ Cached Meeting Found</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("The transcript, summary, and vector database will be reused from the existing record.")
                    st.success("✓ Transcript — Loaded from Cache")
                    st.success("✓ Summary — Loaded from Cache")
                    st.success("✓ Meeting Intelligence — Loaded from Cache")
                    st.success("✓ Vector Database — Loaded from Existing Storage")
                    st.caption(f"Video ID: {cached_video_id}")
                    st.caption(f"Title: {display_value(result.get('title') or display_source, 'Untitled video')}")

                st.success(
                    "Cached video loaded successfully. Transcript is ready; RAG will load when you ask a question."
                )
                return

            except Exception as exc:
                for key, value in previous_processing_state.items():
                    st.session_state[key] = value
                render_exception_feedback(
                    exc,
                    fallback_message="Could not load the cached video.",
                )
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
                target_language=target_language,
                translate=translate,
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
            shown_message = friendly_llm_error(final_error) or (
                "⚠️ Something went wrong while processing your request.\n\n"
                "Please try again. If the problem continues, check your application settings."
            )
            update_processing_display(error_message=shown_message)
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

        st.success("✓ Processing Completed Successfully")
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
        shown_message = render_exception_feedback(
            exc,
            fallback_message="⚠️ Something went wrong while processing your request.\n\nPlease try again. If the problem continues, check your application settings.",
        )
        update_processing_display(error_message=shown_message)


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
    for start in range(0, len(metrics), 2):
        row_columns = st.columns(2)
        for column, (label, value) in zip(row_columns, metrics[start:start + 2]):
            with column:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div>'
                    f'<div class="kpi-value">{escape(str(value))}</div></div>',
                    unsafe_allow_html=True,
                )

def render_overview(result: dict[str, Any]) -> None:
    render_page_banner(
        "Meeting intelligence",
        display_value(result.get("title"), "AI Meeting Assistant"),
        "High-level context, structured overview, and pipeline metadata for the active source.",
        badge_html=status_markup("Active result", "ready"),
    )
    render_kpis(result)

    action_count = len(as_items(result.get("actions")))
    decision_count = len(as_items(result.get("decisions")))
    question_count = len(as_items(result.get("questions")))
    st.markdown(
        '<div class="info-grid">'
        f'<div class="info-card"><div class="info-label">Action items</div><div class="info-value">{action_count}</div></div>'
        f'<div class="info-card"><div class="info-label">Key decisions</div><div class="info-value">{decision_count}</div></div>'
        f'<div class="info-card"><div class="info-label">Open questions</div><div class="info-value">{question_count}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="result-panel"><div class="page-banner-kicker">Summary preview</div>'
        f'<div style="margin-top:0.65rem;color:var(--text);font-size:1rem;line-height:1.75;">{escape(display_value(result.get("summary"), "No summary generated.")).replace(chr(10), "<br>")}</div></div>',
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
            "AI Meeting Intelligence — Complete Meeting Report",
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
                            escape(clean_text(" · ".join(metadata))),
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
    render_page_banner(
        "AI summary",
        "Complete Summary",
        "Structured summary generated from the active meeting or video.",
        badge_html=status_markup("Summary ready", "ready"),
    )
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
                f"PDF ready · {len(pdf_data) / 1024:.1f} KB"
            )

        except Exception as exc:
            render_exception_feedback(
                exc,
                fallback_message="Could not generate the complete PDF report.",
                technical_label="PDF technical details",
            )

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
    render_page_banner(
        "AI assistant",
        "Ask Your Meeting",
        "Ask questions about this video or meeting.",
        badge_html=status_markup("AI ready", "ready"),
    )
    st.caption("Ask anything about the meeting...")

    chat_identity = get_result_identity(result)
    if st.session_state.get("chat_history_source_id") != chat_identity:
        st.session_state.chat_history = []
        st.session_state.chat_history_source_id = chat_identity

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about the meeting...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            rag_chain = get_owned_rag_chain(result)

            if rag_chain is None:
                video_id = get_result_video_id(result)

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
                        answer = render_exception_feedback(
                            exc,
                            fallback_message="Could not load the meeting knowledge base.",
                            technical_label="RAG technical details",
                        )
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
                    answer = ask_rag_question(rag_chain, question)

                st.markdown(answer)
            except Exception as exc:
                answer = "I could not generate an answer from the meeting knowledge."
                render_exception_feedback(
                    exc,
                    fallback_message=answer,
                    technical_label="Chat technical details",
                )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

def render_prompt_studio(result: dict[str, Any]) -> None:
    """Render the Prompt Studio and keep all variables safely scoped to this function."""
    st.subheader("🧠 AI Prompt Studio")
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

    if st.button("✨ Generate AI Prompt", use_container_width=True, key="generate_ai_prompt"):
        try:
            prompt_text = generate_from_current_context()
            st.session_state.generated_prompt = prompt_text
        except Exception as exc:
            render_exception_feedback(
                exc,
                fallback_message="Prompt generation failed.",
            )

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
                        📋 Copy Prompt
                    </button>

                    <script>
                        const button = document.getElementById("copyButton");

                        button.addEventListener("click", async () => {{
                            const text = {safe_payload};

                            try {{
                                await navigator.clipboard.writeText(text);

                                button.innerText = "✅ Copied!";

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
                                    button.innerText = "✅ Copied!";
                                }} else {{
                                    button.innerText = "❌ Copy Failed";
                                }}
                            }}

                            setTimeout(() => {{
                                button.innerText = "📋 Copy Prompt";
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
                label="⬇️ Download .txt",
                data=edited_prompt or generated_prompt,
                file_name="ai_generated_prompt.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key="download_generated_prompt",
            )

        with col3:
            if st.button("🔄 Regenerate", key="regenerate_ai_prompt", use_container_width=True):
                try:
                    regenerated_prompt = generate_from_current_context()
                    st.session_state.generated_prompt = regenerated_prompt
                except Exception as exc:
                    render_exception_feedback(
                        exc,
                        fallback_message="Prompt regeneration failed.",
                    )

def _render_translation_service_error(
    error_info: dict[str, Any],
    *,
    source_id: str | None = None,
) -> None:
    """Render a clean, user-facing translation service error.

    Never shows raw tracebacks or secrets to the user; those go to server
    logs. Applies the appropriate Streamlit primitive per category and offers
    an explicit Retry Translation button for retryable categories (rate limit
    and temporary API failure). Retry only happens on a user click: the
    button sets a one-shot session flag, so no automatic retry loop exists.
    """
    category = error_info.get("category", "unexpected")
    message = error_info.get("message", UNEXPECTED_ERROR_MESSAGE)
    completed = int(error_info.get("completed_chunks") or 0)
    partial = str(error_info.get("partial_translation") or "")

    if category == "rate_limit":
        retry_after = error_info.get("retry_after")
        if retry_after is not None:
            try:
                wait_seconds = int(float(retry_after))
            except (TypeError, ValueError):
                wait_seconds = 0
            if wait_seconds > 0:
                message = f"{message}\n\n⏳ Suggested wait: {wait_seconds} seconds."
        st.warning(message)
    elif category == "temporary":
        st.warning(message)
    elif category == "permanent":
        st.error(message)
    else:
        st.error(message)

    if completed > 0:
        st.info(
            f"ℹ️ Translation was partially completed: {completed} chunk(s) preserved. "
            "Your completed chunks have NOT been discarded."
        )
        if partial.strip():
            with st.expander(f"View preserved translation ({completed} chunk(s))"):
                st.text(partial)

    if category in ("rate_limit", "temporary"):
        unique_key = f"retry_roman_urdu_translation_{source_id or 'unknown'}"
        if st.button("🔄 Retry Translation", key=unique_key, use_container_width=True):
            st.session_state.pop("roman_urdu_translation_error", None)
            st.session_state["roman_urdu_retry_requested"] = True
            st.rerun()


def render_roman_urdu_transcript(
    result: dict[str, Any],
    show_content: bool = True,
    show_exports: bool = True,
) -> None:
    """Render Roman Urdu transcript section."""

    render_page_banner(
        "Roman Urdu",
        "Roman Urdu Transcript",
        "Convert the English transcript into natural Pakistani Roman Urdu while preserving the existing backend workflow.",
        badge_html=status_markup(
            "Translation ready" if st.session_state.get("active_roman_urdu_translation") else "Translation not generated",
            "ready" if st.session_state.get("active_roman_urdu_translation") else "waiting",
        ),
    )

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

    source_id = active_source_id or get_transcript_source_id(transcript)

    if (
        st.session_state.roman_urdu_source_id is not None
        and st.session_state.roman_urdu_source_id != source_id
    ):
        st.session_state.roman_urdu_transcript = None
        st.session_state.roman_urdu_source_id = None

    st.markdown(
        '<div class="helper-pill-row">'
        f'<span class="helper-pill">Source: {escape(compact_label(source_name, 42))}</span>'
        f'<span class="helper-pill">Transcript length: {len(str(transcript)):,} chars</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    run_translation_requested = st.button(
        "🌐 Convert to Roman Urdu",
        key=f"convert_roman_urdu_{source_id}",
        use_container_width=True,
        type="primary",
    )
    # One-shot retry support: "Retry Translation" (rendered on failure) sets a
    # session flag that re-runs the translation flow exactly once. The flag is
    # popped here, so no automatic retry loop is possible -- another failure
    # simply renders the button again for an explicit user click.
    run_translation_requested = run_translation_requested or st.session_state.pop(
        "roman_urdu_retry_requested", False
    )

    if run_translation_requested:
        try:
            try:
                cached_translation = get_cached_translation(
                    transcript,
                    video_id=str(active_source_id) if active_source_id else None,
                )
            except Exception as cache_error:
                cached_translation = None
                traceback.print_exception(
                    type(cache_error),
                    cache_error,
                    cache_error.__traceback__,
                )
                st.warning("Translation cache unavailable; continuing without saved translations.")

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
                st.session_state.pop("roman_urdu_translation_error", None)
                st.success("⚡ Roman Urdu translation loaded from saved database.")
            else:
                st.info("🌐 No saved translation found. Processing Roman Urdu translation...")
                with st.spinner(
                    "Translating transcript into Roman Urdu..."
                ):
                    roman_urdu_result = translate_to_roman_urdu(
                        transcript=transcript,
                        max_retries=3,
                    )

                st.session_state.roman_urdu_transcript = roman_urdu_result
                st.session_state.roman_urdu_source_id = source_id
                st.session_state.pop("roman_urdu_translation_error", None)
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
                    active_model_settings = st.session_state.get("ai_model_settings") or {}
                    active_model_label = (
                        active_model_settings.get("model")
                        or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
                    )
                    save_translation_with_source_identity(
                        transcript,
                        roman_urdu_result,
                        model_name=active_model_label,
                        source_name=source_name,
                        source_id=str(active_source_id) if active_source_id else None,
                    )
                    saved_record = get_saved_translation_by_hash(
                        get_transcript_hash(transcript),
                        video_id=str(active_source_id) if active_source_id else None,
                    )
                except Exception as cache_error:
                    traceback.print_exception(
                        type(cache_error),
                        cache_error,
                        cache_error.__traceback__,
                    )
                    st.warning(
                        "Translation completed, but could not be saved for future use."
                    )
                else:
                    st.success("💾 Roman Urdu translation saved for future use.")
                set_active_translation(
                    transcript,
                    roman_urdu_result,
                    record=saved_record,
                    source=source_name,
                    clear_downstream=True,
                )

        except Exception as error:
            # Production-friendly error handling: rate limits (HTTP 429),
            # temporary API failures, and configuration errors render as clean
            # user-facing messages with an explicit Retry button -- never a raw
            # traceback. Unexpected errors also show a generic message; the
            # full exception goes to server logs only.
            error_info = classify_translation_error(error)
            st.session_state["roman_urdu_translation_error"] = error_info
            if error_info["category"] == "unexpected":
                traceback.print_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            _render_translation_service_error(
                error_info,
                source_id=source_id,
            )

    if st.session_state.roman_urdu_transcript:
        if show_content:
            st.text_area(
                "Roman Urdu Transcript",
                value=st.session_state.roman_urdu_transcript,
                height=500,
                key=f"roman_urdu_transcript_display_{source_id}",
            )

        if show_exports:
            try:
                export_original, export_translation, _ = validate_export_state(result)
            except ExportStateError as error:
                render_safe_ui_message(str(error))
                return
            st.markdown('<div class="section-title">Save Translation</div>', unsafe_allow_html=True)
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
                        "📄 Download TXT",
                        data=txt_bytes,
                        file_name="roman-urdu-translation.txt",
                        mime="text/plain; charset=utf-8",
                        use_container_width=True,
                        key="download_roman_urdu_export_txt",
                    )
                except Exception as error:
                    render_exception_feedback(
                        error,
                        fallback_message="TXT export failed.",
                    )

            with export_columns[1]:
                try:
                    excel_bytes = build_translation_excel(
                        export_original,
                        export_translation,
                    )
                    st.download_button(
                        "📊 Download Excel",
                        data=excel_bytes,
                        file_name="roman-urdu-translation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_roman_urdu_export_excel",
                    )
                except Exception as error:
                    render_exception_feedback(
                        error,
                        fallback_message="Excel export failed.",
                    )

            with export_columns[2]:
                try:
                    pdf_bytes = build_translation_pdf(
                        export_original,
                        export_translation,
                        title=export_title,
                    )
                    st.download_button(
                        "📕 Download PDF",
                        data=pdf_bytes,
                        file_name=_safe_roman_urdu_pdf_filename(export_title),
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_roman_urdu_export_pdf",
                    )
                except Exception as error:
                    render_exception_feedback(
                        error,
                        fallback_message="PDF export failed.",
                    )

def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
            AI Meeting Intelligence · Whisper · Mistral · Chroma RAG
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    configure_page()
    inject_css()
    initialize_state()
    _apply_ai_model_settings()
    _render_ai_config_notification()
    selected_page = render_sidebar()
    render_header()
    render_reopen_notice()

    if selected_page == "🎙 Transcript":
        render_transcript_page()
    elif selected_page == "🌐 Roman Urdu":
        render_roman_urdu_page()
    elif selected_page == "📊 Summary & Analysis":
        render_summary_analysis_page()
    elif selected_page == "📚 Saved History":
        render_translation_history()
    elif selected_page == "📚 Video Library":
        render_video_library_page()
    elif selected_page == "📥 Export":
        render_export_page()
    else:
        render_home_page()

    render_footer()


if __name__ == "__main__":
    main()
