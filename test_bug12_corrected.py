
"""
BUG #12 — STANDALONE UPLOAD LIFECYCLE TEST
============================================

Run:
    python -m py_compile test_bug12.py
    python test_bug12.py

This test intentionally injects a fake Streamlit module BEFORE importing
the application. It must therefore not produce Streamlit bare-mode warnings.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import Mock


PASS = 0
FAIL = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"[PASS] {name}")


def bad(name, exc):
    global FAIL
    FAIL += 1
    print(f"[FAIL] {name}")
    print(f"       {type(exc).__name__}: {exc}")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


# ============================================================
# STREAMLIT STUB — MUST BE INSTALLED BEFORE app IMPORT
# ============================================================

class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


st = types.ModuleType("streamlit")
st.session_state = SessionState()


def cache_decorator(func=None, **kwargs):
    def decorator(fn):
        return fn

    if callable(func):
        return func

    return decorator


st.cache_resource = cache_decorator
st.cache_data = cache_decorator


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def dummy(*args, **kwargs):
    return None


def false(*args, **kwargs):
    return False


def empty(*args, **kwargs):
    return ""


def none(*args, **kwargs):
    return None


for name in [
    "error", "warning", "info", "success", "write", "markdown",
    "caption", "title", "header", "subheader", "divider", "text",
    "exception", "image", "audio", "video", "set_page_config",
]:
    setattr(st, name, dummy)

for name in [
    "button", "download_button", "checkbox", "toggle",
    "form_submit_button",
]:
    setattr(st, name, false)

for name in [
    "text_area", "text_input", "selectbox", "radio",
]:
    setattr(st, name, empty)

st.file_uploader = none
st.number_input = lambda *a, **k: 0
st.slider = lambda *a, **k: 0
st.multiselect = lambda *a, **k: []
st.columns = lambda *a, **k: []
st.tabs = lambda *a, **k: []
st.container = lambda *a, **k: DummyContext()
st.expander = lambda *a, **k: DummyContext()
st.spinner = lambda *a, **k: DummyContext()
st.form = lambda *a, **k: DummyContext()
st.stop = dummy

# Some applications inspect these attributes.
st.runtime = types.SimpleNamespace()
st.rerun = dummy

sys.modules["streamlit"] = st


# ============================================================
# GENERIC IMPORT STUBS
# ============================================================

def module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# dotenv
dotenv = module("dotenv")
dotenv.load_dotenv = dummy


# core package
core = module("core")
core.__path__ = []


def core_module(name):
    m = module(f"core.{name}")
    setattr(core, name, m)
    return m


# database
db = core_module("database")
db.delete_video = Mock(return_value=True)
db.get_video = Mock(return_value=None)
db.list_videos = Mock(return_value=[])
db.video_exists = Mock(return_value=False)
db.create_video = Mock(return_value=True)
db.update_video = Mock(return_value=True)
db.save_completed_video = Mock(return_value=True)
db.mark_video_failed = Mock(return_value=True)


# vector store
vs = core_module("vector_store")
vs.delete_vector_store = Mock(return_value=True)


# prompt generator
pg = core_module("prompt_generator")
pg.generate_prompt = Mock(return_value="test prompt")


# roman urdu
ru = core_module("roman_urdu_translator")
ru.translate_to_roman_urdu = Mock(return_value="roman urdu")


# translator
tr = core_module("translator")
tr.translate_text = Mock(return_value="translated")
tr.translate_to_language = Mock(return_value="translated")


# transcriber
tc = core_module("transcriber")
tc.transcribe_all = Mock(return_value="transcript")


# summarizer
sm = core_module("summarize")
sm.summarize = Mock(return_value="summary")
sm.generate_title = Mock(return_value="title")


# rag
rg = core_module("rag_engine")
rg.build_rag_chain = Mock(return_value=None)
rg.ask_question = Mock(return_value="")


# translation cache
cache = core_module("translation_cache")

_cache_names = [
    "get_cached_translation",
    "get_translation",
    "load_translation",
    "translation_exists",

    # Saved translation lookups
    "get_saved_translation_by_hash",
    "get_saved_translation_by_id",

    # Transcript/cache helpers
    "get_transcript_hash",
    "get_translation_count",
    "list_saved_translations",

    # Save operations
    "save_translation",
    "save_translation_cache",

    # Delete operations
    "delete_translation",
    "delete_translation_by_id",
    "delete_cached_translation",
]

for name in _cache_names:
    setattr(cache, name, Mock(return_value=None))


# translation exporter
exporter = core_module("translation_exporter")

_export_names = [
    "build_translation_txt",
    "build_translation_excel",
    "build_translation_pdf",
    "_aligned_records",
]

for name in _export_names:
    setattr(exporter, name, Mock(return_value=b"TEST"))


# analytics
analytics = core_module("video_library_analytics")

for name in [
    # Existing analytics functions
    "get_video_stats",
    "get_library_analytics",
    "get_language_distribution",

    # Current app.py analytics imports
    "calculate_completion_percentage",
    "calculate_library_analytics",
    "format_duration",
    "get_video_content_status",
]:
    setattr(
        analytics,
        name,
        Mock(return_value=0),
    )
    setattr(analytics, name, Mock(return_value={}))

# utils
utils = module("utils")
utils.__path__ = []


audio = module("utils.audio_processor")
audio.process_input = Mock(return_value=[])
audio.cleanup_chunks = Mock()
audio.cleanup_file = Mock()


# extractor
extractor = module("extractor")
extractor.extract_action_items = Mock(return_value=[])
extractor.extract_decisions = Mock(return_value=[])
extractor.extract_questions = Mock(return_value=[])


# ============================================================
# IMPORT APPLICATION
# ============================================================

print()
print("=" * 70)
print("BUG #12 — STANDALONE UPLOAD LIFECYCLE TEST")
print("=" * 70)
print()

try:
    try:
        app = importlib.import_module("app")
    except ModuleNotFoundError:
        app = importlib.import_module("app_bug11_fixed")

    print("[PASS] Application imported successfully")

except Exception as exc:
    print()
    print("=" * 70)
    print("ERROR: APPLICATION COULD NOT BE IMPORTED")
    print("=" * 70)
    print()
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(1)


# ============================================================
# TEST DIRECTORY
# ============================================================

ROOT = Path(tempfile.mkdtemp(prefix="video_agent_bug12_"))
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FAKE STREAMLIT UPLOAD
# ============================================================

class FakeUpload:
    def __init__(self, name, data, file_id=None):
        self.name = name
        self.data = data
        self.file_id = file_id
        self.size = len(data)
        self._position = 0

    def seek(self, position):
        self._position = position

    def read(self, size=-1):
        if self._position >= len(self.data):
            return b""

        if size < 0:
            chunk = self.data[self._position:]
            self._position = len(self.data)
            return chunk

        end = min(
            self._position + size,
            len(self.data),
        )

        chunk = self.data[self._position:end]
        self._position = end
        return chunk


# ============================================================
# HELPERS
# ============================================================

def configure_upload_dir():
    """
    Patch the application upload directory only if the app exposes it.
    This avoids hard-coding a project-specific path in the test.
    """

    if hasattr(app, "UPLOAD_DIR"):
        app.UPLOAD_DIR = UPLOAD_DIR


def clear_state():
    st.session_state.clear()


configure_upload_dir()


# ============================================================
# TEST 1 — SAME IDENTITY
# ============================================================

def test_same_identity():

    clear_state()

    a = FakeUpload(
        "meeting.mp4",
        b"AAAA",
        "UPLOAD-A",
    )

    b = FakeUpload(
        "meeting.mp4",
        b"AAAA",
        "UPLOAD-A",
    )

    first = app.get_upload_identity(a)
    second = app.get_upload_identity(b)

    check(
        first == second,
        "Same upload does not produce stable identity",
    )

    ok("Same upload identity is stable")


# ============================================================
# TEST 2 — DIFFERENT IDENTITY
# ============================================================

def test_different_identity():

    clear_state()

    a = FakeUpload(
        "meeting.mp4",
        b"AAAA",
        "UPLOAD-A",
    )

    b = FakeUpload(
        "meeting.mp4",
        b"BBBB",
        "UPLOAD-B",
    )

    first = app.get_upload_identity(a)
    second = app.get_upload_identity(b)

    check(
        first != second,
        "Different uploads received same identity",
    )

    ok("Different uploads have different identities")


# ============================================================
# TEST 3 — PERSISTENCE
# ============================================================

def test_persistence():

    clear_state()

    upload = FakeUpload(
        "meeting.mp4",
        b"REAL-UPLOAD-CONTENT",
        "UPLOAD-PERSIST",
    )

    original_dir = getattr(app, "UPLOAD_DIR", None)

    app.UPLOAD_DIR = UPLOAD_DIR

    try:
        saved = app.persist_uploaded_file(upload)
    finally:
        app.UPLOAD_DIR = original_dir

    saved = Path(saved)

    check(
        saved.exists(),
        f"Persisted upload does not exist: {saved}",
    )

    check(
        saved.read_bytes() == b"REAL-UPLOAD-CONTENT",
        "Persisted upload contents are incorrect",
    )

    check(
        saved.parent.resolve() == UPLOAD_DIR.resolve(),
        "Upload was persisted outside configured UPLOAD_DIR",
    )

    ok("Uploaded file persists in configured upload directory")


# ============================================================
# TEST 4 — NO PART FILE
# ============================================================

def test_no_part_file():

    leftovers = list(
        UPLOAD_DIR.glob(".upload-*.part")
    )

    check(
        not leftovers,
        f"Partial upload files remain: {leftovers}",
    )

    ok("Successful persistence leaves no .part orphan")


# ============================================================
# TEST 5 — CLEAN OLD FILE
# ============================================================

def test_cleanup_old_file():

    clear_state()

    old = UPLOAD_DIR / "old.mp4"
    new = UPLOAD_DIR / "new.mp4"

    old.write_bytes(b"OLD")
    new.write_bytes(b"NEW")

    app._cleanup_upload_file(old)

    check(
        not old.exists(),
        "Old upload was not cleaned",
    )

    check(
        new.exists(),
        "Unrelated new upload was deleted",
    )

    ok("Old upload cleanup is isolated")


# ============================================================
# TEST 6 — ACTIVE SOURCE PROTECTION
# ============================================================

def test_active_source_protection():

    clear_state()

    active = UPLOAD_DIR / "active.mp4"
    active.write_bytes(b"ACTIVE")

    st.session_state["result"] = {
        "source": str(active.resolve()),
    }

    app._cleanup_upload_file(active)

    check(
        active.exists(),
        "Active source was deleted",
    )

    ok("Active source is protected")


# ============================================================
# TEST 7 — MISSING FILE
# ============================================================

def test_missing_file():

    clear_state()

    missing = UPLOAD_DIR / "missing.mp4"

    check(
        not missing.exists(),
        "Missing test file unexpectedly exists",
    )

    app._cleanup_upload_file(missing)

    ok("Missing upload is handled safely")


# ============================================================
# TEST 8 — CLEANUP FAILURE
# ============================================================

def test_cleanup_failure():

    clear_state()

    class BadPath:
        def resolve(self):
            raise OSError("simulated cleanup failure")

    try:
        app._cleanup_upload_file(BadPath())
    except Exception as exc:
        raise AssertionError(
            f"Cleanup failure escaped: {exc}"
        )

    ok("Cleanup failure does not crash application")


# ============================================================
# TEST 9 — UNRELATED FILE
# ============================================================

def test_unrelated_file():

    clear_state()

    permanent = UPLOAD_DIR / "permanent.mp4"
    target = UPLOAD_DIR / "target.mp4"

    permanent.write_bytes(b"PERMANENT")
    target.write_bytes(b"TARGET")

    app._cleanup_upload_file(target)

    check(
        not target.exists(),
        "Target upload was not cleaned",
    )

    check(
        permanent.exists(),
        "Unrelated permanent file was deleted",
    )

    ok("Unrelated/permanent file is protected")


# ============================================================
# TEST 10 — RESET SESSION
# ============================================================

def test_reset_session():

    clear_state()

    pending = UPLOAD_DIR / "pending.mp4"
    pending.write_bytes(b"PENDING")

    st.session_state["pending_upload_path"] = str(pending)
    st.session_state["pending_upload_name"] = "pending.mp4"
    st.session_state["pending_upload_identity"] = "PENDING-ID"

    app.reset_session()

    check(
        "pending_upload_path" not in st.session_state,
        "pending_upload_path was not cleared",
    )

    check(
        "pending_upload_name" not in st.session_state,
        "pending_upload_name was not cleared",
    )

    check(
        "pending_upload_identity" not in st.session_state,
        "pending_upload_identity was not cleared",
    )

    if pending.exists():
        print(
            "       NOTE: reset_session cleared state but retained "
            "the pending physical file."
        )

    ok("reset_session clears pending upload state")


# ============================================================
# RUN
# ============================================================

tests = [
    test_same_identity,
    test_different_identity,
    test_persistence,
    test_no_part_file,
    test_cleanup_old_file,
    test_active_source_protection,
    test_missing_file,
    test_cleanup_failure,
    test_unrelated_file,
    test_reset_session,
]


print()
print("=" * 70)
print("RUNNING BUG #12 TESTS")
print("=" * 70)
print()

for test in tests:
    try:
        test()
    except Exception as exc:
        bad(test.__name__, exc)


print()
print("=" * 70)
print(f"BUG #12 TESTS: {PASS} passed, {FAIL} failed")
print("=" * 70)

if FAIL == 0:
    print()
    print("BUG #12 TEST HARNESS: PASS")
    print("Upload identity: PASS")
    print("Persistence: PASS")
    print("Partial-file handling: PASS")
    print("Cleanup isolation: PASS")
    print("Active source protection: PASS")
    print("Missing-file safety: PASS")
    print("Cleanup error safety: PASS")
    print("Unrelated-file protection: PASS")
    print("Session-state cleanup: PASS")
    print()
    print("BUG #12 VALIDATION COMPLETE")
else:
    print()
    print("BUG #12 TEST HARNESS: FAIL")
    print(f"Investigate {FAIL} failing test(s).")
    raise SystemExit(1)
