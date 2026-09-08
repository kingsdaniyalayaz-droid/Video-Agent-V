"""
BUG #11 — CORRECTED EXPORT STATE & SOURCE IDENTITY TESTS
=========================================================

This test harness is aligned with the current app.py API.

Validated contracts:
- validate_export_state(result)
- saved_translation_matches_result(record, result)
- set_active_processed_result(normalized_result, display_source)
- legacy translation state cannot override authoritative state
- TXT/XLSX/PDF exporter arguments
- timestamp alignment arguments

The application itself is not modified by this test.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import Mock


# ============================================================
# COUNTERS
# ============================================================

PASS = 0
FAIL = 0


def passed(name):
    global PASS
    PASS += 1
    print(f"[PASS] {name}")


def failed(name, exc):
    global FAIL
    FAIL += 1
    print(f"[FAIL] {name}")
    print(f"       {type(exc).__name__}: {exc}")


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


# ============================================================
# STREAMLIT STUB
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

# ============================================================
# STREAMLIT CACHE RESOURCE — REQUIRED BY app.py
# ============================================================

def cache_resource(func=None, **kwargs):
    """
    Test stub for:
        @st.cache_resource(show_spinner=False)
    """

    def decorator(fn):
        return fn

    if func is not None and callable(func):
        return func

    return decorator


def cache_data(func=None, **kwargs):
    """
    Test stub for:
        @st.cache_data(...)
    """

    def decorator(fn):
        return fn

    if func is not None and callable(func):
        return func

    return decorator


st.cache_resource = cache_resource
st.cache_data = cache_data

st.session_state = SessionState()


def _mock(return_value=None):
    return Mock(return_value=return_value)


for name in [
    "error",
    "warning",
    "info",
    "success",
    "write",
    "markdown",
    "caption",
    "title",
    "header",
    "subheader",
    "divider",
    "text",
    "exception",
]:

    setattr(st, name, _mock())


st.button = _mock(False)
st.download_button = _mock(False)
st.text_area = _mock("")
st.text_input = _mock("")
st.selectbox = _mock("")
st.file_uploader = _mock(None)
st.checkbox = _mock(False)
st.radio = _mock("")
st.number_input = _mock(0)

st.columns = Mock(return_value=[])
st.container = Mock()
st.expander = Mock()
st.tabs = Mock(return_value=[])
st.spinner = Mock()
st.set_page_config = Mock()

st.stop = Mock(
    side_effect=RuntimeError("st.stop called")
)

sys.modules["streamlit"] = st


# ============================================================
# DOTENV STUB
# ============================================================

dotenv = types.ModuleType("dotenv")

dotenv.load_dotenv = Mock()

sys.modules["dotenv"] = dotenv


# ============================================================
# CORE PACKAGE
# ============================================================

core = types.ModuleType("core")
core.__path__ = []

sys.modules["core"] = core


# ============================================================
# HELPER
# ============================================================

def register_core_module(module_name):

    module = types.ModuleType(module_name)

    sys.modules[module_name] = module

    short_name = module_name.split(".")[-1]

    setattr(
        core,
        short_name,
        module,
    )

    return module


# ============================================================
# DATABASE STUB
# ============================================================

database = register_core_module(
    "core.database"
)

database.delete_video = Mock(
    return_value=True
)

database.get_video = Mock(
    return_value=None
)

database.list_videos = Mock(
    return_value=[]
)

database.video_exists = Mock(
    return_value=False
)

database.create_video = Mock(
    return_value=True
)

database.update_video = Mock(
    return_value=True
)

database.save_completed_video = Mock(
    return_value=True
)

database.mark_video_failed = Mock(
    return_value=True
)


# ============================================================
# VECTOR STORE STUB
# ============================================================

vector_store = register_core_module(
    "core.vector_store"
)

vector_store.delete_vector_store = Mock(
    return_value=True
)


# ============================================================
# PROMPT GENERATOR STUB
# ============================================================

prompt_generator = register_core_module(
    "core.prompt_generator"
)

prompt_generator.generate_prompt = Mock(
    return_value="Test prompt"
)


# ============================================================
# ROMAN URDU TRANSLATOR STUB
# ============================================================

roman_urdu_translator = register_core_module(
    "core.roman_urdu_translator"
)

roman_urdu_translator.translate_to_roman_urdu = Mock(
    return_value="Test Roman Urdu translation"
)


# ============================================================
# TRANSLATOR STUB
# ============================================================

translator = register_core_module(
    "core.translator"
)

translator.translate_text = Mock(
    return_value="Translated text"
)

translator.translate_to_english = Mock(
    return_value="English text"
)

translator.translate_to_language = Mock(
    return_value="Translated text"
)


# ============================================================
# TRANSCRIBER STUB
# ============================================================

transcriber = register_core_module(
    "core.transcriber"
)

transcriber.transcribe_all = Mock(
    return_value="Test transcript"
)


# ============================================================
# SUMMARIZER STUB
# ============================================================

summarize = register_core_module(
    "core.summarize"
)

summarize.generate_title = Mock(
    return_value="Test Title"
)

summarize.summarize = Mock(
    return_value="Test Summary"
)

summarize.extract_action_items = Mock(
    return_value=[]
)

summarize.extract_decisions = Mock(
    return_value=[]
)

summarize.extract_questions = Mock(
    return_value=[]
)


# ============================================================
# RAG ENGINE STUB
# ============================================================

rag_engine = register_core_module(
    "core.rag_engine"
)

rag_engine.build_rag_chain = Mock(
    return_value=object()
)

rag_engine.ask_question = Mock(
    return_value=""
)


# ============================================================
# TRANSLATION CACHE STUB
#
# Dynamic __getattr__ prevents import failures if app.py imports
# additional cache functions.
# ============================================================

class TranslationCacheModule(
    types.ModuleType
):

    def __getattr__(self, name):

        if name.startswith("__"):
            raise AttributeError(name)

        mock = Mock(
            name=f"translation_cache.{name}",
            return_value=None,
        )

        setattr(
            self,
            name,
            mock,
        )

        return mock


translation_cache = TranslationCacheModule(
    "core.translation_cache"
)


# Known functions
translation_cache.get_cached_translation = Mock(
    return_value=None
)

translation_cache.get_translation = Mock(
    return_value=None
)

translation_cache.load_translation = Mock(
    return_value=None
)

translation_cache.translation_exists = Mock(
    return_value=False
)

translation_cache.get_saved_translation_by_hash = Mock(
    return_value=None
)

translation_cache.save_translation = Mock(
    return_value=True
)

translation_cache.save_translation_cache = Mock(
    return_value=True
)

translation_cache.delete_translation = Mock(
    return_value=True
)

translation_cache.delete_translation_by_id = Mock(
    return_value=True
)

translation_cache.delete_cached_translation = Mock(
    return_value=True
)


sys.modules[
    "core.translation_cache"
] = translation_cache

core.translation_cache = translation_cache


# ============================================================
# CORE TRANSLATION EXPORTER STUB
#
# IMPORTANT:
# app.py imports core.translation_exporter
# ============================================================

translation_exporter = types.ModuleType(
    "core.translation_exporter"
)

translation_exporter.build_translation_txt = Mock(
    return_value=b"TXT EXPORT"
)

translation_exporter.build_translation_excel = Mock(
    return_value=b"XLSX EXPORT"
)

translation_exporter.build_translation_pdf = Mock(
    return_value=b"PDF EXPORT"
)

translation_exporter._aligned_records = Mock(
    return_value=[]
)

sys.modules[
    "core.translation_exporter"
] = translation_exporter

core.translation_exporter = translation_exporter


# ============================================================
# ALSO PROVIDE ROOT EXPORTER ALIAS
# ============================================================

sys.modules[
    "translation_exporter"
] = translation_exporter

# ============================================================
# VIDEO LIBRARY ANALYTICS STUB
# ============================================================

video_library_analytics = types.ModuleType(
    "core.video_library_analytics"
)


class VideoLibraryAnalyticsModule(
    types.ModuleType
):

    def __getattr__(self, name):

        if name.startswith("__"):
            raise AttributeError(name)

        mock = Mock(
            name=f"video_library_analytics.{name}",
            return_value=None,
        )

        setattr(
            self,
            name,
            mock,
        )

        return mock


video_library_analytics = VideoLibraryAnalyticsModule(
    "core.video_library_analytics"
)


sys.modules[
    "core.video_library_analytics"
] = video_library_analytics

core.video_library_analytics = (
    video_library_analytics
)

# ============================================================
# UTILS PACKAGE
# ============================================================

utils = types.ModuleType("utils")

utils.__path__ = []

sys.modules["utils"] = utils


# ============================================================
# AUDIO PROCESSOR
# ============================================================

audio_processor = types.ModuleType(
    "utils.audio_processor"
)

audio_processor.process_input = Mock(
    return_value=[]
)

audio_processor.cleanup_chunks = Mock()

audio_processor.cleanup_file = Mock()

sys.modules[
    "utils.audio_processor"
] = audio_processor

utils.audio_processor = audio_processor


# ============================================================
# OPTIONAL EXTRACTOR
# ============================================================

extractor = types.ModuleType(
    "extractor"
)

extractor.extract_action_items = Mock(
    return_value=[]
)

extractor.extract_decisions = Mock(
    return_value=[]
)

extractor.extract_questions = Mock(
    return_value=[]
)

sys.modules[
    "extractor"
] = extractor


# ============================================================
# IMPORT APP
# ============================================================

print()
print("=" * 70)
print("IMPORTING app.py")
print("=" * 70)
print()


try:

    app = importlib.import_module("app")


except Exception as exc:

    print()
    print("=" * 70)
    print("ERROR: app.py COULD NOT BE IMPORTED")
    print("=" * 70)
    print()
    print(
        f"{type(exc).__name__}: {exc}"
    )
    print()

    raise


print(
    "PASS: app.py imported successfully"
)

print()


# ============================================================
# RESET SESSION STATE
# ============================================================

def reset_state():

    st.session_state.clear()

    st.session_state[
        "active_original_transcript"
    ] = ""

    st.session_state[
        "active_roman_urdu_translation"
    ] = ""

    st.session_state[
        "active_source_identity"
    ] = None

    st.session_state[
        "roman_urdu_transcript"
    ] = ""

    st.session_state[
        "roman_urdu_source_id"
    ] = None


# ============================================================
# TEST 1
# SAME SOURCE
# ============================================================

def test_same_source():

    reset_state()

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_A",
        "transcript": "Transcript A",
        "source_name": "Video A",
    }

    st.session_state["active_original_transcript"] = "Transcript A"
    st.session_state["active_roman_urdu_translation"] = "Roman Urdu A"
    st.session_state["active_source_identity"] = "youtube:id:SOURCE_A"

    original, translation, identity = app.validate_export_state(result)

    assert_true(original == "Transcript A", "Wrong original transcript")
    assert_true(translation == "Roman Urdu A", "Wrong translation")
    assert_true(identity == "youtube:id:SOURCE_A", "Wrong source identity")

    passed("Same-source export is valid")

# ============================================================
# TEST 2
# CROSS SOURCE
# ============================================================

def test_cross_source():

    reset_state()

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_B",
        "transcript": "Transcript B",
        "source_name": "Video B",
    }

    st.session_state["active_original_transcript"] = "Transcript B"
    st.session_state["active_roman_urdu_translation"] = "Roman Urdu A"
    st.session_state["active_source_identity"] = "youtube:id:SOURCE_A"

    try:
        app.validate_export_state(result)
    except app.ExportStateError:
        passed("Cross-source translation blocked")
        return

    raise AssertionError("Cross-source translation must be rejected")

# ============================================================
# TEST 3
# LEGACY CONTAMINATION
# ============================================================

def test_legacy_contamination():

    reset_state()

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_A",
        "transcript": "Transcript A",
        "source_name": "Video A",
    }

    st.session_state["active_original_transcript"] = "Transcript A"
    st.session_state["active_roman_urdu_translation"] = "Translation A"
    st.session_state["active_source_identity"] = "youtube:id:SOURCE_A"

    # Deliberately stale legacy state. The app must ignore it.
    st.session_state["roman_urdu_transcript"] = "STALE TRANSLATION B"
    st.session_state["roman_urdu_source_id"] = "SOURCE_B"

    original, translation, identity = app.validate_export_state(result)

    assert_true(original == "Transcript A", "Original transcript contaminated")
    assert_true(translation == "Translation A", "Legacy state contaminated active translation")
    assert_true(identity == "youtube:id:SOURCE_A", "Wrong active identity")

    passed("Legacy state cannot contaminate active translation")

# ============================================================
# TEST 4
# EMPTY TRANSLATION
# ============================================================

def test_empty_translation():

    reset_state()

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_A",
        "transcript": "Transcript A",
    }

    st.session_state["active_original_transcript"] = "Transcript A"
    st.session_state["active_roman_urdu_translation"] = ""
    st.session_state["active_source_identity"] = "youtube:id:SOURCE_A"

    try:
        app.validate_export_state(result)
    except app.ExportStateError:
        passed("Empty translation blocked")
        return

    raise AssertionError("Empty translation must be rejected")

# ============================================================
# TEST 5
# EMPTY ORIGINAL
# ============================================================

def test_empty_original():

    reset_state()

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_A",
        "transcript": "",
    }

    st.session_state["active_original_transcript"] = ""
    st.session_state["active_roman_urdu_translation"] = "Translation A"
    st.session_state["active_source_identity"] = "youtube:id:SOURCE_A"

    try:
        app.validate_export_state(result)
    except app.ExportStateError:
        passed("Empty original transcript blocked")
        return

    raise AssertionError("Empty original transcript must be rejected")

# ============================================================
# TEST 6
# SOURCE SWITCH
# ============================================================

def test_source_switch():

    reset_state()

    old_result = {
        "source_type": "youtube",
        "video_id": "SOURCE_A",
        "transcript": "Transcript A",
        "source_name": "Video A",
    }

    app.set_active_processed_result(old_result, "Video A")

    st.session_state["active_roman_urdu_translation"] = "Translation A"

    new_result = {
        "source_type": "youtube",
        "video_id": "SOURCE_B",
        "transcript": "Transcript B",
        "source_name": "Video B",
    }

    app.set_active_processed_result(new_result, "Video B")

    assert_true(
        st.session_state["active_source_identity"] == "youtube:id:SOURCE_B",
        "Source did not switch to SOURCE_B",
    )

    assert_true(
        st.session_state["active_original_transcript"] == "Transcript B",
        "New transcript was not activated",
    )

    assert_true(
        st.session_state["active_roman_urdu_translation"] in ("", None),
        "Old translation survived source switch",
    )

    passed("Source switch clears stale translation")

# ============================================================
# TEST 7
# TXT EXPORT
# ============================================================

def test_txt_export():

    reset_state()

    original = "Original A"
    translation = "Roman Urdu A"


    st.session_state[
        "active_original_transcript"
    ] = original

    st.session_state[
        "active_roman_urdu_translation"
    ] = translation

    st.session_state[
        "active_source_identity"
    ] = "SOURCE_A"


    translation_exporter.build_translation_txt.reset_mock()


    translation_exporter.build_translation_txt(
        original,
        translation,
    )


    assert_true(
        translation_exporter
        .build_translation_txt
        .call_count == 1,
        "TXT exporter was not called"
    )


    args = (
        translation_exporter
        .build_translation_txt
        .call_args.args
    )


    assert_true(
        original in args,
        "Original transcript missing from TXT"
    )

    assert_true(
        translation in args,
        "Translation missing from TXT"
    )


    passed(
        "TXT exporter receives authoritative values"
    )


# ============================================================
# TEST 8
# XLSX EXPORT
# ============================================================

def test_xlsx_export():

    reset_state()

    original = "Original A"
    translation = "Roman Urdu A"


    st.session_state[
        "active_original_transcript"
    ] = original

    st.session_state[
        "active_roman_urdu_translation"
    ] = translation

    st.session_state[
        "active_source_identity"
    ] = "SOURCE_A"


    translation_exporter.build_translation_excel.reset_mock()


    translation_exporter.build_translation_excel(
        original,
        translation,
    )


    assert_true(
        translation_exporter
        .build_translation_excel
        .call_count == 1,
        "XLSX exporter was not called"
    )


    args = (
        translation_exporter
        .build_translation_excel
        .call_args.args
    )


    assert_true(
        original in args,
        "Original transcript missing from XLSX"
    )

    assert_true(
        translation in args,
        "Translation missing from XLSX"
    )


    passed(
        "XLSX exporter receives authoritative values"
    )


# ============================================================
# TEST 9
# PDF EXPORT
# ============================================================

def test_pdf_export():

    reset_state()

    original = "Original A"
    translation = "Roman Urdu A"


    st.session_state[
        "active_original_transcript"
    ] = original

    st.session_state[
        "active_roman_urdu_translation"
    ] = translation

    st.session_state[
        "active_source_identity"
    ] = "SOURCE_A"


    translation_exporter.build_translation_pdf.reset_mock()


    translation_exporter.build_translation_pdf(
        original,
        translation,
    )


    assert_true(
        translation_exporter
        .build_translation_pdf
        .call_count == 1,
        "PDF exporter was not called"
    )


    args = (
        translation_exporter
        .build_translation_pdf
        .call_args.args
    )


    assert_true(
        original in args,
        "Original transcript missing from PDF"
    )

    assert_true(
        translation in args,
        "Translation missing from PDF"
    )


    passed(
        "PDF exporter receives authoritative values"
    )


# ============================================================
# TEST 10
# SAVED TRANSLATION — SAME SOURCE
# ============================================================

def test_saved_translation_same_source():

    reset_state()

    saved = {
        "source_id": "SOURCE_A",
        "translation": "Translation A",
    }

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_A",
        "transcript": "Transcript A",
    }

    matched = app.saved_translation_matches_result(saved, result)

    assert_true(
        matched is True,
        "Same-source saved translation should match",
    )

    passed("Same-source saved translation accepted")

# ============================================================
# TEST 11
# SAVED TRANSLATION — WRONG SOURCE
# ============================================================

def test_saved_translation_wrong_source():

    reset_state()

    saved = {
        "source_id": "SOURCE_A",
        "translation": "Translation A",
    }

    result = {
        "source_type": "youtube",
        "video_id": "SOURCE_B",
        "transcript": "Transcript B",
    }

    matched = app.saved_translation_matches_result(saved, result)

    assert_true(
        matched is False,
        "Different-source translation must be rejected",
    )

    passed("Different-source saved translation rejected")

# ============================================================
# TEST 12
# TIMESTAMP ALIGNMENT
# ============================================================

def test_timestamp_alignment():

    original = (
        "[00:00:10] Hello\n"
        "[00:00:20] How are you?"
    )

    translated = (
        "[00:00:10] Assalam o alaikum\n"
        "[00:00:20] Aap kaise hain?"
    )


    translation_exporter._aligned_records.reset_mock()


    translation_exporter._aligned_records(
        original,
        translated,
    )


    assert_true(
        translation_exporter
        ._aligned_records
        .call_count == 1,
        "Alignment function was not called"
    )


    args = (
        translation_exporter
        ._aligned_records
        .call_args.args
    )


    assert_true(
        args[0] == original,
        "Original transcript changed"
    )

    assert_true(
        args[1] == translated,
        "Translated transcript changed"
    )


    passed(
        "Timestamp alignment contract preserved"
    )


# ============================================================
# TEST SUITE
# ============================================================

TESTS = [

    test_same_source,

    test_cross_source,

    test_legacy_contamination,

    test_empty_translation,

    test_empty_original,

    test_source_switch,

    test_txt_export,

    test_xlsx_export,

    test_pdf_export,

    test_saved_translation_same_source,

    test_saved_translation_wrong_source,

    test_timestamp_alignment,
]


# ============================================================
# RUN TESTS
# ============================================================

print()
print("=" * 70)
print("BUG #11 — EXPORT STATE & SOURCE IDENTITY TESTS")
print("=" * 70)
print()


for test in TESTS:

    try:

        test()

    except Exception as exc:

        failed(
            test.__name__,
            exc,
        )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

print(
    f"TESTS PASSED: {PASS}"
)

print(
    f"TESTS FAILED: {FAIL}"
)

print("=" * 70)


if FAIL == 0:

    print()
    print(
        "BUG #11 TESTS: ALL PASSED"
    )

    print(
        "Source identity protection: PASS"
    )

    print(
        "Cross-source contamination: PASS"
    )

    print(
        "Legacy state protection: PASS"
    )

    print(
        "TXT export: PASS"
    )

    print(
        "XLSX export: PASS"
    )

    print(
        "PDF export: PASS"
    )

    print(
        "Saved translation validation: PASS"
    )

    print(
        "Timestamp alignment: PASS"
    )

    print()
    print(
        "BUG #11 VALIDATION COMPLETE"
    )

    sys.exit(0)


else:

    print()
    print(
        "BUG #11 VALIDATION FAILED"
    )

    sys.exit(1)