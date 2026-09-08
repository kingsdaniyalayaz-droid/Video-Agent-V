"""
BUG #10 TEST SUITE
==================

Cached result must respect the CURRENT translation configuration.

Tests:
1. Exact compatible cache
2. Translation newly requested
3. Different target language
4. Translation disabled
5. Different source language
6. Compatible cache avoids processing
7. Language alias/case normalization

Regression:
8. Bug #7
9. Bug #8
10. Bug #9

No real YouTube / Whisper / Mistral calls.
"""

from __future__ import annotations

import sys
import traceback
import types
from unittest import mock


# ============================================================
# HERMETIC MODULE STUBS
# ============================================================

def _module(name, package="", **attrs):
    mod = types.ModuleType(name)
    mod.__package__ = package
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


# ------------------------------------------------------------
# dotenv
# ------------------------------------------------------------

_module(
    "dotenv",
    load_dotenv=lambda *args, **kwargs: None,
)


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------

_module(
    "utils",
    package="utils",
)

_module(
    "utils.audio_processor",
    package="utils",
    process_input=lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            NotImplementedError(
                "process_input must be mocked"
            )
        )
    ),
    cleanup_chunks=lambda *args, **kwargs: None,
)


# ------------------------------------------------------------
# core
# ------------------------------------------------------------

_module(
    "core",
    package="core",
)


# ------------------------------------------------------------
# transcriber
# ------------------------------------------------------------

_module(
    "core.transcriber",
    package="core",
    transcribe_all=lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            NotImplementedError(
                "transcribe_all must be mocked"
            )
        )
    ),
)


# ------------------------------------------------------------
# translator
# ------------------------------------------------------------

_module(
    "core.translator",
    package="core",
    translate_text=lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            NotImplementedError(
                "translate_text must be mocked"
            )
        )
    ),
)


# ------------------------------------------------------------
# summarize
# ------------------------------------------------------------

_module(
    "core.summarize",
    package="core",
    summarize=lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            NotImplementedError(
                "summarize must be mocked"
            )
        )
    ),
    generate_title=lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            NotImplementedError(
                "generate_title must be mocked"
            )
        )
    ),
)


# ------------------------------------------------------------
# RAG
# ------------------------------------------------------------

_module(
    "core.rag_engine",
    package="core",
    build_rag_chain=lambda *args, **kwargs: object(),
    ask_question=lambda *args, **kwargs: "",
)


# ------------------------------------------------------------
# DATABASE
#
# IMPORTANT:
# main.py imports video_exists as well.
# This was missing in the previous test stub.
# ------------------------------------------------------------

_module(
    "core.database",
    package="core",

    get_video=lambda *args, **kwargs: (
        (_ for _ in ()).throw(
            NotImplementedError(
                "get_video must be mocked"
            )
        )
    ),

    create_video=lambda *args, **kwargs: None,

    mark_video_failed=lambda *args, **kwargs: None,

    save_completed_video=lambda *args, **kwargs: None,

    update_video=lambda *args, **kwargs: None,

    # Required by current main.py import
    video_exists=lambda *args, **kwargs: False,
)


# Package attributes
sys.modules["utils"].audio_processor = (
    sys.modules["utils.audio_processor"]
)

sys.modules["core"].transcriber = (
    sys.modules["core.transcriber"]
)

sys.modules["core"].translator = (
    sys.modules["core.translator"]
)

sys.modules["core"].summarize = (
    sys.modules["core.summarize"]
)

sys.modules["core"].rag_engine = (
    sys.modules["core.rag_engine"]
)

sys.modules["core"].database = (
    sys.modules["core.database"]
)


# ------------------------------------------------------------
# extractor
# ------------------------------------------------------------

_module(
    "extractor",

    extract_action_items=lambda transcript: [],

    extract_decisions=lambda transcript: [],

    extract_questions=lambda transcript: [],
)


# ============================================================
# IMPORT ACTUAL APPLICATION
# ============================================================

import main


# ============================================================
# TEST DATA
# ============================================================

YT_SOURCE = (
    "https://www.youtube.com/watch?v=ABC123"
)

YT_ID = "ABC123"


ORIGINAL_TRANSCRIPT = (
    "Meeting started. We discussed the roadmap "
    "for the product launch next quarter. "
    "We agreed on a two-phase release and "
    "assigned owners for every milestone. "
    "The budget was approved and we scheduled "
    "the beta for March."
)


URDU_TRANSLATION = (
    "اجلاس شروع ہوا۔ ہم نے اگلی سہ ماہی میں "
    "پروڈکٹ لانچ کے حوالے سے روڈ میپ پر "
    "تبادلہ خیال کیا۔"
)


HINDI_TRANSLATION = (
    "बैठक शुरू हुई। हमने अगली तिमाही में "
    "उत्पाद लॉन्च के लिए रोडमैप पर चर्चा की।"
)


# ============================================================
# CACHED RECORD FACTORY
# ============================================================

def cached_record(**overrides):

    record = {

        "status": "completed",

        "video_id": YT_ID,

        "language": "en",

        "transcript": ORIGINAL_TRANSCRIPT,

        # Same as transcript = no translation
        "translated_transcript": (
            ORIGINAL_TRANSCRIPT
        ),

        "translation_enabled": False,

        "target_language": None,

        "title": "Cached Title",

        "summary": "Cached summary.",

        "actions": [
            "Assign owners"
        ],

        "decisions": [
            "Two-phase release"
        ],

        "questions": [
            "When is the beta?"
        ],

        "analysis_status": "completed",

        "analysis_error": None,

        "duration": 180,

        "chunk_count": 2,
    }

    record.update(overrides)

    return record


# ============================================================
# PATCH ALL EXPENSIVE DEPENDENCIES
# ============================================================

def patch_pipeline_mocks():

    patchers = [

        mock.patch.object(
            main,
            "get_video",
        ),

        mock.patch.object(
            main,
            "create_video",
        ),

        mock.patch.object(
            main,
            "update_video",
        ),

        mock.patch.object(
            main,
            "save_completed_video",
        ),

        mock.patch.object(
            main,
            "mark_video_failed",
        ),

        mock.patch.object(
            main,
            "process_input",
        ),

        mock.patch.object(
            main,
            "transcribe_all",
        ),

        mock.patch.object(
            main,
            "translate_text",
        ),

        mock.patch.object(
            main,
            "generate_title",
        ),

        mock.patch.object(
            main,
            "summarize",
        ),

        mock.patch.object(
            main,
            "build_rag_chain",
        ),

        mock.patch.object(
            main,
            "cleanup_chunks",
        ),
    ]

    for patcher in patchers:
        patcher.start()

    return patchers


def stop_mocks(patchers):

    for patcher in reversed(patchers):
        patcher.stop()


# ============================================================
# FULL PIPELINE CONFIGURATION
# ============================================================

def configure_full_pipeline(
    transcript="Processed transcript.",
):

    main.process_input.return_value = [
        "/tmp/chunk_01.mp3"
    ]

    main.transcribe_all.return_value = (
        transcript
    )

    main.generate_title.return_value = (
        "Processed Title"
    )

    main.summarize.return_value = (
        "Processed summary."
    )

    main.build_rag_chain.return_value = (
        object()
    )


# ============================================================
# TEST #1
# EXACT COMPATIBLE CACHE
# ============================================================

def test_01_exact_cache_match():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = (
            cached_record()
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=False,

            target_language=None,
        )

        assert result is not None

        assert (
            result.metadata.get("cached")
            is True
        )

        assert (
            result.transcript
            == ORIGINAL_TRANSCRIPT
        )

        assert (
            result.translated_transcript
            == ORIGINAL_TRANSCRIPT
        )

        assert (
            main.translate_text.call_count
            == 0
        )

        assert (
            main.process_input.call_count
            == 0
        )

        assert (
            main.transcribe_all.call_count
            == 0
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST #2
# TRANSLATION NEWLY REQUESTED
# ============================================================

def test_02_translation_newly_requested():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = (
            cached_record()
        )

        main.translate_text.side_effect = (
            lambda text,
            source_language,
            target_language:

                "URDU:" + text
                if target_language == "urdu"
                else "X:" + text
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=True,

            target_language="urdu",
        )

        # No new processing
        assert (
            main.process_input.call_count
            == 0
        )

        assert (
            main.transcribe_all.call_count
            == 0
        )

        # Translation runs
        assert (
            main.translate_text.call_count
            == 1
        )

        call = (
            main.translate_text.call_args
        )

        assert (
            call.kwargs["source_language"]
            == "english"
        )

        assert (
            call.kwargs["target_language"]
            == "urdu"
        )

        assert (
            call.kwargs["text"]
            == ORIGINAL_TRANSCRIPT
        )

        # Correct result
        assert (
            result.transcript
            == ORIGINAL_TRANSCRIPT
        )

        assert (
            result.translated_transcript
            == "URDU:" + ORIGINAL_TRANSCRIPT
        )

        # Persist translation
        main.update_video.assert_any_call(

            YT_ID,

            translated_transcript=(
                "URDU:" + ORIGINAL_TRANSCRIPT
            ),

            translation_enabled=True,

            target_language="urdu",
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST #3
# DIFFERENT TARGET LANGUAGE
# ============================================================

def test_03_different_target_language():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = (
            cached_record(

                translated_transcript=(
                    URDU_TRANSLATION
                ),

                translation_enabled=True,

                target_language="urdu",
            )
        )

        main.translate_text.side_effect = (
            lambda text,
            source_language,
            target_language:

                "HI:" + text
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=True,

            target_language="hindi",
        )

        assert (
            main.translate_text.call_count
            == 1
        )

        call = (
            main.translate_text.call_args
        )

        assert (
            call.kwargs["target_language"]
            == "hindi"
        )

        # Must translate ORIGINAL transcript
        assert (
            call.kwargs["text"]
            == ORIGINAL_TRANSCRIPT
        )

        # Must not return Urdu as Hindi
        assert (
            result.translated_transcript
            == "HI:" + ORIGINAL_TRANSCRIPT
        )

        assert (
            result.translated_transcript
            != URDU_TRANSLATION
        )

        assert (
            main.process_input.call_count
            == 0
        )

        assert (
            main.transcribe_all.call_count
            == 0
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST #4
# TRANSLATION DISABLED
# ============================================================

def test_04_translation_disabled():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = (
            cached_record(

                translated_transcript=(
                    URDU_TRANSLATION
                ),

                translation_enabled=True,

                target_language="urdu",
            )
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=False,

            target_language=None,
        )

        assert (
            result.translated_transcript
            == ORIGINAL_TRANSCRIPT
        )

        assert (
            result.translated_transcript
            != URDU_TRANSLATION
        )

        assert (
            main.translate_text.call_count
            == 0
        )

        assert (
            main.process_input.call_count
            == 0
        )

        assert (
            main.transcribe_all.call_count
            == 0
        )

        # Existing stored translation must
        # not be overwritten.
        for call in (
            main.update_video.call_args_list
        ):

            assert (
                "translated_transcript"
                not in call.kwargs
            )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST #5
# DIFFERENT SOURCE LANGUAGE
# ============================================================

def test_05_different_source_language():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = (
            cached_record(
                language="en"
            )
        )

        configure_full_pipeline(
            transcript=(
                "اردو میں مکمل نشست کا "
                "ٹرانسکرپٹ۔"
            )
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="ur",

            translate=False,

            target_language=None,
        )

        # Full processing must run
        assert (
            main.process_input.call_count
            == 1
        )

        assert (
            main.transcribe_all.call_count
            == 1
        )

        assert (
            main.transcribe_all.call_args
            .kwargs["language"]
            == "ur"
        )

        assert (
            result.language
            == "ur"
        )

        assert (
            result.transcript
            != ORIGINAL_TRANSCRIPT
        )

        assert (
            result.metadata.get("cached")
            is not True
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST #6
# COMPATIBLE CACHE AVOIDS PROCESSING
# ============================================================

def test_06_compatible_cache_avoids_processing():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = (
            cached_record()
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=False,

            target_language=None,
        )

        assert (
            main.process_input.call_count
            == 0
        )

        assert (
            main.transcribe_all.call_count
            == 0
        )

        assert (
            main.translate_text.call_count
            == 0
        )

        assert (
            result.metadata.get("cached")
            is True
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST #7
# LANGUAGE ALIAS / CASE NORMALIZATION
# ============================================================

def test_07_case_and_alias_normalization():

    patchers = patch_pipeline_mocks()

    try:

        # ----------------------------------------------------
        # English alias
        # ----------------------------------------------------

        main.get_video.return_value = (
            cached_record(
                language="en"
            )
        )

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="English",

            translate=False,

            target_language=None,
        )

        assert (
            main.process_input.call_count
            == 0
        )

        assert (
            result.transcript
            == ORIGINAL_TRANSCRIPT
        )

        # ----------------------------------------------------
        # Hindi alias
        # ----------------------------------------------------

        main.get_video.return_value = (
            cached_record()
        )

        main.translate_text.side_effect = (
            lambda text,
            source_language,
            target_language:

                "HI:" + text
        )

        main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=True,

            target_language="hi",
        )

        assert (
            main.translate_text
            .call_args
            .kwargs["target_language"]
            == "hindi"
        )

        main.translate_text.reset_mock()

        main.update_video.reset_mock()

        # ----------------------------------------------------
        # HINDI uppercase
        # ----------------------------------------------------

        main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=True,

            target_language="HINDI",
        )

        assert (
            main.translate_text
            .call_args
            .kwargs["target_language"]
            == "hindi"
        )

        # ----------------------------------------------------
        # Urdu stored code
        # ----------------------------------------------------

        main.get_video.return_value = (
            cached_record(

                translated_transcript=(
                    URDU_TRANSLATION
                ),

                translation_enabled=True,

                target_language="ur",
            )
        )

        main.translate_text.reset_mock()

        result = main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            translate=True,

            target_language="urdu",
        )

        # Alias matches, therefore full cache hit
        assert (
            main.translate_text.call_count
            == 0
        )

        assert (
            result.translated_transcript
            == URDU_TRANSLATION
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# REGRESSION #7
# ============================================================

def test_regression_07_chunked_transcription():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = None

        configure_full_pipeline()

        main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            chunk_minutes=15,
        )

        assert (
            main.process_input.call_count
            == 1
        )

        assert (
            main.process_input
            .call_args
            .kwargs["chunk_minutes"]
            == 15
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# REGRESSION #8
# ============================================================

def test_regression_08_translation_delegation():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = None

        configure_full_pipeline()

        callback = (
            lambda event: None
        )

        main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",

            progress_callback=callback,
        )

        assert (
            main.transcribe_all.call_count
            == 1
        )

        call = (
            main.transcribe_all.call_args
        )

        assert (
            call.kwargs["chunks"]
            == ["/tmp/chunk_01.mp3"]
        )

        assert (
            call.kwargs["translate"]
            is False
        )

        assert (
            call.kwargs["language"]
            == "en"
        )

        assert (
            call.kwargs["progress_callback"]
            is callback
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# REGRESSION #9
# ============================================================

def test_regression_09_cleanup():

    patchers = patch_pipeline_mocks()

    try:

        main.get_video.return_value = None

        main.process_input.return_value = [

            "/tmp/current_01.mp3",

            "/tmp/current_02.mp3",
        ]

        main.transcribe_all.return_value = (
            "Transcript text."
        )

        main.generate_title.return_value = (
            "Title"
        )

        main.summarize.return_value = (
            "Summary."
        )

        main.build_rag_chain.return_value = (
            object()
        )

        main.run_pipeline(

            source=YT_SOURCE,

            source_type="youtube",

            language="en",
        )

        main.cleanup_chunks.assert_called_once_with(

            [
                "/tmp/current_01.mp3",
                "/tmp/current_02.mp3",
            ]
        )

    finally:

        stop_mocks(patchers)


# ============================================================
# TEST GROUPS
# ============================================================

BUG10_TESTS = [

    test_01_exact_cache_match,

    test_02_translation_newly_requested,

    test_03_different_target_language,

    test_04_translation_disabled,

    test_05_different_source_language,

    test_06_compatible_cache_avoids_processing,

    test_07_case_and_alias_normalization,
]


REGRESSION_TESTS = [

    test_regression_07_chunked_transcription,

    test_regression_08_translation_delegation,

    test_regression_09_cleanup,
]


# ============================================================
# RUNNER
# ============================================================

def run_group(
    tests,
    label,
):

    print()

    print("=" * 70)

    print(label)

    print("=" * 70)

    passed = 0

    failed = 0

    for test in tests:

        try:

            test()

            passed += 1

            print(
                f"[PASS] {test.__name__}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[FAIL] {test.__name__}"
            )

            print(
                f"Reason: {exc}"
            )

            traceback.print_exc()

    print("-" * 70)

    print(
        f"{label}: "
        f"{passed} passed, "
        f"{failed} failed"
    )

    return failed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "BUG #10 — CACHE TRANSLATION "
        "CONFIGURATION TEST"
    )

    print("=" * 70)

    failures = 0

    failures += run_group(

        BUG10_TESTS,

        "BUG #10 TESTS",
    )

    failures += run_group(

        REGRESSION_TESTS,

        "REGRESSION TESTS — BUG #7 / #8 / #9",
    )

    print()

    print("=" * 70)

    if failures == 0:

        print(
            "RESULT: ALL TESTS PASSED"
        )

        print()

        print(
            "Bug #10: PASS"
        )

        print(
            "Bug #7 regression: PASS"
        )

        print(
            "Bug #8 regression: PASS"
        )

        print(
            "Bug #9 regression: PASS"
        )

        print()

        print(
            "BUG #10 VALIDATION COMPLETE"
        )

        sys.exit(0)

    else:

        print(
            f"RESULT: {failures} TEST(S) FAILED"
        )

        sys.exit(1)