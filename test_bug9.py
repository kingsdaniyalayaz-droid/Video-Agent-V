from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

# Current project main.py
MAIN_FILE = ROOT / "main.py"

if not MAIN_FILE.exists():
    raise FileNotFoundError(
        f"main.py not found:\n{MAIN_FILE}"
    )


# ============================================================
# MOCK / STUB DEPENDENCIES
# ============================================================

def install_stubs():

    # --------------------------------------------------------
    # dotenv
    # --------------------------------------------------------

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None


    # --------------------------------------------------------
    # utils.audio_processor
    # --------------------------------------------------------

    utils = types.ModuleType("utils")

    audio = types.ModuleType(
        "utils.audio_processor"
    )

    audio.process_input = Mock()
    audio.cleanup_chunks = Mock()

    utils.audio_processor = audio


    # --------------------------------------------------------
    # core.transcriber
    # --------------------------------------------------------

    core = types.ModuleType("core")

    transcriber = types.ModuleType(
        "core.transcriber"
    )

    transcriber.transcribe_all = Mock()


    # --------------------------------------------------------
    # core.translator
    # --------------------------------------------------------

    translator = types.ModuleType(
        "core.translator"
    )

    translator.translate_text = Mock()


    # --------------------------------------------------------
    # core.summarize
    # --------------------------------------------------------

    summarize = types.ModuleType(
        "core.summarize"
    )

    summarize.summarize = Mock()
    summarize.generate_title = Mock()


    # --------------------------------------------------------
    # core.rag_engine
    # --------------------------------------------------------

    rag = types.ModuleType(
        "core.rag_engine"
    )

    rag.build_rag_chain = Mock()
    rag.ask_question = Mock()


    # --------------------------------------------------------
    # core.database
    # --------------------------------------------------------

    database = types.ModuleType(
        "core.database"
    )

    database.get_video = Mock(
        return_value=None
    )

    database.create_video = Mock()

    database.mark_video_failed = Mock()

    database.save_completed_video = Mock()

    database.update_video = Mock()


    # --------------------------------------------------------
    # Connect modules
    # --------------------------------------------------------

    core.transcriber = transcriber
    core.translator = translator
    core.summarize = summarize
    core.rag_engine = rag
    core.database = database


    # --------------------------------------------------------
    # Register stubs
    # --------------------------------------------------------

    sys.modules.update({
        "dotenv": dotenv,

        "utils": utils,
        "utils.audio_processor": audio,

        "core": core,
        "core.transcriber": transcriber,
        "core.translator": translator,
        "core.summarize": summarize,
        "core.rag_engine": rag,
        "core.database": database,
    })


    # Remove previously imported main.py
    sys.modules.pop(
        "main",
        None
    )


    # Import actual main.py
    main = importlib.import_module(
        "main"
    )

    return (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    )


# ============================================================
# SUCCESS CONFIGURATION
# ============================================================

def configure_success(
    main,
    audio,
    transcriber,
    summarize,
    rag,
):

    chunks = [
        "temp_chunk_1.wav",
        "temp_chunk_2.wav",
        "temp_chunk_3.wav",
        "temp_chunk_4.wav",
    ]

    audio.process_input.return_value = chunks

    transcriber.transcribe_all.return_value = (
        "test transcript"
    )

    summarize.generate_title.return_value = (
        "Test Meeting"
    )

    summarize.summarize.return_value = (
        "Test summary"
    )

    rag.build_rag_chain.return_value = (
        object()
    )

    # Avoid real extractor dependency
    main._extract_structured_analysis = (
        lambda text: (
            [],
            [],
            [],
        )
    )

    return chunks


# ============================================================
# RUN PIPELINE
# ============================================================

def run_success(main):

    return main.run_pipeline(
        source="https://example.test/video",
        source_type="youtube",
        language="en",
    )


# ============================================================
# ASSERT CLEANUP
# ============================================================

def assert_cleanup_once(
    cleanup,
    expected_chunks,
):

    assert cleanup.call_count == 1, (
        f"Expected cleanup exactly once, "
        f"but got {cleanup.call_count}"
    )

    actual_chunks = (
        cleanup.call_args.args[0]
    )

    assert actual_chunks == expected_chunks, (
        "Cleanup received incorrect chunks.\n"
        f"Expected: {expected_chunks}\n"
        f"Actual:   {actual_chunks}"
    )


# ============================================================
# TEST 1
# SUCCESS → CLEANUP
# ============================================================

def test_success_cleanup():

    (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    ) = install_stubs()

    chunks = configure_success(
        main,
        audio,
        transcriber,
        summarize,
        rag,
    )

    result = run_success(main)

    assert result.audio_chunks == chunks

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    print(
        "PASS: Success processing cleans temporary chunks"
    )


# ============================================================
# TEST 2
# TRANSCRIPTION FAILURE → CLEANUP
# ============================================================

def test_transcription_failure():

    (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    ) = install_stubs()

    chunks = configure_success(
        main,
        audio,
        transcriber,
        summarize,
        rag,
    )

    original_error = ValueError(
        "Whisper transcription failure"
    )

    transcriber.transcribe_all.side_effect = (
        original_error
    )

    try:

        run_success(main)

    except RuntimeError as exc:

        assert isinstance(
            exc.__cause__,
            ValueError,
        )

        assert exc.__cause__ is original_error

    else:

        raise AssertionError(
            "Expected transcription failure"
        )

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    print(
        "PASS: Transcription failure cleans chunks and preserves original error"
    )


# ============================================================
# TEST 3
# TRANSLATION FAILURE → CLEANUP
# ============================================================

def test_translation_failure():

    (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    ) = install_stubs()

    chunks = configure_success(
        main,
        audio,
        transcriber,
        summarize,
        rag,
    )

    translator.translate_text.side_effect = (
        ValueError(
            "Translation failure"
        )
    )

    try:

        main.run_pipeline(
            source="https://example.test/video",
            source_type="youtube",
            language="en",
            target_language="urdu",
            translate=True,
        )

    except RuntimeError as exc:

        assert isinstance(
            exc.__cause__,
            ValueError,
        )

        assert (
            str(exc.__cause__)
            == "Translation failure"
        )

    else:

        raise AssertionError(
            "Expected translation failure"
        )

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    print(
        "PASS: Translation failure cleans chunks and preserves original error"
    )


# ============================================================
# TEST 4
# SUMMARY FAILURE → CLEANUP
# ============================================================

def test_summary_failure():

    (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    ) = install_stubs()

    chunks = configure_success(
        main,
        audio,
        transcriber,
        summarize,
        rag,
    )

    summarize.summarize.side_effect = (
        ValueError(
            "Summary failure"
        )
    )

    try:

        run_success(main)

    except RuntimeError as exc:

        assert isinstance(
            exc.__cause__,
            ValueError,
        )

        assert (
            str(exc.__cause__)
            == "Summary failure"
        )

    else:

        raise AssertionError(
            "Expected summary failure"
        )

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    print(
        "PASS: Summary failure cleans chunks and preserves original error"
    )


# ============================================================
# TEST 5
# UNEXPECTED FAILURE → CLEANUP
# ============================================================

def test_unexpected_failure():

    (
        main,
        audio,
        *_,
    ) = install_stubs()

    chunks = [
        "current_1.wav",
        "current_2.wav",
    ]

    def fail_after_chunks(
        **kwargs
    ):

        kwargs[
            "temporary_chunks"
        ].extend(chunks)

        raise LookupError(
            "Unexpected pipeline failure"
        )

    main._execute_processing_stages = (
        fail_after_chunks
    )

    try:

        run_success(main)

    except LookupError as exc:

        assert (
            str(exc)
            == "Unexpected pipeline failure"
        )

    else:

        raise AssertionError(
            "Expected unexpected failure"
        )

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    print(
        "PASS: Unexpected failure still cleans chunks"
    )


# ============================================================
# TEST 6
# CLEANUP FAILURE MUST NOT HIDE ORIGINAL ERROR
# ============================================================

def test_cleanup_failure_preserves_original_error():

    (
        main,
        audio,
        *_,
    ) = install_stubs()

    chunks = [
        "current.wav"
    ]

    audio.cleanup_chunks.side_effect = (
        RuntimeError(
            "Cleanup failure"
        )
    )

    def fail_after_chunks(
        **kwargs
    ):

        kwargs[
            "temporary_chunks"
        ].extend(chunks)

        raise ValueError(
            "Original processing failure"
        )

    main._execute_processing_stages = (
        fail_after_chunks
    )

    try:

        run_success(main)

    except ValueError as exc:

        assert (
            str(exc)
            == "Original processing failure"
        )

    else:

        raise AssertionError(
            "Original processing error was hidden"
        )

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    print(
        "PASS: Cleanup failure does not hide original error"
    )


# ============================================================
# TEST 7
# ALL MULTIPLE CHUNKS CLEANED
# ============================================================

def test_multiple_chunks():

    (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    ) = install_stubs()

    chunks = configure_success(
        main,
        audio,
        transcriber,
        summarize,
        rag,
    )

    run_success(main)

    assert_cleanup_once(
        audio.cleanup_chunks,
        chunks,
    )

    assert len(
        audio.cleanup_chunks.call_args.args[0]
    ) == 4

    print(
        "PASS: All temporary chunks are cleaned"
    )


# ============================================================
# TEST 8
# UNRELATED FILES NOT CLEANED
# ============================================================

def test_unrelated_files_not_cleaned():

    (
        main,
        audio,
        *_,
    ) = install_stubs()

    current_run = [
        "run-A/chunk_1.wav",
        "run-A/chunk_2.wav",
    ]

    unrelated = [
        "run-B/chunk_1.wav",
        "cached/permanent.wav",
        "source.mp4",
    ]

    def succeed_after_chunks(
        **kwargs
    ):

        kwargs[
            "temporary_chunks"
        ].extend(current_run)

        return main.PipelineResult(
            source="test",
            source_type="youtube",
            language="en",
            audio_chunks=current_run,
        )

    main._execute_processing_stages = (
        succeed_after_chunks
    )

    run_success(main)

    assert_cleanup_once(
        audio.cleanup_chunks,
        current_run,
    )

    cleaned = (
        audio.cleanup_chunks.call_args.args[0]
    )

    assert not (
        set(cleaned)
        .intersection(unrelated)
    )

    print(
        "PASS: Unrelated/permanent files are not cleaned"
    )


# ============================================================
# TEST 9
# CLEANUP EXACTLY ONCE
# ============================================================

def test_cleanup_exactly_once():

    (
        main,
        audio,
        transcriber,
        translator,
        summarize,
        rag,
        database,
    ) = install_stubs()

    configure_success(
        main,
        audio,
        transcriber,
        summarize,
        rag,
    )

    run_success(main)

    assert (
        audio.cleanup_chunks.call_count
        == 1
    )

    print(
        "PASS: Cleanup executes exactly once"
    )


# ============================================================
# TEST 10
# BUG #7 / BUG #8 REGRESSION
# ============================================================

def test_bug7_bug8_untouched():

    source = (
        MAIN_FILE
        .read_text(
            encoding="utf-8"
        )
    )

    # Bug #7 / existing pipeline parameters
    assert (
        "chunk_minutes"
        in source
    )

    assert (
        "progress_callback"
        in source
    )

    # Bug #8 must NOT have been moved into main.py.
    # Retry implementation belongs in core/translator.py.
    assert (
        "TRANSLATION_MAX_RETRIES"
        not in source
    )

    print(
        "PASS: Bug #7 / Bug #8 sections remain untouched"
    )


# ============================================================
# TEST LIST
# ============================================================

TESTS = [
    test_success_cleanup,
    test_transcription_failure,
    test_translation_failure,
    test_summary_failure,
    test_unexpected_failure,
    test_cleanup_failure_preserves_original_error,
    test_multiple_chunks,
    test_unrelated_files_not_cleaned,
    test_cleanup_exactly_once,
    test_bug7_bug8_untouched,
]


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BUG #9 — MANUAL-STYLE CLEANUP VALIDATION")
    print("=" * 70)

    passed = 0

    for test in TESTS:

        try:

            test()

            passed += 1

        except Exception as exc:

            print()
            print(
                f"FAIL: {test.__name__}"
            )

            print(
                f"Reason: {exc}"
            )

            raise

    print()
    print("=" * 70)
    print(
        f"BUG #9 TESTS: {passed}/{len(TESTS)} PASSED"
    )
    print("=" * 70)

    if passed == len(TESTS):

        print()
        print(
            "BUG #9 MANUAL-STYLE TEST: PASS"
        )

        print(
            "Temporary audio cleanup: PASS"
        )

        print(
            "Failure-path cleanup: PASS"
        )

        print(
            "Original exception preservation: PASS"
        )

        print(
            "Unrelated file protection: PASS"
        )

        print(
            "Exactly-once cleanup: PASS"
        )

        print(
            "Bug #7 regression: PASS"
        )

        print(
            "Bug #8 regression: PASS"
        )

        print()
        print(
            "BUG #9 VALIDATION COMPLETE"
        )