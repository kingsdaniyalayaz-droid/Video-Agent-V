import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = ROOT / "main.py"
DB_PATH = ROOT / "database.py"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_file = Path(self.temp_dir.name) / "video_agent.db"
        os.environ["VIDEO_DB_PATH"] = str(self.db_file)
        self._clear_modules()

    def tearDown(self):
        self._clear_modules()
        self.temp_dir.cleanup()
        os.environ.pop("VIDEO_DB_PATH", None)

    def _clear_modules(self):
        for module_name in [
            "main",
            "database",
            "core",
            "core.database",
            "core.transcriber",
            "core.translator",
            "core.summarize",
            "core.rag_engine",
            "utils",
            "utils.audio_processor",
            "extractor",
            "dotenv",
        ]:
            sys.modules.pop(module_name, None)

    def _load_database(self):
        return load_module("database", DB_PATH)

    def _prepare_main(self, *, action_text, decision_text, question_text):
        database = self._load_database()

        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = []
        utils_pkg = types.ModuleType("utils")
        utils_pkg.__path__ = []
        sys.modules["core"] = core_pkg
        sys.modules["utils"] = utils_pkg
        sys.modules["core.database"] = database
        setattr(core_pkg, "database", database)

        audio_processor = types.ModuleType("utils.audio_processor")
        audio_processor.process_input = lambda source, chunk_minutes: ["chunk-1.wav"]
        sys.modules["utils.audio_processor"] = audio_processor
        setattr(utils_pkg, "audio_processor", audio_processor)

        transcriber = types.ModuleType("core.transcriber")
        transcriber.transcribe_all = lambda chunks, translate, language, progress_callback=None: "Discuss roadmap. Send recap. Approve budget. What is next?"
        sys.modules["core.transcriber"] = transcriber
        setattr(core_pkg, "transcriber", transcriber)

        translator = types.ModuleType("core.translator")
        translator.translate_text = lambda text, source_language, target_language: text
        sys.modules["core.translator"] = translator
        setattr(core_pkg, "translator", translator)

        summarize = types.ModuleType("core.summarize")
        summarize.summarize = lambda transcript, source_type: "Meeting summary"
        summarize.generate_title = lambda transcript, source_type: "Meeting title"
        sys.modules["core.summarize"] = summarize
        setattr(core_pkg, "summarize", summarize)

        rag_engine = types.ModuleType("core.rag_engine")
        rag_engine.build_rag_chain = lambda transcript, source_type, top_k, video_id=None: {"rag": "ready", "video_id": video_id}
        rag_engine.ask_question = lambda rag_chain, question: "answer"
        sys.modules["core.rag_engine"] = rag_engine
        setattr(core_pkg, "rag_engine", rag_engine)

        extractor = types.ModuleType("extractor")
        extractor.extract_action_items = lambda transcript: action_text
        extractor.extract_decisions = lambda transcript: decision_text
        extractor.extract_questions = lambda transcript: question_text
        sys.modules["extractor"] = extractor

        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv

        main = load_module("main", MAIN_PATH)
        return main, database

    def test_pipeline_result_and_result_to_dict_include_analysis_fields(self):
        main, _ = self._prepare_main(
            action_text="## Action Items\n1. Send recap email",
            decision_text="## Decisions\n1. Approve budget",
            question_text="## Questions\n1. What is next?",
        )

        result = main.PipelineResult(
            source="src",
            source_type="youtube",
            language="en",
            summary="summary",
            actions=["Send recap email"],
            decisions=["Approve budget"],
            questions=["What is next?"],
        )

        result_dict = main.result_to_dict(result)

        self.assertEqual(result.actions, ["Send recap email"])
        self.assertEqual(result.decisions, ["Approve budget"])
        self.assertEqual(result.questions, ["What is next?"])
        self.assertEqual(result_dict["summary"], "summary")
        self.assertEqual(result_dict["actions"], ["Send recap email"])
        self.assertEqual(result_dict["decisions"], ["Approve budget"])
        self.assertEqual(result_dict["questions"], ["What is next?"])

    def test_new_video_persists_analysis_fields(self):
        main, database = self._prepare_main(
            action_text="## Action Items\n1. Send recap email\n2. Schedule follow-up",
            decision_text="## Decisions\n1. Approve Q4 budget",
            question_text="## Questions\n1. What should ship first?",
        )

        result = main.run_pipeline(
            source="https://www.youtube.com/watch?v=abc123xyz01",
            source_type="youtube",
            language="en",
            video_id="abc123xyz01",
        )

        self.assertEqual(result.actions, ["Send recap email", "Schedule follow-up"])
        self.assertEqual(result.decisions, ["Approve Q4 budget"])
        self.assertEqual(result.questions, ["What should ship first?"])

        record = database.get_video("abc123xyz01")
        self.assertIsNotNone(record)
        self.assertEqual(record["actions"], ["Send recap email", "Schedule follow-up"])
        self.assertEqual(record["decisions"], ["Approve Q4 budget"])
        self.assertEqual(record["questions"], ["What should ship first?"])

    def test_cached_video_restores_analysis_fields_without_reprocessing(self):
        main, database = self._prepare_main(
            action_text="## Action Items\n1. Send recap email",
            decision_text="## Decisions\n1. Approve budget",
            question_text="## Questions\n1. What is next?",
        )

        first = main.run_pipeline(
            source="https://www.youtube.com/watch?v=cached000001",
            source_type="youtube",
            language="en",
            video_id="cached000001",
        )
        self.assertEqual(first.actions, ["Send recap email"])

        main.process_input = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("process_input should not run for cached videos"))
        cached = main.run_pipeline(
            source="https://www.youtube.com/watch?v=cached000001",
            source_type="youtube",
            language="en",
            video_id="cached000001",
        )

        self.assertTrue(cached.metadata.get("cached"))
        self.assertEqual(cached.actions, ["Send recap email"])
        self.assertEqual(cached.decisions, ["Approve budget"])
        self.assertEqual(cached.questions, ["What is next?"])
        self.assertEqual(database.get_video("cached000001")["actions"], ["Send recap email"])

    def test_old_video_records_without_analysis_columns_load_safely(self):
        legacy_db = Path(self.temp_dir.name) / "legacy.db"
        connection = sqlite3.connect(legacy_db)
        connection.executescript(
            """
            CREATE TABLE videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                language TEXT,
                transcript TEXT,
                translated_transcript TEXT,
                summary TEXT,
                duration REAL,
                chunk_count INTEGER,
                status TEXT NOT NULL DEFAULT 'processing',
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO videos (
                video_id, source, source_type, title, language, transcript,
                translated_transcript, summary, duration, chunk_count, status,
                error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy001",
                "legacy-source",
                "youtube",
                "Legacy Title",
                "en",
                "Legacy transcript",
                "Legacy transcript",
                "Legacy summary",
                12.0,
                1,
                "completed",
                None,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        os.environ["VIDEO_DB_PATH"] = str(legacy_db)
        self._clear_modules()
        database = self._load_database()

        record = database.get_video("legacy001")
        self.assertEqual(record["actions"], [])
        self.assertEqual(record["decisions"], [])
        self.assertEqual(record["questions"], [])

    def test_empty_analysis_defaults_to_empty_lists(self):
        main, database = self._prepare_main(
            action_text="No action items found.",
            decision_text="No decisions found.",
            question_text="No questions found.",
        )

        result = main.run_pipeline(
            source="/tmp/local-meeting.mp4",
            source_type="meeting",
            language="en",
            video_id="meeting_deadbeef0001",
        )

        self.assertEqual(result.actions, [])
        self.assertEqual(result.decisions, [])
        self.assertEqual(result.questions, [])

        record = database.get_video("meeting_deadbeef0001")
        self.assertEqual(record["actions"], [])
        self.assertEqual(record["decisions"], [])
        self.assertEqual(record["questions"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
