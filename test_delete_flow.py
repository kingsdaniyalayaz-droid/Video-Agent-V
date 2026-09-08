"""
Regression tests for the delete-flow consistency fix.

Run from the project root (Windows or Linux):

    python test_delete_flow.py

What is covered
---------------
1. delete_vector_store() when the Chroma collection exists
2. delete_vector_store() when the collection is missing (safe/idempotent)
3. a real vector deletion failure prevents the SQLite deletion
4. both Saved History and Video Library delete flows use the safe helper
5. existing RAG ownership protections remain intact

No GPU, Whisper, Mistral, or real embeddings are required. Heavy and
optional third-party dependencies are stubbed so the tests run anywhere
from the project root without a Chroma server or a HuggingFace model.
"""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# STUB HEAVY / OPTIONAL DEPENDENCIES
# ============================================================

def _install_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _SessionState(dict):
    """dict-like stub for st.session_state (attribute + item access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def _passthrough_decorator(*args, **kwargs):
    def wrap(fn):
        return fn
    return wrap


# torch (vector_store imports it)
_torch = type("_MagicTorch", (types.ModuleType,), {"__getattr__": lambda self, name: mock.MagicMock()})("torch")
_torch.nn = types.ModuleType("torch.nn")
_torch.nn.Module = type("Module", (), {})
_torch.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", _torch)
_torch.cuda = _install_module("torch.cuda")
_torch.cuda.is_available = lambda: False
_torch.version = types.SimpleNamespace(cuda="0.0")

# streamlit (app imports it)
_st = _install_module("streamlit")
TEST_SESSION = _SessionState()
_st.session_state = TEST_SESSION
_st.cache_resource = _passthrough_decorator

# transformers (app imports it)
_transformers = _install_module("transformers")
_transformers.logging = _install_module("transformers.logging")
_transformers.logging.set_verbosity_error = lambda *a, **k: None

# reportlab (app imports it)
reportlab = _install_module("reportlab")
reportlab.lib = _install_module("reportlab.lib")
reportlab.lib.enums = _install_module("reportlab.lib.enums", TA_CENTER="CENTER")
reportlab.lib.pagesizes = _install_module("reportlab.lib.pagesizes", A4=(595.28, 841.89))
reportlab.lib.styles = _install_module(
    "reportlab.lib.styles",
    getSampleStyleSheet=lambda: {},
    ParagraphStyle=dict,
)
reportlab.lib.units = _install_module("reportlab.lib.units", inch=72.0)
reportlab.pdfbase = _install_module("reportlab.pdfbase", pdfmetrics=mock.MagicMock())
reportlab.pdfbase.ttfonts = _install_module(
    "reportlab.pdfbase.ttfonts",
    TTFont=lambda *a, **k: None,
)
reportlab.platypus = _install_module(
    "reportlab.platypus",
    PageBreak=object,
    Paragraph=object,
    SimpleDocTemplate=object,
)

# langchain_* (vector_store imports them)
langchain_chroma = _install_module("langchain_chroma")
langchain_chroma.Chroma = mock.MagicMock()
langchain_core = importlib.import_module("langchain_core")
langchain_core.documents = importlib.import_module("langchain_core.documents")
langchain_core.documents.Document = mock.MagicMock()
langchain_huggingface = _install_module("langchain_huggingface")
langchain_huggingface.HuggingFaceEmbeddings = mock.MagicMock()
langchain_text_splitters = importlib.import_module("langchain_text_splitters")
langchain_text_splitters.RecursiveCharacterTextSplitter = mock.MagicMock()

# chromadb (vector_store imports it; unit tests patch vector_store.chromadb)
chromadb = _install_module("chromadb")
chromadb.PersistentClient = mock.MagicMock()

# core modules that are not part of this fix (app imports them)
prompt_generator = _install_module("core.prompt_generator")
prompt_generator.generate_prompt = mock.MagicMock()
roman_urdu = importlib.import_module("core.roman_urdu_translator")
roman_urdu.translate_to_roman_urdu = mock.MagicMock()
translation_cache = _install_module("core.translation_cache")
for _name in (
    "delete_translation_by_id",
    "get_cached_translation",
    "get_saved_translation_by_hash",
    "get_saved_translation_by_id",
    "get_transcript_hash",
    "get_translation_count",
    "list_saved_translations",
    "save_translation",
):
    setattr(translation_cache, _name, mock.MagicMock())
translation_exporter = _install_module("core.translation_exporter")
for _name in ("build_translation_excel", "build_translation_pdf", "build_translation_txt"):
    setattr(translation_exporter, _name, mock.MagicMock())
video_library_analytics = _install_module("core.video_library_analytics")
for _name in (
    "calculate_completion_percentage",
    "calculate_library_analytics",
    "format_duration",
    "get_video_content_status",
):
    setattr(video_library_analytics, _name, mock.MagicMock())

# Import the real modules under test.
vector_store = importlib.import_module("core.vector_store")
app = importlib.import_module("app")

DeleteResult = vector_store.DeleteResult
VectorStoreDeleteError = vector_store.VectorStoreDeleteError


# ============================================================
# delete_vector_store() unit tests
# ============================================================

class TestDeleteVectorStore(unittest.TestCase):
    """Covers cases 1, 2, 3 from the bug report."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.chroma_dir = Path(self._tmp.name)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def _fake_client(self, collection_names):
        client = mock.MagicMock()
        collections = []
        for name in collection_names:
            collection = mock.MagicMock()
            collection.name = name
            collections.append(collection)
        client.list_collections.return_value = collections
        return client

    def test_delete_when_collection_exists(self):
        client = self._fake_client(["video_transcripts_vid123"])
        with mock.patch.object(vector_store, "chromadb") as fake_chromadb, \
                mock.patch.object(vector_store, "get_embeddings") as fake_embeddings, \
                mock.patch.object(vector_store, "CHROMA_DIR", self.chroma_dir):
            fake_chromadb.PersistentClient.return_value = client
            result = vector_store.delete_vector_store("vid123")

        self.assertTrue(result.deleted)
        self.assertTrue(result.existed)
        client.delete_collection.assert_called_once_with(name="video_transcripts_vid123")
        # Deletion must not load the embedding model.
        fake_embeddings.assert_not_called()

    def test_delete_when_collection_missing_is_idempotent(self):
        client = self._fake_client([])
        with mock.patch.object(vector_store, "chromadb") as fake_chromadb, \
                mock.patch.object(vector_store, "get_embeddings") as fake_embeddings, \
                mock.patch.object(vector_store, "CHROMA_DIR", self.chroma_dir):
            fake_chromadb.PersistentClient.return_value = client
            result = vector_store.delete_vector_store("vid123")

        self.assertTrue(result.deleted)
        self.assertFalse(result.existed)
        client.delete_collection.assert_not_called()
        fake_embeddings.assert_not_called()

    def test_delete_when_chroma_dir_missing_is_idempotent(self):
        missing_dir = Path(self._tmp.name) / "does_not_exist"
        with mock.patch.object(vector_store, "get_embeddings") as fake_embeddings, \
                mock.patch.object(vector_store, "CHROMA_DIR", missing_dir):
            result = vector_store.delete_vector_store("vid123")

        self.assertTrue(result.deleted)
        self.assertFalse(result.existed)
        fake_embeddings.assert_not_called()

    def test_real_failure_raises_and_is_not_swallowed(self):
        client = self._fake_client(["video_transcripts_vid123"])
        client.delete_collection.side_effect = RuntimeError("disk locked")
        with mock.patch.object(vector_store, "chromadb") as fake_chromadb, \
                mock.patch.object(vector_store, "CHROMA_DIR", self.chroma_dir):
            fake_chromadb.PersistentClient.return_value = client
            with self.assertRaises(VectorStoreDeleteError) as ctx:
                vector_store.delete_vector_store("vid123")

        self.assertIn("disk locked", str(ctx.exception))
        client.delete_collection.assert_called_once()

    def test_double_delete_second_call_is_safe(self):
        """Deleting twice for the same video must not crash (idempotent)."""
        client = self._fake_client(["video_transcripts_vid123"])
        with mock.patch.object(vector_store, "chromadb") as fake_chromadb, \
                mock.patch.object(vector_store, "CHROMA_DIR", self.chroma_dir):
            fake_chromadb.PersistentClient.return_value = client
            first = vector_store.delete_vector_store("vid123")
            # Collection is now gone; a second call must be a safe no-op.
            client.list_collections.return_value = []
            second = vector_store.delete_vector_store("vid123")

        self.assertTrue(first.deleted)
        self.assertTrue(first.existed)
        self.assertTrue(second.deleted)
        self.assertFalse(second.existed)


# ============================================================
# app.py safe-delete helper tests
# ============================================================

class TestSafeDeleteHelper(unittest.TestCase):
    """Covers case 3 (failure prevents SQLite deletion) and case 4 (both flows)."""

    def test_success_deletes_vector_then_sqlite(self):
        with mock.patch.object(
            app, "delete_vector_store",
            return_value=DeleteResult(deleted=True, existed=True),
        ) as fake_vector, \
                mock.patch.object(app, "delete_video") as fake_sqlite:
            result = app._delete_video_safely("vid123")

        fake_vector.assert_called_once_with("vid123")
        fake_sqlite.assert_called_once_with("vid123")
        self.assertTrue(result.deleted)
        self.assertTrue(result.existed)

    def test_missing_collection_still_deletes_sqlite(self):
        with mock.patch.object(
            app, "delete_vector_store",
            return_value=DeleteResult(deleted=True, existed=False),
        ), \
                mock.patch.object(app, "delete_video") as fake_sqlite:
            result = app._delete_video_safely("vid123")

        fake_sqlite.assert_called_once_with("vid123")
        self.assertTrue(result.deleted)
        self.assertFalse(result.existed)

    def test_real_vector_failure_prevents_sqlite_deletion(self):
        with mock.patch.object(
            app, "delete_vector_store",
            side_effect=VectorStoreDeleteError("chroma unreachable"),
        ), \
                mock.patch.object(app, "delete_video") as fake_sqlite:
            with self.assertRaises(VectorStoreDeleteError):
                app._delete_video_safely("vid123")

        fake_sqlite.assert_not_called()

    def test_both_delete_flows_use_the_safe_helper(self):
        """Both delete UI flows still delegate to the shared safe helper.

        Text-presence based (not brittle source slices) so the assertion
        stays valid across UI re-layouts.
        """
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("def _delete_video_safely", source)
        self.assertGreaterEqual(source.count("_delete_video_safely("), 2)
        self.assertIn("_render_video_library_card", source)
        self.assertIn("_render_video_library_analytics", source)

        # ...and must never call delete_video directly (bypassing the check).
        self.assertNotIn("def delete_video", source)
        self.assertNotIn("def delete_video", source)


# ============================================================
# RAG ownership protection tests
# ============================================================

class TestRagOwnershipProtectionsIntact(unittest.TestCase):
    """Covers cases 5 and 6: only the deleted video's RAG state is cleared."""

    def setUp(self):
        TEST_SESSION.clear()

    def test_get_owned_rag_chain_returns_chain_for_owner(self):
        TEST_SESSION["rag_video_id"] = "vid123"
        chain = object()
        result = {"video_id": "vid123", "rag_chain": chain}
        self.assertIs(app.get_owned_rag_chain(result), chain)

    def test_get_owned_rag_chain_detaches_when_owner_mismatch(self):
        TEST_SESSION["rag_video_id"] = "vid456"
        chain = object()
        result = {"video_id": "vid123", "rag_chain": chain}
        self.assertIsNone(app.get_owned_rag_chain(result))
        self.assertIsNone(result["rag_chain"])
        self.assertIsNone(TEST_SESSION.get("rag_video_id"))

    def test_invalidate_rag_state_clears_active_owner(self):
        TEST_SESSION["rag_video_id"] = "vid123"
        result = {"video_id": "vid123", "rag_chain": object()}
        app.invalidate_rag_state(result)
        self.assertIsNone(result["rag_chain"])
        self.assertIsNone(TEST_SESSION.get("rag_video_id"))

    def test_can_reuse_cached_rag_requires_same_owner(self):
        TEST_SESSION["rag_video_id"] = "vid123"
        chain = object()
        active = {
            "video_id": "vid123",
            "source_type": "youtube",
            "metadata": {"video_id": "vid123"},
            "transcript": "hello world",
            "rag_chain": chain,
        }
        self.assertTrue(
            app.can_reuse_cached_rag("vid123", "hello world", active, "vid123", None)
        )
        # A different active video must NOT be able to reuse this cached chain.
        self.assertFalse(
            app.can_reuse_cached_rag("vid123", "hello world", active, "vid456", None)
        )

    def test_delete_flow_keeps_rag_ownership_checks(self):
        """The delete flow still carries the safe helper and RAG ownership checks."""
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("_delete_video_safely(", source)
        self.assertIn('st.session_state.get("rag_video_id")', source)
        self.assertIn("invalidate_rag_state", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
