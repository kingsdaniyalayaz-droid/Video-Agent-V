from pathlib import Path
import importlib.util
import sys
import tempfile
import types

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_PATH = PROJECT_ROOT / "vector_store.py"


def install_import_stubs():
    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = object
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.version = types.SimpleNamespace(cuda=None)

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class Chroma:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_documents(cls, *args, **kwargs):
            return cls()

    langchain_chroma = types.ModuleType("langchain_chroma")
    langchain_chroma.Chroma = Chroma
    langchain_core_documents = types.ModuleType("langchain_core.documents")
    langchain_core_documents.Document = Document
    langchain_huggingface = types.ModuleType("langchain_huggingface")
    langchain_huggingface.HuggingFaceEmbeddings = object
    langchain_text_splitters = types.ModuleType("langchain_text_splitters")
    langchain_text_splitters.RecursiveCharacterTextSplitter = object

    sys.modules.update({
        "chromadb": chromadb,
        "torch": torch,
        "langchain_chroma": langchain_chroma,
        "langchain_core.documents": langchain_core_documents,
        "langchain_huggingface": langchain_huggingface,
        "langchain_text_splitters": langchain_text_splitters,
    })
    return chromadb, Chroma


chromadb, Chroma = install_import_stubs()
spec = importlib.util.spec_from_file_location("vector_store_under_test", MODULE_PATH)
vs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vs
spec.loader.exec_module(vs)
ORIGINAL_DELETE_VECTOR_STORE = vs.delete_vector_store


class FakeCollection:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count


class FakeClient:
    def __init__(self, names=None, counts=None, error=None, list_sequence=None, get_error=None):
        self.names = list(names or [])
        self.counts = counts or {}
        self.error = error
        self.list_sequence = list(list_sequence or [])
        self.get_error = get_error

    def list_collections(self):
        if self.error:
            raise self.error
        if self.list_sequence:
            return self.list_sequence.pop(0)
        return self.names

    def get_collection(self, name):
        if self.get_error:
            raise self.get_error
        return FakeCollection(self.counts[name])

    def delete_collection(self, name):
        if name in self.names:
            self.names.remove(name)
        else:
            raise RuntimeError("collection not found")


def make_chroma_dir():
    return tempfile.TemporaryDirectory()


def test_vector_store_exists_missing_collection():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        chromadb.PersistentClient = lambda path: FakeClient(names=[])
        vs.get_embeddings = lambda: (_ for _ in ()).throw(AssertionError("embeddings loaded"))
        assert vs.vector_store_exists("video-a") is False


def test_vector_store_exists_with_vectors_without_embeddings():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        name = vs.get_collection_name("video-a")
        chromadb.PersistentClient = lambda path: FakeClient(names=[name], counts={name: 3})
        vs.get_embeddings = lambda: (_ for _ in ()).throw(AssertionError("embeddings loaded"))
        assert vs.vector_store_exists("video-a") is True


def test_vector_store_exists_real_error_is_raised():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        chromadb.PersistentClient = lambda path: FakeClient(error=OSError("database locked"))
        try:
            vs.vector_store_exists("video-a")
        except vs.VectorStoreCheckError as exc:
            assert "database locked" in str(exc)
        else:
            raise AssertionError("database failure was masked as False")


def test_vector_store_exists_not_found_race_returns_false():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        name = vs.get_collection_name("video-a")
        chromadb.PersistentClient = lambda path: FakeClient(
            list_sequence=[[name], []],
            get_error=RuntimeError("collection disappeared"),
        )
        assert vs.vector_store_exists("video-a") is False


def test_vector_store_exists_non_not_found_error_is_raised():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        name = vs.get_collection_name("video-a")
        chromadb.PersistentClient = lambda path: FakeClient(
            list_sequence=[[name], [name]],
            get_error=PermissionError("permission denied"),
        )
        try:
            vs.vector_store_exists("video-a")
        except vs.VectorStoreCheckError as exc:
            assert "permission denied" in str(exc)
        else:
            raise AssertionError("non-race infrastructure error was masked")


def patch_build_dependencies():
    vs.create_documents = lambda **kwargs: [object()]
    vs.get_embeddings = lambda: object()
    vs.Chroma.from_documents = classmethod(lambda cls, **kwargs: "rebuilt-store")


def test_force_rebuild_existing_vectors_deletes_without_existence_gate():
    patch_build_dependencies()
    calls = []
    vs.delete_vector_store = lambda video_id=None: calls.append(video_id)
    vs.vector_store_exists = lambda video_id: (_ for _ in ()).throw(
        AssertionError("force rebuild checked vector count")
    )
    assert vs.build_vector_store("transcript", video_id="video-a", force_rebuild=True) == "rebuilt-store"
    assert calls == ["video-a"]


def test_force_rebuild_empty_collection_still_deletes():
    patch_build_dependencies()
    calls = []
    vs.delete_vector_store = lambda video_id=None: calls.append(video_id)
    assert vs.build_vector_store("transcript", video_id="video-empty", force_rebuild=True) == "rebuilt-store"
    assert calls == ["video-empty"]


def test_force_rebuild_missing_collection_continues():
    patch_build_dependencies()
    calls = []
    vs.delete_vector_store = lambda video_id=None: calls.append(video_id)
    assert vs.build_vector_store("transcript", video_id="video-missing", force_rebuild=True) == "rebuilt-store"
    assert calls == ["video-missing"]


def test_reset_missing_legacy_collection_continues():
    patch_build_dependencies()
    calls = []
    vs.delete_vector_store = lambda video_id=None: calls.append(video_id)
    assert vs.build_vector_store("transcript", reset=True) == "rebuilt-store"
    assert calls == [None]


def test_delete_client_initialization_failure_is_chained():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        original_error = OSError("cannot open Chroma database")
        chromadb.PersistentClient = lambda path: (_ for _ in ()).throw(original_error)
        try:
            ORIGINAL_DELETE_VECTOR_STORE("video-a")
        except vs.VectorStoreDeleteError as exc:
            assert not isinstance(exc, UnboundLocalError)
            assert "cannot open Chroma database" in str(exc)
            assert exc.__cause__ is original_error
        else:
            raise AssertionError("client initialization failure was not raised")


def test_delete_race_is_idempotent():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        name = vs.get_collection_name("video-a")
        client = FakeClient(list_sequence=[[name], []])
        chromadb.PersistentClient = lambda path: client
        original_delete = client.delete_collection
        client.delete_collection = lambda name: (_ for _ in ()).throw(RuntimeError("already deleted"))
        result = ORIGINAL_DELETE_VECTOR_STORE("video-a")
        assert result.deleted is True
        assert result.existed is True
        assert original_delete is not None


def test_delete_real_failure_propagates():
    with make_chroma_dir() as directory:
        vs.CHROMA_DIR = Path(directory)
        name = vs.get_collection_name("video-a")
        client = FakeClient(list_sequence=[[name], [name]])
        chromadb.PersistentClient = lambda path: client
        client.delete_collection = lambda name: (_ for _ in ()).throw(PermissionError("permission denied"))
        try:
            ORIGINAL_DELETE_VECTOR_STORE("video-a")
        except vs.VectorStoreDeleteError as exc:
            assert "permission denied" in str(exc)
        else:
            raise AssertionError("real delete failure was masked")


def test_reset_delete_failure_stops_rebuild():
    rebuild_called = []
    vs.create_documents = lambda **kwargs: rebuild_called.append(True)

    def fail_delete(video_id=None):
        raise vs.VectorStoreDeleteError("permission denied")

    vs.delete_vector_store = fail_delete
    try:
        vs.build_vector_store("transcript", video_id="video-a", reset=True)
    except vs.VectorStoreDeleteError as exc:
        assert "permission denied" in str(exc)
    else:
        raise AssertionError("delete failure was swallowed")
    assert rebuild_called == []


if __name__ == "__main__":
    tests = [
        test_vector_store_exists_missing_collection,
        test_vector_store_exists_with_vectors_without_embeddings,
        test_vector_store_exists_real_error_is_raised,
        test_vector_store_exists_not_found_race_returns_false,
        test_vector_store_exists_non_not_found_error_is_raised,
        test_force_rebuild_existing_vectors_deletes_without_existence_gate,
        test_force_rebuild_empty_collection_still_deletes,
        test_force_rebuild_missing_collection_continues,
        test_reset_missing_legacy_collection_continues,
        test_delete_client_initialization_failure_is_chained,
        test_delete_race_is_idempotent,
        test_delete_real_failure_propagates,
        test_reset_delete_failure_stops_rebuild,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All vector-store regression tests passed.")
