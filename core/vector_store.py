"""
core/vector_store.py

ChromaDB + HuggingFace Embeddings based vector store.

Purpose
-------
Store transcript chunks as vectors and retrieve the most
relevant chunks for RAG.

Pipeline:

    Transcript
        ↓
    Text Chunking
        ↓
    LangChain Documents
        ↓
    HuggingFace Embeddings
        ↓
    ChromaDB
        ↓
    Retriever
        ↓
    RAG
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
import torch

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = Path(
    os.getenv(
        "CHROMA_DIR",
        "vector_db",
    )
)


COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "video_transcripts",
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


CHUNK_SIZE = int(
    os.getenv(
        "RAG_CHUNK_SIZE",
        "500",
    )
)


CHUNK_OVERLAP = int(
    os.getenv(
        "RAG_CHUNK_OVERLAP",
        "50",
    )
)


DEFAULT_TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "4",
    )
)


# ============================================================
# EMBEDDING MODEL CACHE
# ============================================================

_embeddings: Optional[HuggingFaceEmbeddings] = None


# ============================================================
# DEVICE DETECTION
# ============================================================

def get_device() -> str:
    """
    Detect the best available device.

    CUDA available:
        NVIDIA GPU

    Otherwise:
        CPU
    """

    if torch.cuda.is_available():

        device = "cuda"

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        print(
            f"🎮 Embedding GPU: {gpu_name}"
        )

        print(
            f"🔥 CUDA: {torch.version.cuda}"
        )

        return device

    print(
        "⚠️ CUDA unavailable. "
        "Embeddings will use CPU."
    )

    return "cpu"


# ============================================================
# GET EMBEDDINGS
# ============================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and cache the HuggingFace embedding model.

    Model is loaded only once during the application
    lifetime.
    """

    global _embeddings

    if _embeddings is not None:

        return _embeddings

    device = get_device()

    print("\n" + "=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    print(
        f"Model : {EMBEDDING_MODEL}"
    )

    print(
        f"Device: {device}"
    )

    try:

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,

            model_kwargs={
                "device": device,
            },

            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load HuggingFace embedding model. "
            f"Reason: {exc}"
        ) from exc

    print(
        "✅ Embedding model loaded successfully."
    )

    print("=" * 60)

    return _embeddings


# ============================================================
# INPUT VALIDATION HELPERS
# ============================================================

def _validate_k(k: Any) -> int:
    """Validate a retriever/search k value strictly.

    k must be a real integer (bool is rejected because it is an int
    subclass), must not be None, and must be greater than zero.  Raises
    ``ValueError`` otherwise and returns the validated integer when valid.
    """

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(
            "k must be an integer. "
            f"Got {type(k).__name__}."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    return k


def _validate_search_query(query: Any) -> str:
    """Validate and normalize a similarity-search query.

    query must be a string; None / non-string values and empty or
    whitespace-only strings raise a clear ``ValueError``.  Returns the
    stripped query.
    """

    if not isinstance(query, str):
        raise ValueError(
            "query must be a string. "
            f"Got {type(query).__name__}."
        )

    normalized = query.strip()

    if not normalized:
        raise ValueError(
            "Search query cannot be empty."
        )

    return normalized


def _normalize_video_id(video_id: Any) -> str | None:
    """Validate and normalize a video_id for collection logic.

    ``None`` is valid and selects legacy/global collection mode.  A string
    is stripped; empty or whitespace-only strings raise.  Any non-string
    value (including falsy values like ``0`` / ``False`` and truthy values
    like ``123`` / ``[]`` / ``{}``) raises a clear ``ValueError`` instead of
    silently entering legacy mode.
    """

    if video_id is None:
        return None

    if not isinstance(video_id, str):
        raise ValueError(
            "video_id must be a string or None. "
            f"Got {type(video_id).__name__}."
        )

    normalized = video_id.strip()

    if not normalized:
        raise ValueError(
            "video_id cannot be empty."
        )

    return normalized


# ============================================================
# VALIDATE TRANSCRIPT
# ============================================================

def validate_transcript(
    transcript: str,
) -> None:
    """
    Validate transcript before vectorization.
    """

    if not isinstance(transcript, str):
        raise ValueError(
            "transcript must be a string. "
            f"Got {type(transcript).__name__}."
        )

    if not transcript:

        raise ValueError(
            "Transcript cannot be empty."
        )

    if not transcript.strip():

        raise ValueError(
            "Transcript contains no readable text."
        )


# ============================================================
# GENERATE DOCUMENT ID
# ============================================================

def generate_document_id(
    transcript: str,
    source: str,
) -> str:
    """
    Generate a deterministic ID for the transcript.

    Same transcript + same source
    → same ID.
    """

    raw = (
        f"{source}|{transcript}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[:16]


# ============================================================
# SPLIT TRANSCRIPT
# ============================================================

def split_transcript(
    transcript: str,
) -> list[str]:
    """
    Split transcript into smaller semantic chunks.
    """

    validate_transcript(
        transcript
    )

    if CHUNK_OVERLAP >= CHUNK_SIZE:

        raise ValueError(
            "CHUNK_OVERLAP must be smaller "
            "than CHUNK_SIZE."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_text(
        transcript
    )

    if not chunks:

        raise RuntimeError(
            "Transcript chunking produced no chunks."
        )

    return chunks


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents(
    transcript: str,
    source: str = "meeting",
    video_id: str | None = None,
) -> list[Document]:
    """
    Convert transcript chunks into LangChain Documents.

    Metadata is preserved for future filtering and
    source tracking.
    """

    validate_transcript(
        transcript
    )

    if source is None or (isinstance(source, str) and not source.strip()):
        source = "meeting"
    elif not isinstance(source, str):
        raise ValueError(
            "source must be a string. "
            f"Got {type(source).__name__}."
        )
    else:
        source = source.strip()

    # Normalize/validate video_id through the module's centralized helper
    # BEFORE any document metadata is created.  Only None is legacy/no-video
    # mode; invalid falsy values (False / 0 / "") raise instead of silently
    # skipping the metadata.
    video_id = _normalize_video_id(video_id)

    chunks = split_transcript(
        transcript
    )

    print(
        f"📄 Transcript split into "
        f"{len(chunks)} chunks."
    )

    document_id = generate_document_id(
        transcript,
        source,
    )

    total_chunks = len(
        chunks
    )

    documents: list[Document] = []

    for index, chunk in enumerate(
        chunks
    ):

        document = Document(
            page_content=chunk,

            metadata={
                "document_id": document_id,
                "source": source,
                **({"video_id": video_id} if video_id else {}),
                "chunk_index": index,
                "total_chunks": total_chunks,
            },
        )

        documents.append(
            document
        )

    return documents


# ============================================================
# ENSURE CHROMA DIRECTORY
# ============================================================

def ensure_chroma_directory() -> None:
    """
    Create Chroma persistence directory if required.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# VIDEO-SPECIFIC COLLECTIONS
# ============================================================


class VectorStoreCheckError(RuntimeError):
    """Raised when Chroma collection existence cannot be determined."""


_COLLECTION_NAME_MAX_LENGTH = 63
_COLLECTION_NAME_HASH_LENGTH = 12


def _get_legacy_collection_name(video_id: str | None = None) -> str:
    """Reproduce the PREVIOUS collection-naming behavior exactly.

    Used ONLY for backward-compatible lookup of collections persisted before
    the collision-resistant naming scheme existed.  Never used to create new
    collections.
    """
    video_id = _normalize_video_id(video_id)

    if video_id is None:
        return COLLECTION_NAME

    safe_video_id = "".join(
        char for char in video_id
        if char.isalnum() or char in {"-", "_"}
    )

    if not safe_video_id:
        raise ValueError(f"Invalid video_id: {video_id}")

    return f"{COLLECTION_NAME}_{safe_video_id}"[:63]


def get_collection_name(video_id: str | None = None) -> str:
    """Return a collision-resistant, deterministic Chroma collection name.

    Format::

        COLLECTION_NAME_<readable_prefix>_<short_hash>

    The readable prefix is the sanitized ``video_id`` (alphanumeric / ``-`` /
    ``_`` only).  The short hash is a truncated SHA-256 digest of the ORIGINAL
    UNSANITIZED ``video_id``, so distinct IDs that sanitize to the same prefix
    (``video@123`` vs ``video#123``) or share a long prefix still produce
    distinct collection names.
    """
    video_id = _normalize_video_id(video_id)

    if video_id is None:
        return COLLECTION_NAME

    safe_prefix = "".join(
        char for char in video_id
        if char.isalnum() or char in {"-", "_"}
    )

    if not safe_prefix:
        raise ValueError(f"Invalid video_id: {video_id}")

    digest = hashlib.sha256(
        video_id.encode("utf-8")
    ).hexdigest()
    short_hash = digest[:_COLLECTION_NAME_HASH_LENGTH]

    suffix = f"_{short_hash}"
    prefix_budget = (
        _COLLECTION_NAME_MAX_LENGTH
        - len(COLLECTION_NAME)
        - len(suffix)
        - 1
    )
    readable_prefix = safe_prefix[:max(prefix_budget, 1)]

    return f"{COLLECTION_NAME}_{readable_prefix}{suffix}"


def _list_collection_names(client) -> set[str]:
    """Return the set of existing Chroma collection names."""
    existing = client.list_collections()
    return {
        getattr(collection, "name", collection)
        for collection in existing
    }


def _find_existing_collection_name(
    video_id: str | None,
    names: set[str],
) -> Optional[str]:
    """Return the existing collection name for ``video_id``.

    The NEW (collision-resistant) name is preferred; when it does not exist
    the LEGACY name is returned so previously persisted collections stay
    accessible.  Returns None when neither exists.
    """
    video_id = _normalize_video_id(video_id)

    if video_id is None:
        return COLLECTION_NAME if COLLECTION_NAME in names else None
    new_name = get_collection_name(video_id)
    if new_name in names:
        return new_name
    legacy_name = _get_legacy_collection_name(video_id)
    if legacy_name in names:
        return legacy_name
    return None


def _resolve_existing_collection_name(
    video_id: str | None = None,
) -> Optional[str]:
    """Return the name of an existing collection for ``video_id``.

    NEW collection first, LEGACY fallback.  Returns None when neither exists.
    Real Chroma failures raise :class:`VectorStoreCheckError` instead of
    being treated as "not found".
    """
    video_id = _normalize_video_id(video_id)

    if not CHROMA_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        names = _list_collection_names(client)
    except Exception as exc:
        raise VectorStoreCheckError(
            f"Failed to list Chroma collections while resolving "
            f"video_id '{video_id or 'legacy'}'. Reason: {exc}"
        ) from exc
    return _find_existing_collection_name(video_id, names)


def vector_store_exists(video_id: str | None = None) -> bool:
    """Return whether a collection contains vectors without loading embeddings.

    The NEW collection is checked first, then the LEGACY collection.  A
    missing Chroma directory or collection is an expected negative result.
    Errors while opening or querying an existing database are raised because
    treating infrastructure failure as "not found" could trigger an unsafe
    rebuild.
    """
    video_id = _normalize_video_id(video_id)

    if not CHROMA_DIR.exists():
        return False

    found = _resolve_existing_collection_name(video_id)
    if found is None:
        return False

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=found)
        return collection.count() > 0
    except Exception as exc:
        # The collection may have been deleted after the existence check.
        # Re-check through the same lightweight flow so only that expected
        # TOCTOU disappearance is treated as a negative result.
        try:
            remaining = _resolve_existing_collection_name(video_id)
        except Exception as verify_exc:
            raise VectorStoreCheckError(
                f"Failed to verify Chroma collection '{found}' after a check "
                f"error. Reason: {exc}; verification failed: {verify_exc}"
            ) from exc

        if remaining is None:
            return False

        raise VectorStoreCheckError(
            f"Failed to check Chroma collection '{found}'. "
            f"Reason: {exc}"
        ) from exc


# ============================================================
# DELETE RESULT
# ============================================================

@dataclass
class DeleteResult:
    """
    Outcome of a vector collection deletion attempt.

    Attributes
    ----------
    deleted:
        True when the collection is gone after this call, either
        because it was actually deleted or because it never existed
        (safe/idempotent success).
    existed:
        True when the collection existed before this call.
    error:
        Populated only when a real failure occurred. On failure the
        function raises instead of returning, so this is informational.
    """

    deleted: bool
    existed: bool
    error: Optional[str] = None


class VectorStoreDeleteError(RuntimeError):
    """Raised when a real Chroma deletion failure occurs."""


# ============================================================
# DELETE EXISTING COLLECTION
# ============================================================

def _collection_names_for_delete(
    video_id: str | None = None,
) -> list[str]:
    """Return the collection names that MAY correspond to ``video_id``.

    The first is the NEW collision-resistant name (uniquely tied to the
    original unsanitized ``video_id``).  The second is the LEGACY name, which
    is NOT collision-safe: distinct video_ids can sanitize onto one legacy
    name.  Callers must verify legacy ownership from stored metadata before
    deleting a legacy collection.
    """
    video_id = _normalize_video_id(video_id)

    if video_id is None:
        return [COLLECTION_NAME]
    return [
        get_collection_name(video_id),
        _get_legacy_collection_name(video_id),
    ]


def _legacy_collection_owned_by(
    client,
    collection_name: str,
    video_id: str,
) -> bool:
    """Return True only when the legacy collection's stored metadata confirms
    ownership by the EXACT original ``video_id``.

    A legacy collection name is NOT proof of ownership: sanitization maps
    distinct original video_ids (e.g. ``video@123`` and ``video#123``) onto
    the same legacy name.  Ownership is therefore verified from the stored
    document metadata, comparing the stored original ``video_id`` exactly
    against the requested one.  Real inspection failures raise
    :class:`VectorStoreDeleteError` -- they are never treated as "not owned".
    """
    try:
        collection = client.get_collection(name=collection_name)
        data = collection.get(include=["metadatas"])
    except Exception as exc:
        raise VectorStoreDeleteError(
            f"Failed to inspect legacy collection '{collection_name}' "
            f"for ownership verification. Reason: {exc}"
        ) from exc

    metadatas = data.get("metadatas") if isinstance(data, dict) else None
    if not metadatas:
        return False
    for meta in metadatas:
        if meta and str(meta.get("video_id", "")) == str(video_id):
            return True
    return False


def delete_vector_store(
    video_id: str | None = None,
) -> DeleteResult:
    """
    Delete the Chroma collection(s) for one video.

    The NEW collision-resistant collection (hash of the original unsanitized
    video_id) is uniquely tied to the requested video and is deleted when
    present.  A LEGACY collection is deleted ONLY after its stored document
    metadata confirms ownership by the exact original video_id.  When a
    legacy collection exists but ownership cannot be verified, it is NOT
    deleted and a clear safety error is raised -- the legacy name alone is
    not collision-safe and could belong to another video.

    Returns
    -------
    DeleteResult
        ``deleted=True`` when the collections are gone after the call
        (either actually deleted, or already absent which is treated as
        a safe/idempotent success). ``existed`` reports whether a
        collection existed before this call.

    Raises
    ------
    VectorStoreDeleteError
        When a real Chroma/database failure prevents the deletion, or when a
        legacy collection's ownership cannot be verified. Errors are NEVER
        silently swallowed here. Callers must not proceed with dependent
        cleanup (e.g. deleting the SQLite record) when this is raised.

    Notes
    -----
    Deletion uses the raw Chroma PersistentClient and does NOT load the
    embedding model, so removing a collection requires no GPU/CPU model
    initialization.
    """

    video_id = _normalize_video_id(video_id)

    if not CHROMA_DIR.exists():

        print(
            "ℹ️ Vector database does not exist. Nothing to delete."
        )

        return DeleteResult(
            deleted=True,
            existed=False,
        )

    candidates: list[str] = []
    client = None

    try:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        names = _list_collection_names(client)

        # ----------------------------------------------------
        # Resolve which collections may be deleted for this video.
        # ----------------------------------------------------
        if video_id is None:
            # Global / legacy store reset: unambiguous by definition.
            candidates = [COLLECTION_NAME]
        else:
            new_name, legacy_name = _collection_names_for_delete(video_id)
            candidates = []
            if new_name in names:
                candidates.append(new_name)

            if legacy_name in names:
                # The legacy name alone is NOT proof of ownership.  Verify
                # the stored original video_id before allowing deletion.
                if _legacy_collection_owned_by(
                    client, legacy_name, video_id
                ):
                    candidates.append(legacy_name)
                else:
                    # Ambiguous legacy ownership.  NEW and LEGACY are handled
                    # independently: when a safely identifiable NEW hashed
                    # collection exists, it is deleted while the ambiguous
                    # legacy collection is PRESERVED untouched.  Only when NO
                    # NEW collection exists does the ambiguity block the
                    # operation with a clear safety error.
                    if new_name in names:
                        print(
                            f"🔒 Preserving ambiguous legacy collection "
                            f"'{legacy_name}' for video_id '{video_id}': "
                            f"ownership could not be verified from stored "
                            f"metadata. The legacy collection name is not "
                            f"collision-safe, so it is left untouched."
                        )
                    else:
                        raise VectorStoreDeleteError(
                            f"Refusing to delete legacy collection "
                            f"'{legacy_name}' for video_id '{video_id}': "
                            f"ownership could not be verified from stored "
                            f"metadata. The legacy collection name is not "
                            f"collision-safe; deleting it could destroy "
                            f"another video's data."
                        )

        existing_targets = [
            name for name in candidates
            if name in names
        ]

        if not existing_targets:

            print(
                f"ℹ️ Collection not found "
                f"(safe/idempotent): {', '.join(candidates)}"
            )

            return DeleteResult(
                deleted=True,
                existed=False,
            )

        for name in existing_targets:
            client.delete_collection(
                name=name
            )
            print(
                f"🗑️ Collection deleted: "
                f"{name}"
            )

        return DeleteResult(
            deleted=True,
            existed=True,
        )

    except VectorStoreDeleteError:

        raise

    except Exception as exc:
        if client is None:
            raise VectorStoreDeleteError(
                f"Failed to initialize Chroma client while deleting "
                f"collections '{', '.join(candidates)}'. Reason: {exc}"
            ) from exc

        # A concurrent delete is an expected idempotent race. Verify that the
        # targets are now absent before treating the original delete failure
        # as safe; permission, lock, corruption, and other failures remain
        # errors.
        try:
            remaining = _list_collection_names(client)
        except Exception as verify_exc:
            raise VectorStoreDeleteError(
                f"Failed to delete Chroma collections "
                f"'{', '.join(candidates)}'. "
                f"Reason: {exc}; deletion verification failed: {verify_exc}"
            ) from exc

        remaining_targets = [
            name for name in candidates
            if name in remaining
        ]
        if not remaining_targets:
            print(
                f"ℹ️ Collections already absent after delete race "
                f"(safe/idempotent): {', '.join(candidates)}"
            )
            return DeleteResult(
                deleted=True,
                existed=True,
            )

        raise VectorStoreDeleteError(
            f"Failed to delete Chroma collections "
            f"'{', '.join(candidates)}'. Reason: {exc}"
        ) from exc


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(
    transcript: str,
    source: str = "meeting",
    reset: bool = False,
    video_id: str | None = None,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Build or reuse a persistent, video-specific Chroma vector store.

    New video:
        transcript -> chunks -> embeddings -> Chroma

    Existing video:
        existing Chroma -> reuse
        embeddings are NOT regenerated

    force_rebuild=True:
        only the requested video's collection is rebuilt.
    """

    if not isinstance(transcript, str):
        raise ValueError(
            "transcript must be a string. "
            f"Got {type(transcript).__name__}."
        )

    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcript cannot be empty."
        )

    video_id = _normalize_video_id(video_id)

    # --------------------------------------------------------
    # IMPORTANT:
    # Use video-specific collection
    # --------------------------------------------------------

    collection_name = get_collection_name(video_id)

    print("\n" + "=" * 70)
    print("VIDEO VECTOR STORE")
    print("=" * 70)
    print(f"Video ID   : {video_id or 'legacy'}")
    print(f"Source     : {source}")
    print(f"Collection : {collection_name}")
    print(f"Embedding  : {EMBEDDING_MODEL}")

    # --------------------------------------------------------
    # Existing vector store
    # --------------------------------------------------------

    if (
        video_id
        and not reset
        and not force_rebuild
        and vector_store_exists(video_id)
    ):
        print("\n♻️ EXISTING VECTOR STORE FOUND")
        print(f"Video ID   : {video_id}")
        print(f"Collection : {collection_name}")
        print("Embedding  : SKIPPED")
        print("Action     : LOAD EXISTING VECTORS")
        print("=" * 70)

        return load_vector_store(video_id)

    # --------------------------------------------------------
    # Rebuild requested
    # --------------------------------------------------------

    if video_id and (reset or force_rebuild):
        print("\n⚠️ REBUILDING VIDEO VECTOR STORE")
        print(f"Video ID   : {video_id}")
        print(f"Collection : {collection_name}")

        # Deletion is intentionally unconditional: an empty collection may
        # still contain stale metadata or failed-build state. The delete API
        # is idempotent for a missing collection and raises real failures.
        delete_vector_store(
            video_id=video_id
        )

    # --------------------------------------------------------
    # Legacy / no video_id
    # --------------------------------------------------------

    if not video_id and reset:
        # Missing collections are handled idempotently by delete_vector_store;
        # real deletion failures must stop the rebuild and reach the caller.
        delete_vector_store(
            video_id=None
        )

    # --------------------------------------------------------
    # Create documents
    # --------------------------------------------------------

    documents = create_documents(
        transcript=transcript,
        source=source,
        video_id=video_id,
    )

    print(
        f"Documents  : {len(documents)}"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = get_embeddings()

    print("\nCreating embeddings...")

    # --------------------------------------------------------
    # Persistent Chroma
    # --------------------------------------------------------

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )

    print("\n✅ Vector store created successfully.")
    print(f"📁 Database   : {CHROMA_DIR}")
    print(f"📦 Collection : {collection_name}")
    print(f"📄 Documents  : {len(documents)}")
    print("=" * 70)

    return vector_store

# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store(
    video_id: str | None = None,
) -> Chroma:
    """
    Load an existing Chroma vector store.

    The requested collection (NEW name first, LEGACY fallback) is explicitly
    verified to exist BEFORE the embedding model is loaded, so a missing
    collection raises ``FileNotFoundError`` without initializing embeddings.
    """

    video_id = _normalize_video_id(video_id)

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            f"Vector database does not exist: "
            f"{CHROMA_DIR}"
        )

    # Explicitly verify the requested collection exists (NEW first, then
    # LEGACY) BEFORE any embedding initialization.  A missing collection is
    # a hard error -- never an empty/misleading vector-store wrapper.
    collection_name = _resolve_existing_collection_name(video_id)
    if collection_name is None:
        requested = get_collection_name(video_id)
        raise FileNotFoundError(
            f"Vector store collection does not exist for "
            f"video_id: {video_id or 'legacy'} "
            f"(checked '{requested}')."
        )

    # Only after the collection is confirmed to exist do we initialize the
    # (cached) embedding model.
    embeddings = get_embeddings()

    try:

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(
                CHROMA_DIR
            ),
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to load Chroma vector store. "
            f"Reason: {exc}"
        ) from exc

    print(
        f"✅ Vector store loaded: "
        f"{collection_name}"
    )

    return vector_store


# ============================================================
# GET RETRIEVER
# ============================================================

def get_retriever(
    vector_store: Chroma,
    k: int = DEFAULT_TOP_K,
):
    """
    Convert Chroma vector store into a LangChain retriever.
    """

    k = _validate_k(k)

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def search_similar_chunks(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> list[Document]:
    """
    Retrieve the most relevant transcript chunks.
    """

    query = _validate_search_query(query)

    k = _validate_k(k)

    try:

        results = vector_store.similarity_search(
            query,
            k=k,
        )

    except Exception as exc:

        raise RuntimeError(
            "Similarity search failed. "
            f"Reason: {exc}"
        ) from exc

    return results


# ============================================================
# RETRIEVAL WITH SCORES
# ============================================================

def search_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
):
    """
    Similarity search with relevance scores.

    Useful for debugging RAG retrieval quality.
    """

    query = _validate_search_query(query)

    k = _validate_k(k)

    return vector_store.similarity_search_with_score(
        query,
        k=k,
    )


# ============================================================
# COLLECTION INFO
# ============================================================

def get_collection_count(
    vector_store: Chroma,
) -> int:
    """
    Return number of stored vector documents.
    """

    try:

        collection = (
            vector_store._collection
        )

        return collection.count()

    except Exception as exc:

        raise RuntimeError(
            "Unable to read Chroma collection count. "
            f"Reason: {exc}"
        ) from exc


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "VECTOR STORE MODULE"
    )

    print("=" * 60)

    print(
        f"Chroma directory : {CHROMA_DIR}"
    )

    print(
        f"Collection       : {COLLECTION_NAME}"
    )

    print(
        f"Embedding model  : {EMBEDDING_MODEL}"
    )

    print(
        f"Chunk size       : {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap    : {CHUNK_OVERLAP}"
    )

    print(
        f"Default Top-K    : {DEFAULT_TOP_K}"
    )

    print(
        f"Device           : {get_device()}"
    )

    print(
        "\nVector store module loaded successfully."
    )