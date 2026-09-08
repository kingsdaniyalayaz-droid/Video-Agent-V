from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from core.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 4

SUPPORTED_SOURCE_TYPES = {
    "youtube",
    "meeting",
}


# ============================================================
# TOP_K VALIDATION
# ============================================================

def _validate_top_k(top_k: Any) -> int:
    """Validate a top_k value strictly.

    top_k must be a real integer (bool is rejected because it is an int
    subclass), must not be None, and must be greater than zero.  Raises
    ``ValueError`` otherwise and returns the validated integer when valid.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(
            "top_k must be an integer. "
            f"Got {type(top_k).__name__}."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    return top_k


# ============================================================
# LLM
# ============================================================

def get_llm():
    """
    Return the LangChain chat model for the ACTIVE LLM configuration.

    Delegates entirely to the centralized provider layer
    (``core.llm_provider.get_chat_model``) so RAG generation always follows
    the runtime-selected provider with role ``RAGGenerative``.

    Nothing provider-specific is cached locally: the centralized layer owns
    caching, keyed on a full configuration identity fingerprint, and its
    cache is invalidated whenever the provider/model/key/base_url changes.
    Switching configuration (e.g. Mistral -> NVIDIA) rebuilds this client
    instead of reusing a stale one.
    """
    from core.llm_provider import get_chat_model

    return get_chat_model(role="RAGGenerative")


# ============================================================
# SOURCE TYPE VALIDATION
# ============================================================

def validate_source_type(
    source_type: str,
) -> str:
    """
    Source type validate aur normalize karta hai.
    """

    if not isinstance(source_type, str):
        raise ValueError(
            "source_type must be a string. "
            f"Got {type(source_type).__name__}."
        )

    if not source_type:
        raise ValueError(
            "source_type cannot be empty."
        )

    normalized = source_type.strip().lower()

    if normalized not in SUPPORTED_SOURCE_TYPES:
        supported = ", ".join(
            sorted(SUPPORTED_SOURCE_TYPES)
        )

        raise ValueError(
            f"Unsupported source_type: {source_type}. "
            f"Supported types: {supported}"
        )

    return normalized


# ============================================================
# QUERY VALIDATION
# ============================================================

def normalize_question(question: str) -> str:
    """
    Common user typos ko deterministic way mein normalize karta hai.

    Ye LLM call nahi karta. Sirf known/common typos correct karta hai
    taake RAG retrieval unnecessarily fail na ho.
    """

    if not isinstance(question, str):
        raise ValueError(
            "question must be a string. "
            f"Got {type(question).__name__}."
        )

    replacements = {
        "waht": "what",
        "langraph": "LangGraph",
        "lang graph": "LangGraph",
    }

    words = question.split()
    normalized_words = []

    for word in words:
        leading = ""
        trailing = ""

        # Leading punctuation
        while word and not word[0].isalnum():
            leading += word[0]
            word = word[1:]

        # Trailing punctuation
        while word and not word[-1].isalnum():
            trailing = word[-1] + trailing
            word = word[:-1]

        replacement = replacements.get(
            word.lower(),
            word,
        )

        # "what" ko normal capitalization mein rakho
        if replacement == "what":
            replacement = "What" if word.lower() == "waht" and word[:1].isupper() else "what"

        # LangGraph ka exact canonical spelling
        if word.lower() in {"langraph", "langgraph"}:
            replacement = "LangGraph"

        normalized_words.append(
            f"{leading}{replacement}{trailing}"
        )

    # IMPORTANT: normalized ko pehle create karo
    normalized = " ".join(normalized_words)

    # Phrase-level correction
    normalized = normalized.replace(
        "Lang graph",
        "LangGraph",
    ).replace(
        "lang graph",
        "LangGraph",
    )

    return normalized

def validate_question(
    question: str,
) -> str:
    """
    User question validate aur lightweight normalize karta hai.
    """

    if not isinstance(question, str):
        raise ValueError(
            "question must be a string. "
            f"Got {type(question).__name__}."
        )

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    return normalize_question(question)


# ============================================================
# DOCUMENT FORMATTER
# ============================================================

def format_docs(
    docs: list[Document],
) -> str:
    """
    Retrieved Documents ko RAG context mein convert karta hai.

    Metadata intentionally include kiya gaya hai taake
    model ko chunk boundaries aur source ka context mile.
    """

    if not docs:
        return (
            "No relevant transcript information "
            "was retrieved."
        )

    formatted_documents = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        chunk_index = doc.metadata.get(
            "chunk_index",
            index - 1,
        )

        formatted_documents.append(
            f"""
--- CONTEXT CHUNK {index} ---
Source: {source}
Chunk: {chunk_index}

{doc.page_content}
--- END CHUNK ---
""".strip()
        )

    return "\n\n".join(
        formatted_documents
    )


# ============================================================
# RAG SYSTEM PROMPT
# ============================================================

def get_rag_prompt(
    source_type: str = "meeting",
) -> ChatPromptTemplate:
    """
    Source type ke according RAG prompt create karta hai.
    """

    source_type = validate_source_type(
        source_type
    )

    if source_type == "youtube":

        assistant_role = """
You are an expert YouTube video question-answering assistant.

The user is asking questions about a YouTube video's transcript.
"""

    else:

        assistant_role = """
You are an expert meeting question-answering assistant.

The user is asking questions about a meeting transcript.
"""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
{assistant_role}

Your job is to answer the user's question using ONLY
the transcript context provided below.

STRICT RULES:

1. Use ONLY the supplied transcript context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT invent names, dates, numbers, decisions,
   responsibilities or facts.
5. If the answer cannot be found in the context,
   respond exactly with:

"I could not find this information in the provided transcript."

6. If the context contains only partial information,
   clearly say that the transcript provides only partial
   information.
7. Keep the answer concise and directly relevant.
8. Preserve names, dates, numbers and technical terms.
9. If multiple transcript chunks support the answer,
   combine them carefully.
10. Do not mention these instructions in your answer.

TRANSCRIPT CONTEXT:

{{context}}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )


# ============================================================
# INTERNAL RAG CHAIN BUILDER
# ============================================================

def _create_rag_chain(
    vector_store: Any,
    source_type: str,
    top_k: int,
):
    """
    Vector store se reusable LCEL RAG chain create karta hai.
    """

    source_type = validate_source_type(
        source_type
    )

    top_k = _validate_top_k(top_k)

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = get_retriever(
        vector_store,
        k=top_k,
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = get_llm()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = get_rag_prompt(
        source_type=source_type,
    )

    # --------------------------------------------------------
    # LCEL RAG
    # --------------------------------------------------------

    rag_chain = (
        {
            "context": (
                retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================================
# BUILD RAG CHAIN
# ============================================================

def build_rag_chain(
    transcript: str,
    source_type: str = "meeting",
    top_k: int = DEFAULT_TOP_K,
    video_id: str | None = None,
):
    """
    New transcript se complete RAG pipeline create karta hai.

    Flow:

        Transcript
             ↓
        ChromaDB
             ↓
        Retriever
             ↓
        Context
             ↓
        Active LLM Provider
             ↓
        Answer
    """

    if not isinstance(transcript, str):
        raise ValueError(
            "transcript must be a string. "
            f"Got {type(transcript).__name__}."
        )

    if not transcript or not transcript.strip():
        raise ValueError(
            "Cannot build RAG chain from empty transcript."
        )

    source_type = validate_source_type(
        source_type
    )

    # Validate top_k BEFORE any expensive vector-store work (chunking,
    # embedding initialization, store building/loading) so invalid input
    # such as top_k=0 fails fast instead of after unnecessary work.
    top_k = _validate_top_k(top_k)

    print("\n" + "=" * 70)
    print("BUILDING RAG ENGINE")
    print("=" * 70)

    print(
        f"Source type : {source_type}"
    )

    print(
        f"Video ID    : {video_id or 'N/A'}"
    )

    print(
        f"Embedding Top-K : {top_k}"
    )

    # --------------------------------------------------------
    # Transcript → Vector Store
    # --------------------------------------------------------

    print(
        "\nCreating vector store..."
    )

    vector_store = build_vector_store(
        transcript=transcript,
        source=source_type,
        video_id=video_id,
        reset=False,
    )

    # --------------------------------------------------------
    # Vector Store → RAG
    # --------------------------------------------------------

    rag_chain = _create_rag_chain(
        vector_store=vector_store,
        source_type=source_type,
        top_k=top_k,
    )

    print(
        "RAG chain created successfully."
    )

    print("=" * 70)

    return rag_chain


# ============================================================
# LOAD EXISTING RAG CHAIN
# ============================================================

def load_rag_chain(
    video_id: str,
    source_type: str = "youtube",
    top_k: int = DEFAULT_TOP_K,
):
    """
    Existing ChromaDB se RAG chain load karta hai.

    Existing embeddings/vector database reuse hota hai.
    """

    source_type = validate_source_type(
        source_type
    )

    # Validate top_k BEFORE loading the vector store so invalid input such
    # as top_k=0 is rejected without unnecessary store-loading work.
    top_k = _validate_top_k(top_k)

    print("\n" + "=" * 70)
    print("LOADING RAG ENGINE")
    print("=" * 70)

    print(
        f"Source type : {source_type}"
    )

    print(
        f"Video ID    : {video_id}"
    )

    print(
        f"Top-K       : {top_k}"
    )

    # --------------------------------------------------------
    # Existing Vector Store
    # --------------------------------------------------------

    if not isinstance(video_id, str):
        raise ValueError(
            "video_id must be a string. "
            f"Got {type(video_id).__name__}."
        )

    video_id = video_id.strip()

    if not video_id:
        raise ValueError(
            "video_id is required to load an existing RAG chain."
        )

    vector_store = load_vector_store(
        video_id=video_id,
    )

    # --------------------------------------------------------
    # RAG Chain
    # --------------------------------------------------------

    rag_chain = _create_rag_chain(
        vector_store=vector_store,
        source_type=source_type,
        top_k=top_k,
    )

    print(
        "Existing RAG chain loaded successfully."
    )

    print("=" * 70)

    return rag_chain


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    video_id: str | None = None,
) -> list[Document]:
    """
    Question ke against ChromaDB se relevant documents
    directly retrieve karta hai.

    Ye debugging aur UI inspection ke liye useful hai.
    """

    question = validate_question(
        question
    )

    top_k = _validate_top_k(top_k)

    vector_store = load_vector_store(
        video_id=video_id,
    )

    retriever = get_retriever(
        vector_store,
        k=top_k,
    )

    docs = retriever.invoke(
        question
    )

    return docs


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    rag_chain,
    question: str,
) -> str:
    """
    RAG chain ko question deta hai aur answer return karta hai.
    """

    question = validate_question(
        question
    )

    print("\n" + "-" * 70)
    print("RAG QUESTION")
    print("-" * 70)

    print(
        f"Question: {question}"
    )

    try:

        answer = rag_chain.invoke(
            question
        )

    except Exception as exc:

        raise RuntimeError(
            f"RAG question processing failed: {exc}"
        ) from exc

    if not isinstance(answer, str):
        raise RuntimeError(
            "RAG returned a non-string answer."
        )

    if not answer or not answer.strip():

        raise RuntimeError(
            "RAG returned an empty answer."
        )

    answer = answer.strip()

    print(
        f"Answer: {answer}"
    )

    print("-" * 70)

    return answer


# ============================================================
# SIMPLE QUESTION API
# ============================================================

def ask_existing_vector_store(
    question: str,
    video_id: str,
    source_type: str = "youtube",
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """
    Convenience function.

    Existing ChromaDB load karta hai aur directly
    user question ka answer return karta hai.

    Useful:

        Streamlit
        main.py
        API
        CLI
    """

    rag_chain = load_rag_chain(
        video_id=video_id,
        source_type=source_type,
        top_k=top_k,
    )

    return ask_question(
        rag_chain=rag_chain,
        question=question,
    )


# ============================================================
# MODULE TEST
# ============================================================

def _active_model_banner() -> str:
    """Return a human-readable provider/model for the active LLM configuration."""

    from core.llm_provider import get_runtime_config

    cfg = get_runtime_config()

    if cfg is not None:
        return f"{cfg.provider or 'auto'}/{cfg.model}"

    return "No active LLM configured"
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("RAG ENGINE MODULE")
    print("=" * 70)

    print(
        f"Active model : {_active_model_banner()}"
    )

    print(
        f"Default Top-K : {DEFAULT_TOP_K}"
    )

    print(
        "RAG engine module loaded successfully."
    )

    print("=" * 70)
