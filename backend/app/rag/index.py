"""Chroma-backed persistent indexes for schema chunks and SQL examples."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
import logging
import os
import warnings

from backend.app.rag.chunks import build_column_chunks
from backend.app.rag.examples import EXAMPLE_PAIRS


class RAGIndexMissing(Exception):
    """Raised when the persistent RAG index has not been built yet."""


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    _quiet_transformers_import_noise()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Accessing `__path__` from `\.models\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Returning `__path__` instead.*",
        )
        from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _make_ef() -> Any:
    """Wrap the already-downloaded sentence-transformers model as a ChromaDB EF.

    Passing this to every collection create/get prevents ChromaDB from downloading
    its own 79 MB ONNX model at runtime.
    """
    from chromadb import EmbeddingFunction

    class _SentenceTransformerEF(EmbeddingFunction):
        def __call__(self, input: list[str]) -> list[list[float]]:
            return get_embedder().encode(list(input)).tolist()

    return _SentenceTransformerEF()


def build_chroma_indexes(persist_dir: Path) -> None:
    _quiet_transformers_import_noise()
    import chromadb

    ef = _make_ef()
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    schema_collection = client.get_or_create_collection("schema_chunks", embedding_function=ef)
    example_collection = client.get_or_create_collection("examples", embedding_function=ef)
    _upsert(schema_collection, build_column_chunks())
    _upsert(example_collection, [
        {**pair, "text": f"Q: {pair['question']}\nSQL: {pair['sql']}"}
        for pair in EXAMPLE_PAIRS
    ])


def load_chroma_indexes(persist_dir: Path) -> tuple[Any, Any]:
    if not persist_dir.exists():
        raise RAGIndexMissing(f"RAG index not found at {persist_dir}. Run scripts/build_rag_index.py.")
    try:
        import chromadb

        ef = _make_ef()
        client = chromadb.PersistentClient(path=str(persist_dir))
        return (
            client.get_collection("schema_chunks", embedding_function=ef),
            client.get_collection("examples", embedding_function=ef),
        )
    except Exception as exc:
        raise RAGIndexMissing(str(exc)) from exc


def _upsert(collection: Any, docs: list[dict[str, Any]]) -> None:
    ids = [str(doc["id"]) for doc in docs]
    documents = [str(doc["text"]) for doc in docs]
    metadatas = [
        {k: str(v) for k, v in doc.items() if k not in {"id", "text"}}
        for doc in docs
    ]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def _quiet_transformers_import_noise() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*Accessing `__path__` from `\.models\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Returning `__path__` instead.*",
    )
