"""
memory/vector/embeddings.py — ChromaDB-backed vector store.

Uses OpenAI text-embedding-3-small for embeddings.
Falls back to an in-process ephemeral client when ChromaDB is unreachable
(useful for local development and CI).

Two defects are fixed here. The module previously annotated a module-level
name as ``chromadb.Client | None``; ``chromadb.Client`` is a factory function
rather than a class, so evaluating that annotation raised ``TypeError`` at
import time and made this module — and everything importing it, including the
application orchestrator — unimportable.

It also indexed every repository into one collection named ``code_index``, so
chunks from different projects shared a namespace and could be retrieved for
the wrong repository. Collections are now namespaced per index.

This store backs the interactive application. Benchmark runs do not use it:
retrieval there is per-instance and runs inside the task container. See
``services/repo_retrieval.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

import chromadb
from openai import OpenAI

logger = logging.getLogger(__name__)

_chroma_client = None
_embed_client: OpenAI | None = None
DEFAULT_NAMESPACE = "code_index"


def collection_name(namespace: str | None = None) -> str:
    """
    Return a Chroma-legal collection name for *namespace*.

    Chroma requires 3-63 characters, alphanumeric plus underscore and hyphen,
    starting and ending alphanumeric — so an arbitrary repository path is
    slugified and suffixed with a hash to keep distinct paths distinct.
    """
    if not namespace:
        return DEFAULT_NAMESPACE
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(namespace)).strip("_-")[:40]
    digest = hashlib.sha1(str(namespace).encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'idx'}_{digest}"


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8001"))
        try:
            _chroma_client = chromadb.HttpClient(host=host, port=port)
            _chroma_client.heartbeat()          # verify connectivity
            logger.info("Connected to ChromaDB at %s:%s", host, port)
        except Exception:
            logger.warning("ChromaDB unreachable — using in-process ephemeral client.")
            _chroma_client = chromadb.Client()
    return _chroma_client


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _embed_client


def _get_collection(namespace: str | None = None):
    return _get_chroma().get_or_create_collection(
        name=collection_name(namespace),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    """Return an embedding vector for *text* using OpenAI."""
    text = text[:12_000]        # guard against token-limit errors
    response = _get_embed_client().embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def store_embedding(
    embedding: list[float],
    metadata: dict,
    doc_id: str,
    namespace: str | None = None,
) -> None:
    """Upsert an embedding + metadata into the namespaced collection."""
    col = _get_collection(namespace)
    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[metadata.get("content", "")[:2000]],
    )


def query_embedding(
    query_vector: list[float],
    top_k: int = 5,
    namespace: str | None = None,
) -> list[dict]:
    """Return the top-k most similar documents as a list of dicts."""
    col = _get_collection(namespace)
    results = col.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, col.count() or 1),
        include=["metadatas", "distances"],
    )
    return [
        {"metadata": meta, "score": 1 - dist}
        for meta, dist in zip(
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def delete_by_path(rel_path: str, namespace: str | None = None) -> None:
    """Remove all embeddings whose 'path' metadata matches *rel_path*."""
    _get_collection(namespace).delete(where={"path": rel_path})
