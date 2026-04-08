"""
memory/vector/embeddings.py — ChromaDB-backed vector store.

Uses OpenAI text-embedding-3-small for embeddings.
Falls back to an in-process ephemeral client when ChromaDB is unreachable
(useful for local development and CI).
"""

import logging
import os

import chromadb
from openai import OpenAI

logger = logging.getLogger(__name__)

_chroma_client: chromadb.Client | None = None
_embed_client:  OpenAI | None          = None
COLLECTION_NAME = "code_index"


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

def _get_chroma() -> chromadb.Client:
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


def _get_collection() -> chromadb.Collection:
    return _get_chroma().get_or_create_collection(
        name=COLLECTION_NAME,
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
) -> None:
    """Upsert an embedding + metadata into the ChromaDB collection."""
    col = _get_collection()
    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[metadata.get("content", "")[:2000]],
    )


def query_embedding(
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Return the top-k most similar documents as a list of dicts."""
    col = _get_collection()
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


def delete_by_path(rel_path: str) -> None:
    """Remove all embeddings whose 'path' metadata matches *rel_path*."""
    _get_collection().delete(where={"path": rel_path})
