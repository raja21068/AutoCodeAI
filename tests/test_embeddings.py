"""tests/test_embeddings.py — Unit tests for ChromaDB embedding helpers."""

from unittest.mock import MagicMock, patch
import pytest

import memory.vector.embeddings as emb_module


@patch("memory.vector.embeddings._get_embed_client")
def test_get_embedding_truncates_long_text(mock_client):
    mock_response       = MagicMock()
    mock_response.data  = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.return_value.embeddings.create.return_value = mock_response

    long_text = "x" * 20_000
    result    = emb_module.get_embedding(long_text)

    call_args = mock_client.return_value.embeddings.create.call_args
    assert len(call_args.kwargs["input"]) <= 12_000
    assert result == [0.1, 0.2, 0.3]


@patch("memory.vector.embeddings._get_collection")
def test_store_embedding_calls_upsert(mock_col):
    mock_collection = MagicMock()
    mock_col.return_value = mock_collection

    emb_module.store_embedding(
        [0.1, 0.2],
        {"path": "foo.py", "content": "x = 1", "type": "file"},
        doc_id="foo.py::abc123",
    )

    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args.kwargs
    assert call_kwargs["ids"] == ["foo.py::abc123"]


@patch("memory.vector.embeddings._get_collection")
def test_query_embedding_returns_hits(mock_col):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock_collection.query.return_value = {
        "metadatas": [[{"path": "a.py", "content": "x=1"}]],
        "distances": [[0.1]],
    }
    mock_col.return_value = mock_collection

    results = emb_module.query_embedding([0.1, 0.2], top_k=1)
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.9)
    assert results[0]["metadata"]["path"] == "a.py"
