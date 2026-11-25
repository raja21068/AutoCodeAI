"""tests/test_orchestrator.py — Unit tests for Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.orchestrator import Orchestrator


@pytest.fixture
def orch():
    with patch("services.orchestrator.DockerSandbox"), \
         patch("services.orchestrator.RepoIndexer"):
        o = Orchestrator()
        o.sandbox  = MagicMock()
        o.indexer  = None
        return o


@pytest.mark.asyncio
async def test_get_file_content_missing_file():
    result = Orchestrator._get_file_content("/nonexistent/path/file.py")
    assert result == ""


@pytest.mark.asyncio
async def test_get_file_content_empty_path():
    result = Orchestrator._get_file_content("")
    assert result == ""


@pytest.mark.asyncio
async def test_run_calls_callback(orch):
    orch.planner.create_plan  = AsyncMock(return_value={
        "explanation": "test plan",
        "steps": [{"agent": "coder", "description": "write hello world"}],
    })

    async def fake_stream(*args, **kwargs):
        yield "def hello(): return 'world'\n"

    orch.coder.stream_code = fake_stream
    orch.critic.review     = AsyncMock(return_value="PASS")
    orch.memory.retrieve   = MagicMock(return_value="")

    messages = []
    await orch.run("write hello world", [], callback=lambda m: messages.append(m) or __import__('asyncio').sleep(0))
    assert any("Plan" in m or "plan" in m for m in messages)


@pytest.mark.asyncio
async def test_run_streaming_yields_chunks(orch):
    orch.planner.create_plan = AsyncMock(return_value={
        "explanation": "simple",
        "steps": [{"agent": "coder", "description": "write code"}],
    })

    async def fake_stream(*args, **kwargs):
        yield "chunk1"
        yield "chunk2"

    orch.coder.stream_code = fake_stream
    orch.critic.review     = AsyncMock(return_value="PASS")
    orch.memory.retrieve   = MagicMock(return_value="")

    chunks = []
    async for chunk in orch.run_streaming("write code", []):
        chunks.append(chunk)

    assert len(chunks) > 0
