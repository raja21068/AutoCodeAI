"""tests/test_agents.py — Unit tests for all agent classes."""

import pytest

from core.agents.agents import CoderAgent, MemoryAgent


class TestCoderAgentApplyDiff:
    def test_no_diff_returns_original(self):
        original = "x = 1\ny = 2\n"
        result   = CoderAgent._apply_diff(original, "")
        assert result == original

    def test_apply_simple_addition(self):
        original = "x = 1\ny = 2\n"
        diff = (
            "--- a/code.py\n"
            "+++ b/code.py\n"
            "@@ -1,2 +1,3 @@\n"
            " x = 1\n"
            " y = 2\n"
            "+z = 3\n"
        )
        result = CoderAgent._apply_diff(original, diff)
        assert "z = 3" in result

    def test_invalid_diff_returns_original(self):
        original = "x = 1\n"
        result   = CoderAgent._apply_diff(original, "not a valid diff at all")
        assert result == original

    def test_extracts_diff_from_fenced_block(self):
        original = "a = 1\n"
        fenced = (
            "Here is the diff:\n"
            "```diff\n"
            "--- a/code.py\n"
            "+++ b/code.py\n"
            "@@ -1 +1 @@\n"
            "-a = 1\n"
            "+a = 99\n"
            "```\n"
        )
        result = CoderAgent._apply_diff(original, fenced)
        assert "99" in result


class TestMemoryAgent:
    def test_store_and_retrieve(self):
        mem = MemoryAgent()
        mem.store("write a sort function", "def sort(lst): return sorted(lst)")
        result = mem.retrieve("sort function")
        assert "sort" in result

    def test_empty_retrieve_returns_empty_string(self):
        mem    = MemoryAgent()
        result = mem.retrieve("something unrelated")
        assert result == ""

    def test_store_caps_at_20_entries(self):
        mem = MemoryAgent()
        for i in range(25):
            mem.store(f"task {i}", f"result {i}")
        assert len(mem._store) == 20

    def test_retrieve_respects_top_k(self):
        mem = MemoryAgent()
        for i in range(5):
            mem.store(f"auth task {i}", f"auth result {i}")
        result = mem.retrieve("auth", top_k=2)
        lines  = [l for l in result.split("\n") if l]
        assert len(lines) <= 2
