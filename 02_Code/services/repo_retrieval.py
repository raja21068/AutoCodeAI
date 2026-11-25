"""
services/repo_retrieval.py — Per-instance repository retrieval.

Retrieval runs *inside* the instance container, over the base checkout only.
This replaces the host-side ChromaDB indexer for benchmark runs, which shared
one global collection across repositories and was in any case never enabled
during evaluation (the benchmark constructed an ``Orchestrator`` without a
``repo_path``, leaving the indexer ``None``).

The strategy is deliberately simple and fully specified, so the paper can
describe it exactly:

1. **Query terms.** Extract candidate identifiers from the issue text:
   ``dotted.paths``, ``CamelCase`` names, ``snake_case`` names of length >= 4,
   and quoted or backticked spans. Terms are ranked by rarity in the issue.
2. **Candidate files.** ``grep -rl`` each term across tracked ``.py`` files,
   excluding test directories. A file's score is the number of distinct query
   terms it matches, with a bonus when the term appears in a ``def``/``class``
   declaration rather than in an arbitrary line.
3. **Chunking.** Each selected file is split at top-level ``def``/``class``
   boundaries into chunks of at most ``CHUNK_LINES`` lines with
   ``CHUNK_OVERLAP`` lines of overlap. Chunks are scored by the same term
   count and the best ones are kept.
4. **Truncation.** Chunks are emitted in descending score order until
   ``MAX_CONTEXT_CHARS`` is reached; a partially-consumed chunk is dropped
   rather than cut mid-line.

No query and no index entry is derived from the reference patch.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CHUNK_LINES = 120
CHUNK_OVERLAP = 20
MAX_FILES = 8
MAX_CHUNKS = 12
MAX_CONTEXT_CHARS = 24_000
EXCLUDE_DIRS = ("*/test/*", "*/tests/*", "*/testing/*", "*/.git/*")

_STOPWORDS = {
    "self", "none", "true", "false", "return", "import", "class", "print",
    "error", "value", "python", "traceback", "example", "should", "which",
    "when", "with", "this", "that", "from", "have", "into", "code", "line",
}


@dataclass
class Chunk:
    path: str
    start_line: int
    text: str
    score: int


class RepoRetriever:
    def __init__(self, env) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_terms(issue: str, limit: int = 12) -> list[str]:
        terms: list[str] = []
        terms += re.findall(r"`([A-Za-z_][\w./]{3,})`", issue)
        terms += re.findall(r"\b([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)\b", issue)
        terms += re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", issue)
        terms += re.findall(r"\b([a-z_][a-z0-9_]{3,})\b", issue)

        seen: dict[str, int] = {}
        for term in terms:
            term = term.strip(".`")
            if len(term) < 4 or term.lower() in _STOPWORDS:
                continue
            seen[term] = seen.get(term, 0) + 1

        # Rare terms first: a name mentioned once is more discriminative than
        # a word repeated throughout the report.
        ordered = sorted(seen, key=lambda t: (seen[t], -len(t)))
        return ordered[:limit]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, issue: str) -> list[Chunk]:
        terms = self.extract_terms(issue)
        if not terms:
            return []

        exclude = " ".join(f"--exclude-dir={d.strip('*/')}" for d in ("test", "tests", "testing"))
        file_scores: dict[str, int] = {}
        decl_hits: dict[str, int] = {}

        for term in terms:
            code, out, _ = self.env.exec(
                f"grep -rl --include='*.py' {exclude} -F {shlex.quote(term)} . "
                f"2>/dev/null | head -40"
            )
            if code != 0 and not out:
                continue
            for path in filter(None, (p.strip() for p in out.splitlines())):
                file_scores[path] = file_scores.get(path, 0) + 1

            code, out, _ = self.env.exec(
                f"grep -rl --include='*.py' {exclude} -E "
                f"{shlex.quote(rf'^[[:space:]]*(def|class)[[:space:]]+{re.escape(term)}')} . "
                f"2>/dev/null | head -20"
            )
            for path in filter(None, (p.strip() for p in out.splitlines())):
                decl_hits[path] = decl_hits.get(path, 0) + 1

        if not file_scores:
            return []

        ranked = sorted(
            file_scores,
            key=lambda p: (file_scores[p] + 3 * decl_hits.get(p, 0)),
            reverse=True,
        )[:MAX_FILES]

        chunks: list[Chunk] = []
        for path in ranked:
            content = self.env.read_file(path.lstrip("./"))
            if content:
                chunks.extend(self._chunk_file(path, content, terms))

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:MAX_CHUNKS]

    @staticmethod
    def _chunk_file(path: str, content: str, terms: list[str]) -> list[Chunk]:
        lines = content.splitlines()
        chunks: list[Chunk] = []
        step = CHUNK_LINES - CHUNK_OVERLAP

        for start in range(0, max(len(lines), 1), step):
            window = lines[start : start + CHUNK_LINES]
            if not window:
                break
            text = "\n".join(window)
            score = sum(1 for t in terms if t in text)
            if score:
                chunks.append(Chunk(path, start + 1, text, score))
        return chunks

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format(chunks: list[Chunk]) -> str:
        out, used = [], 0
        for chunk in chunks:
            block = (
                f"### {chunk.path}:{chunk.start_line}\n"
                f"```python\n{chunk.text}\n```"
            )
            if used + len(block) > MAX_CONTEXT_CHARS:
                break
            out.append(block)
            used += len(block)
        return "\n\n".join(out)
