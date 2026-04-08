"""
memory/repo_indexer.py — Live repository indexer using Watchdog.

Indexes all source files into ChromaDB on startup, then watches for
create / modify / delete events and keeps the index in sync.
"""

import hashlib
import logging
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from memory.vector.embeddings import (
    delete_by_path,
    get_embedding,
    query_embedding,
    store_embedding,
)

logger = logging.getLogger(__name__)

INDEXABLE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs"}


class RepoIndexer(FileSystemEventHandler):
    def __init__(
        self,
        repo_path: str,
        index_path: str = "./.ai_coding_index",
    ) -> None:
        super().__init__()
        self.repo_path  = Path(repo_path).resolve()
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._observer: Observer | None = None
        self._watching = False
        self._index_existing()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _file_hash(filepath: Path) -> str:
        return hashlib.md5(filepath.read_bytes()).hexdigest()

    @staticmethod
    def _should_index(filepath: Path) -> bool:
        return filepath.suffix in INDEXABLE_SUFFIXES

    def _index_file(self, filepath: Path) -> None:
        try:
            rel_path = str(filepath.relative_to(self.repo_path))
            content  = filepath.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return
            doc_id    = f"{rel_path}::{self._file_hash(filepath)}"
            embedding = get_embedding(content)
            store_embedding(
                embedding,
                {"path": rel_path, "content": content, "type": "file"},
                doc_id=doc_id,
            )
            logger.debug("Indexed %s", rel_path)
        except Exception as exc:
            logger.warning("Failed to index %s: %s", filepath, exc)

    def _index_existing(self) -> None:
        logger.info("Indexing %s …", self.repo_path)
        for filepath in self.repo_path.rglob("*"):
            if filepath.is_file() and self._should_index(filepath):
                self._index_file(filepath)
        logger.info("Initial indexing complete.")

    # ------------------------------------------------------------------
    # Watchdog handlers
    # ------------------------------------------------------------------

    def on_created(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if not event.is_directory and self._should_index(path):
            self._index_file(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if not event.is_directory and self._should_index(path):
            self._index_file(path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if not event.is_directory and self._should_index(path):
            try:
                rel_path = str(path.relative_to(self.repo_path))
                delete_by_path(rel_path)
            except Exception as exc:
                logger.warning("Failed to remove index for %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_watching(self) -> None:
        if self._watching:
            return
        self._observer = Observer()
        self._observer.schedule(self, str(self.repo_path), recursive=True)
        self._observer.start()
        self._watching = True
        logger.info("Watching %s for changes.", self.repo_path)

    def stop(self) -> None:
        if self._observer and self._watching:
            self._observer.stop()
            self._observer.join()
            self._watching = False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_relevant(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = get_embedding(query)
        return query_embedding(query_vec, top_k)
