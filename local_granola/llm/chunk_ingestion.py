"""
Take stable live STT segments and feed them into the summary tree.

Preview text is ignored. Work happens on a background thread so the UI
does not freeze during embed/Qwen calls.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

from summary_tree import Tree

MIN_WORDS = 8
MIN_SECONDS = 4.0
IDLE_SECONDS = 3.0


class ChunkIngestionService:
    def __init__(self) -> None:
        self.tree = Tree(word_threshold=25, time_threshold=15.0, offline=False)
        self._incoming: queue.Queue = queue.Queue()
        self._buffer: list[str] = []
        self._buffer_started: float | None = None
        self._last_stable_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._outline_listeners: list[Callable[[str, str], None]] = []
        self._status_listeners: list[Callable[[str], None]] = []

    def add_outline_listener(self, listener: Callable[[str, str], None]) -> None:
        self._outline_listeners.append(listener)

    def add_status_listener(self, listener: Callable[[str], None]) -> None:
        self._status_listeners.append(listener)

    def handle_update(self, update) -> None:
        if update.is_preview:
            return
        text = update.segment.text.strip()
        if not text:
            return
        self._incoming.put(text)

    def start(self) -> None:
        self._stop.clear()
        self.tree = Tree(word_threshold=25, time_threshold=15.0, offline=False)
        self._buffer = []
        self._buffer_started = None
        self._show("(loading nomic… then the outline will appear here)")
        self._thread = threading.Thread(target=self._run, daemon=True, name="tree-ingest")
        self._thread.start()
        qwen = "on" if self.tree.use_qwen else "off"
        self._status("Summary tree started (nomic on, qwen %s)." % qwen)

    def stop(self) -> str:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=180)
            self._thread = None
        return self.tree.outline()

    def _run(self) -> None:
        try:
            from local_models import ensure_embedder

            self._status("Loading nomic embedder...")
            ensure_embedder()
            self._status("Nomic ready. Listening for transcript chunks.")
            self._show("(nomic ready — speak, outline updates after a few seconds)")
        except Exception as exc:
            self._status(f"Model load failed: {exc}")
            self._show(f"(embedder failed: {exc})")

        try:
            while not self._stop.is_set():
                try:
                    text = self._incoming.get(timeout=0.4)
                    self._buffer.append(text)
                    if self._buffer_started is None:
                        self._buffer_started = time.time()
                    self._last_stable_at = time.time()
                    if self._buffer_ready():
                        self._flush_buffer()
                except queue.Empty:
                    if self._should_idle_flush():
                        self._flush_buffer()

            if self._buffer:
                self._flush_buffer()
            self.tree.flush_all()
            self.tree.rollup_dirty_nodes()
            self._emit_outline()
            self._status("Summary tree flushed.")
        except Exception as exc:
            self._status(f"Tree ingest crashed: {exc}")
            self._show(f"(tree ingest crashed: {exc})")

    def _should_idle_flush(self) -> bool:
        if not self._buffer:
            return False
        now = time.time()
        if (now - self._last_stable_at) >= IDLE_SECONDS:
            return True
        if self._buffer_started is not None and (now - self._buffer_started) >= MIN_SECONDS:
            return True
        return False

    def _buffer_ready(self) -> bool:
        words = sum(len(t.split()) for t in self._buffer)
        if words >= MIN_WORDS:
            return True
        if self._buffer_started is None:
            return False
        return (time.time() - self._buffer_started) >= MIN_SECONDS

    def _flush_buffer(self) -> None:
        chunk = " ".join(self._buffer).strip()
        self._buffer = []
        self._buffer_started = None
        if not chunk:
            return
        self._status("Updating topic outline...")
        self._show("(updating outline…)\n" + self.tree.outline())
        try:
            node = self.tree.insert_chunk(chunk)
        except Exception as exc:
            self._status(f"Tree update failed: {exc}")
            self._show(f"(tree update failed: {exc})")
            return
        self._status(f"Tree: {node.title} (d{node.depth}) [{self.tree.last_reason}]")
        self._emit_outline()

    def _emit_outline(self) -> None:
        self._show(self.tree.outline())

    def _show(self, text: str) -> None:
        reason = self.tree.last_reason
        for listener in self._outline_listeners:
            listener(text, reason)

    def _status(self, message: str) -> None:
        for listener in self._status_listeners:
            listener(message)
