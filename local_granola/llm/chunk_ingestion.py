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

MIN_WORDS = 18
MIN_SECONDS = 8.0
IDLE_SECONDS = 6.0


class ChunkIngestionService:
    def __init__(self) -> None:
        self.tree = Tree(word_threshold=150, time_threshold=30.0, offline=False)
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
        self.tree = Tree(word_threshold=150, time_threshold=30.0, offline=False)
        self._buffer = []
        self._buffer_started = None
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
            from local_models import ensure_embedder, ensure_qwen, qwen_ready

            self._status("Loading nomic embedder...")
            ensure_embedder()
            if qwen_ready():
                self._status("Loading Qwen...")
                ensure_qwen()
            self._status("Models ready. Waiting for speech.")
        except Exception as exc:
            self._status(f"Model load failed: {exc}")

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
                if self._buffer and (time.time() - self._last_stable_at) >= IDLE_SECONDS:
                    self._flush_buffer()

        if self._buffer:
            self._flush_buffer()
        self.tree.flush_all()
        self.tree.rollup_dirty_nodes()
        self._emit_outline()
        self._status("Summary tree flushed.")

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
        self._status("Placing chunk in the tree...")
        node = self.tree.insert_chunk(chunk)
        self._status(
            f"Tree: {node.title} (d{node.depth}) [{self.tree.last_reason}]"
        )
        self._emit_outline()

    def _emit_outline(self) -> None:
        text = self.tree.outline()
        reason = self.tree.last_reason
        for listener in self._outline_listeners:
            listener(text, reason)

    def _status(self, message: str) -> None:
        for listener in self._status_listeners:
            listener(message)
