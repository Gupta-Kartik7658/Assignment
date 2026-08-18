"""
Tree-based real-time transcript summarizer.

LLM budget:
  - per chunk: 0 LLM calls if embedder is sure it's the same topic
  - 1 short Qwen call if embedder thinks we might need a new node
  - plus 0 or 1 Qwen call to update that node's own summary (buffer threshold)
  - rollups are lazy, never per-chunk
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from embedder import HIGH, LOW, cosine

MAX_DEPTH = 4
DEBOUNCE = 2  # need this many "not the same topic" votes in a row before Qwen

llm_call_count = 0


def dummy_embed(text: str) -> list[float]:
    return [0.0] * 8


def fake_qwen(prompt: str, max_tokens: int = 120) -> str:
    global llm_call_count
    llm_call_count += 1
    words = prompt.split("---", 1)[-1].split()
    kept = " ".join(words[:30])
    tail = "..." if len(words) > 30 else ""
    return f"[fake #{llm_call_count}] {kept}{tail}"


@dataclass
class SummaryNode:
    title: str = "session"
    parent: Optional["SummaryNode"] = None
    children: list["SummaryNode"] = field(default_factory=list)
    depth: int = 0

    raw_chunks: list[str] = field(default_factory=list)
    summary: str = ""
    subtree_summary: str = ""
    embedding: list[float] = field(default_factory=list)

    status: Literal["active", "stale"] = "stale"
    dirty: bool = False

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def word_count(self) -> int:
        return sum(len(c.split()) for c in self.raw_chunks)

    def topic_text(self) -> str:
        parts = []
        if self.summary:
            parts.append(self.summary)
        parts.extend(self.raw_chunks)
        return " ".join(parts).strip()


class Tree:
    def __init__(
        self,
        word_threshold: int = 150,
        time_threshold: float = 30.0,
        offline: bool = True,
    ):
        self.word_threshold = word_threshold
        self.time_threshold = time_threshold
        self.offline = offline
        if offline:
            self.use_embed = False
            self.use_qwen = False
        else:
            self.use_embed = True
            from local_models import qwen_ready

            self.use_qwen = qwen_ready()
        self.root = SummaryNode(title="session", depth=0, status="active")
        self.active = self.root
        self.away_votes = 0
        self.last_reason = "start"

        if self.use_embed:
            from embedder import embed as real_embed

            self._embed = real_embed
        else:
            self._embed = dummy_embed

    def _qwen(self, prompt: str, max_tokens: int = 120) -> str:
        global llm_call_count
        if not self.use_qwen:
            return fake_qwen(prompt, max_tokens)
        from llm import qwen_call

        text = qwen_call(prompt, max_tokens=max_tokens)
        llm_call_count += 1
        return text

    # ------------------------------------------------------------ placement

    def decide_placement(self, chunk_embedding, chunk: str, hint=None) -> SummaryNode:
        """
        Two-stage:
          1. cosine vs active / nearby nodes
          2. only if that looks like a split, ask Qwen: SAME / SUB / NEW

        `hint` is a test override: None / "child" / "sibling"
        """
        if hint == "child":
            self.last_reason = "hint-child"
            return self._new_node(self.active)
        if hint == "sibling":
            self.last_reason = "hint-sibling"
            parent = self.active.parent or self.root
            return self._new_node(parent)
        if hint is not None:
            raise ValueError(f"unknown hint: {hint}")

        if not self.use_embed:
            self.last_reason = "offline-active"
            return self.active

        if not self.active.topic_text():
            self.last_reason = "first-chunk"
            self.away_votes = 0
            return self.active

        active_score = 0.0
        if self.active.embedding:
            active_score = cosine(chunk_embedding, self.active.embedding)

        best_node = self.active
        best_score = active_score
        for node in self._candidates():
            if not node.embedding:
                continue
            score = cosine(chunk_embedding, node.embedding)
            if score > best_score:
                best_score = score
                best_node = node

        if best_score >= HIGH:
            self.last_reason = f"embed-sure {best_score:.2f}"
            self.away_votes = 0
            return best_node

        self.away_votes += 1
        if self.away_votes < DEBOUNCE:
            self.last_reason = f"debounce {self.away_votes}/{DEBOUNCE} ({best_score:.2f})"
            return self.active

        self.away_votes = 0
        decision = self._verify(chunk, best_score)
        tag = "qwen" if self.use_qwen else "embed-guess"
        self.last_reason = f"{tag}-{decision} ({best_score:.2f})"
        if decision == "SAME":
            return self.active
        if decision == "SUB":
            return self._new_node(self.active)
        parent = self.active.parent or self.root
        return self._new_node(parent)

    def _candidates(self) -> list[SummaryNode]:
        active = self.active
        nodes = [active]
        if active.parent:
            nodes.append(active.parent)
            nodes.extend(active.parent.children)
        nodes.extend(active.children)
        nodes.extend(self.root.children)
        seen = set()
        out = []
        for n in nodes:
            if n.id in seen:
                continue
            seen.add(n.id)
            out.append(n)
        return out

    def _verify(self, chunk: str, best_score: float) -> str:
        if not self.use_qwen:
            # nomic only: don't invent SUB without Qwen
            return "NEW" if best_score < LOW else "SAME"
        parent = self.active.parent or self.root
        siblings = [c.title for c in parent.children if c is not self.active]
        sibling_txt = ", ".join(siblings) if siblings else "(none)"
        prompt = (
            "You are labeling one new sentence in a meeting.\n"
            "Reply with ONLY one word: SAME, SUB, or NEW.\n"
            "SAME = continues the current topic.\n"
            "SUB = a more specific sub-topic of the current topic.\n"
            "NEW = a different topic at the same level.\n\n"
            f"Current topic: {self.active.topic_text() or '(start of meeting)'}\n"
            f"Other topics at this level: {sibling_txt}\n"
            f"New sentence: {chunk}\n\n"
            "Answer:"
        )
        raw = self._qwen(prompt, max_tokens=8).upper()
        for word in ("SAME", "SUB", "NEW"):
            if word in raw:
                return word
        return "SAME"

    def _new_node(self, parent: SummaryNode) -> SummaryNode:
        if parent.depth >= MAX_DEPTH:
            return parent
        node = SummaryNode(
            title=f"topic {len(parent.children) + 1}",
            parent=parent,
            depth=parent.depth + 1,
        )
        parent.children.append(node)
        return node

    # ------------------------------------------------------------ per chunk

    def insert_chunk(self, chunk: str, hint=None) -> SummaryNode:
        chunk_embedding = self._embed(chunk)

        node = self.decide_placement(chunk_embedding, chunk, hint=hint)

        if node.parent is not None and not node.summary and not node.raw_chunks:
            node.title = " ".join(chunk.split()[:5])

        node.raw_chunks.append(chunk)
        self._set_active(node)
        if self.use_embed:
            node.embedding = self._embed(node.topic_text())

        self.maybe_update_node_summary(node)
        return node

    def _set_active(self, node: SummaryNode) -> None:
        self.active.status = "stale"
        node.status = "active"
        self.active = node

    def maybe_update_node_summary(self, node: SummaryNode) -> bool:
        if not node.raw_chunks:
            return False

        big_enough = node.word_count() >= self.word_threshold
        old_enough = (time.time() - node.updated_at) >= self.time_threshold
        if not (big_enough or old_enough):
            return False

        new_text = " ".join(node.raw_chunks)
        prompt = (
            "Update the running summary of one topic in a meeting. "
            "Keep it to 1-2 short sentences. Do not mention other topics.\n---\n"
            f"Previous summary: {node.summary or '(none)'}\n"
            f"New transcript: {new_text}"
        )
        node.summary = self._qwen(prompt, max_tokens=80)
        node.raw_chunks.clear()
        node.updated_at = time.time()
        if self.use_embed:
            node.embedding = self._embed(node.summary)

        node.dirty = True
        self.mark_dirty_ancestors(node)
        return True

    def mark_dirty_ancestors(self, node: SummaryNode) -> None:
        parent = node.parent
        while parent is not None:
            parent.dirty = True
            parent = parent.parent

    # ------------------------------------------------------------ rollups

    def rollup_dirty_nodes(self) -> int:
        dirty = [n for n in self._all_nodes() if n.dirty]
        dirty.sort(key=lambda n: n.depth, reverse=True)
        for node in dirty:
            self._rollup(node)
        return len(dirty)

    def get_subtree_summary(self, node: SummaryNode) -> str:
        for child in node.children:
            if child.dirty:
                self.get_subtree_summary(child)
        if node.dirty:
            self._rollup(node)
        return node.subtree_summary

    def _rollup(self, node: SummaryNode) -> None:
        if not node.children:
            node.subtree_summary = node.summary
            node.dirty = False
            return

        child_text = "\n".join(
            f"- {c.title}: {c.subtree_summary or c.summary or '(empty)'}"
            for c in node.children
        )
        prompt = (
            "Merge a topic summary with its sub-topic summaries into one "
            "short paragraph.\n---\n"
            f"Topic summary: {node.summary or '(none)'}\n"
            f"Sub-topics:\n{child_text}"
        )
        node.subtree_summary = self._qwen(prompt, max_tokens=100)
        node.dirty = False

    # ------------------------------------------------------------ helpers

    def _all_nodes(self, node: Optional[SummaryNode] = None) -> list[SummaryNode]:
        node = node or self.root
        nodes = [node]
        for child in node.children:
            nodes.extend(self._all_nodes(child))
        return nodes

    def flush_all(self) -> None:
        saved = self.word_threshold
        self.word_threshold = 0
        for node in self._all_nodes():
            self.maybe_update_node_summary(node)
        self.word_threshold = saved

    def outline(self, node: Optional[SummaryNode] = None) -> str:
        node = node or self.root
        lines = []
        pad = "  " * node.depth
        mark = "*" if node.status == "active" else " "
        text = node.summary or "(no summary yet)"
        if len(text) > 90:
            text = text[:90] + "..."
        pending = node.word_count()
        extra = f" [+{pending} words buffered]" if pending else ""
        flag = " [dirty]" if node.dirty else ""
        lines.append(f"{mark} {pad}{node.title} (d{node.depth}): {text}{extra}{flag}")
        for child in node.children:
            lines.append(self.outline(child))
        return "\n".join(lines)
