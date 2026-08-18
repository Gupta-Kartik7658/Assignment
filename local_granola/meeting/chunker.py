from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from local_granola.config import DEFAULT_CHUNK_WINDOW_SECONDS

from .transcript import TranscriptDocument, TranscriptSegment, format_timestamp


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    start_seconds: float
    end_seconds: float
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(segment.to_text_line() for segment in self.segments)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "text": self.text,
            "segments": [
                {
                    "start_seconds": round(segment.start_seconds, 3),
                    "end_seconds": round(segment.end_seconds, 3),
                    "text": segment.text,
                    "source": segment.source,
                }
                for segment in self.segments
            ],
        }


def build_chunks(
    transcript: TranscriptDocument,
    *,
    window_seconds: float = DEFAULT_CHUNK_WINDOW_SECONDS,
    include_empty: bool = False,
) -> list[TranscriptChunk]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if not transcript.segments:
        return []

    total_windows = max(1, math.ceil(transcript.duration_seconds / window_seconds))
    buckets: list[list[TranscriptSegment]] = [[] for _ in range(total_windows)]

    for segment in transcript.segments:
        bucket_index = min(int(segment.start_seconds // window_seconds), total_windows - 1)
        buckets[bucket_index].append(segment)

    chunks: list[TranscriptChunk] = []
    for index, segments in enumerate(buckets, start=1):
        if not segments and not include_empty:
            continue
        chunk_start = (index - 1) * window_seconds
        chunk_end = min(index * window_seconds, transcript.duration_seconds)
        chunks.append(
            TranscriptChunk(
                index=index,
                start_seconds=chunk_start,
                end_seconds=chunk_end,
                segments=segments,
            )
        )

    return chunks


def save_chunks_json(chunks: list[TranscriptChunk], path: Path) -> Path:
    payload = {
        "chunk_count": len(chunks),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_chunk_text(chunks: list[TranscriptChunk], path: Path) -> Path:
    lines = ["# Transcript Chunks", ""]
    for chunk in chunks:
        lines.append(
            f"## Chunk {chunk.index} ({format_timestamp(chunk.start_seconds)} - {format_timestamp(chunk.end_seconds)})"
        )
        lines.append("")
        if chunk.segments:
            lines.extend(segment.to_text_line() for segment in chunk.segments)
        else:
            lines.append("(empty)")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
