from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from local_granola.config import DEFAULT_CHUNK_WINDOW_SECONDS

from .chunker import build_chunks, save_chunk_text, save_chunks_json
from .transcript import TranscriptDocument


@dataclass(frozen=True)
class TranscriptArtifacts:
    transcript_json_path: Path
    transcript_text_path: Path
    chunks_json_path: Path
    chunks_text_path: Path
    chunk_count: int


def generate_transcript_artifacts(
    *,
    segments: Iterable[object],
    output_directory: Path,
    source_audio_path: str | None = None,
    language: str | None = None,
    window_seconds: float = DEFAULT_CHUNK_WINDOW_SECONDS,
) -> TranscriptArtifacts:
    output_directory.mkdir(parents=True, exist_ok=True)

    transcript = TranscriptDocument.from_whisper_segments(
        segments,
        source_audio_path=source_audio_path,
        language=language,
    )
    chunks = build_chunks(transcript, window_seconds=window_seconds)

    transcript_json_path = transcript.save_json(output_directory / "transcript.json")
    transcript_text_path = transcript.save_text(output_directory / "transcript.txt")
    chunks_json_path = save_chunks_json(chunks, output_directory / "chunks.json")
    chunks_text_path = save_chunk_text(chunks, output_directory / "chunks.txt")

    return TranscriptArtifacts(
        transcript_json_path=transcript_json_path,
        transcript_text_path=transcript_text_path,
        chunks_json_path=chunks_json_path,
        chunks_text_path=chunks_text_path,
        chunk_count=len(chunks),
    )


def generate_transcript_artifacts_from_document(
    *,
    transcript: TranscriptDocument,
    output_directory: Path,
    window_seconds: float = DEFAULT_CHUNK_WINDOW_SECONDS,
) -> TranscriptArtifacts:
    output_directory.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(transcript, window_seconds=window_seconds)
    transcript_json_path = transcript.save_json(output_directory / "transcript.json")
    transcript_text_path = transcript.save_text(output_directory / "transcript.txt")
    chunks_json_path = save_chunks_json(chunks, output_directory / "chunks.json")
    chunks_text_path = save_chunk_text(chunks, output_directory / "chunks.txt")

    return TranscriptArtifacts(
        transcript_json_path=transcript_json_path,
        transcript_text_path=transcript_text_path,
        chunks_json_path=chunks_json_path,
        chunks_text_path=chunks_text_path,
        chunk_count=len(chunks),
    )
