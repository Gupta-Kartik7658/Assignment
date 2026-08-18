from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str
    source: str = "mixed"

    def to_text_line(self) -> str:
        source_label = _source_label(self.source)
        source_prefix = f"[{source_label}] " if source_label else ""
        return (
            f"[{format_timestamp(self.start_seconds)} - {format_timestamp(self.end_seconds)}] "
            f"{source_prefix}{self.text}"
        )


@dataclass
class TranscriptDocument:
    segments: list[TranscriptSegment] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source_audio_path: str | None = None
    language: str | None = None

    @property
    def duration_seconds(self) -> float:
        return max((segment.end_seconds for segment in self.segments), default=0.0)

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.segments.append(segment)
        self.segments.sort(key=lambda item: (item.start_seconds, item.end_seconds))

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "source_audio_path": self.source_audio_path,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 3),
            "segments": [asdict(segment) for segment in self.segments],
        }

    def save_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def save_text(self, path: Path) -> Path:
        lines = [
            "# Transcript",
            "",
            f"Created: {self.created_at}",
            f"Source audio: {self.source_audio_path or 'N/A'}",
            f"Language: {self.language or 'unknown'}",
            "",
        ]
        lines.extend(segment.to_text_line() for segment in self.segments)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_whisper_segments(
        cls,
        segments: Iterable[object],
        *,
        source_audio_path: str | None = None,
        language: str | None = None,
    ) -> "TranscriptDocument":
        document = cls(source_audio_path=source_audio_path, language=language)

        for segment in segments:
            start_seconds = _read_segment_value(segment, "start")
            end_seconds = _read_segment_value(segment, "end")
            text = _read_segment_text(segment)
            document.add_segment(
                TranscriptSegment(
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                    text=text.strip(),
                )
            )

        return document


def format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    minutes, milliseconds = divmod(total_milliseconds, 60_000)
    seconds_part, milliseconds = divmod(milliseconds, 1000)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def _read_segment_value(segment: object, name: str) -> float:
    if isinstance(segment, dict):
        return float(segment[name])
    return float(getattr(segment, name))


def _read_segment_text(segment: object) -> str:
    if isinstance(segment, dict):
        return str(segment["text"])
    return str(getattr(segment, "text"))


def _source_label(source: str) -> str:
    if source == "microphone":
        return "mic"
    if source == "speakers":
        return "speaker"
    if source == "merged" or source == "mixed":
        return ""
    return source
