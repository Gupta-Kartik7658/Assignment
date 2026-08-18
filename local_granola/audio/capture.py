from __future__ import annotations

import json
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import soundcard as sc

from local_granola.config import (
    BLOCKSIZE,
    OUTPUT_DIR,
    SAMPLE_RATE,
    TRANSCRIPTION_MIX_DOMINANCE_RATIO,
    TRANSCRIPTION_MIX_SILENCE_RMS,
    TRANSCRIPTION_MIX_WINDOW_SECONDS,
)

from .devices import resolve_default_loopback_microphone


class CaptureMode(str, Enum):
    MICROPHONE = "Microphone"
    SPEAKERS = "Speakers"
    BOTH = "Both"


@dataclass(frozen=True)
class ChunkEvent:
    source: str
    frames: int
    peak: float
    rms: float
    start_seconds: float
    end_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class CaptureResult:
    output_directory: Path
    metadata_path: Path
    timeline_path: Path
    microphone_wav: Path | None
    speaker_wav: Path | None
    mixed_wav: Path | None
    duration_seconds: float
    sample_rate: int
    mode: CaptureMode


@dataclass(frozen=True)
class RecordedChunk:
    source: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    peak: float
    rms: float
    audio: np.ndarray


@dataclass(frozen=True)
class AudioWindowSnapshot:
    source: str
    audio: np.ndarray
    window_start_seconds: float
    total_seconds: float


class _RecorderThread(threading.Thread):
    def __init__(
        self,
        *,
        label: str,
        device,
        on_chunk: Callable[[str, np.ndarray, ChunkEvent], None],
        start_time: float,
        sample_rate: int,
        chunk_frames: int,
        blocksize: int,
    ) -> None:
        super().__init__(daemon=True, name=f"{label}-recorder")
        self.label = label
        self.device = device
        self.on_chunk = on_chunk
        self.start_time = start_time
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.blocksize = blocksize
        self.stop_event = threading.Event()
        self.error: Exception | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        channels = max(1, min(getattr(self.device, "channels", 2) or 2, 2))
        frame_cursor = 0

        try:
            with self.device.recorder(
                samplerate=self.sample_rate,
                channels=channels,
                blocksize=self.blocksize,
            ) as recorder:
                while not self.stop_event.is_set():
                    frames = recorder.record(numframes=self.chunk_frames)
                    if frames.size == 0:
                        continue

                    mono = _to_mono(frames)
                    frame_count = int(mono.shape[0])
                    start_frame = frame_cursor
                    end_frame = frame_cursor + frame_count
                    frame_cursor = end_frame
                    event = ChunkEvent(
                        source=self.label,
                        frames=frame_count,
                        peak=float(np.max(np.abs(mono))) if mono.size else 0.0,
                        rms=float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0,
                        start_seconds=start_frame / self.sample_rate,
                        end_seconds=end_frame / self.sample_rate,
                        elapsed_seconds=max(0.0, time.perf_counter() - self.start_time),
                    )
                    self.on_chunk(self.label, mono, event)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.error = exc


class AudioCaptureService:
    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_frames: int = BLOCKSIZE,
        blocksize: int = BLOCKSIZE,
        output_directory: Path = OUTPUT_DIR,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.blocksize = blocksize
        self.output_directory = output_directory
        self.listeners: list[Callable[[ChunkEvent], None]] = []
        self._lock = threading.Lock()
        self._recorders: list[_RecorderThread] = []
        self._chunks: dict[str, list[RecordedChunk]] = {"microphone": [], "speakers": []}
        self._mode = CaptureMode.MICROPHONE
        self._started_at = 0.0
        self._device_labels: dict[str, str] = {}

    @property
    def is_recording(self) -> bool:
        return bool(self._recorders)

    @property
    def elapsed_seconds(self) -> float:
        if not self.is_recording or self._started_at == 0.0:
            return 0.0
        return max(0.0, time.perf_counter() - self._started_at)

    @property
    def active_sources(self) -> tuple[str, ...]:
        if self._mode == CaptureMode.MICROPHONE:
            return ("microphone",)
        if self._mode == CaptureMode.SPEAKERS:
            return ("speakers",)
        return ("microphone", "speakers")

    def add_listener(self, listener: Callable[[ChunkEvent], None]) -> None:
        self.listeners.append(listener)

    def get_available_sources(self) -> list[str]:
        with self._lock:
            return [source for source, chunks in self._chunks.items() if chunks]

    def get_source_audio_snapshot(self, source: str, *, normalized: bool = True) -> np.ndarray:
        with self._lock:
            chunks = list(self._chunks.get(source, []))

        duration_frames = _duration_frames(chunks)
        audio = _render_timeline_audio_window(chunks, start_frame=0, end_frame=duration_frames)
        if not normalized:
            return audio
        return _normalize_transcription_track(audio)

    def get_source_transcription_window(
        self,
        source: str,
        *,
        window_seconds: float,
        normalized: bool = True,
    ) -> AudioWindowSnapshot:
        with self._lock:
            chunks = list(self._chunks.get(source, []))

        duration_frames = _duration_frames(chunks)
        if duration_frames <= 0:
            return AudioWindowSnapshot(
                source=source,
                audio=np.array([], dtype=np.float32),
                window_start_seconds=0.0,
                total_seconds=0.0,
            )

        window_frames = max(1, int(round(window_seconds * self.sample_rate)))
        start_frame = max(0, duration_frames - window_frames)
        audio = _render_timeline_audio_window(chunks, start_frame=start_frame, end_frame=duration_frames)
        if normalized:
            audio = _normalize_transcription_track(audio)

        return AudioWindowSnapshot(
            source=source,
            audio=audio,
            window_start_seconds=start_frame / self.sample_rate,
            total_seconds=duration_frames / self.sample_rate,
        )

    def get_mixed_audio_snapshot(self) -> np.ndarray:
        with self._lock:
            microphone_chunks = list(self._chunks["microphone"])
            speaker_chunks = list(self._chunks["speakers"])

        duration_frames = _duration_frames(microphone_chunks, speaker_chunks)
        microphone_audio = _render_timeline_audio(microphone_chunks, duration_frames)
        speaker_audio = _render_timeline_audio(speaker_chunks, duration_frames)
        return _mix_audio(microphone_audio, speaker_audio, self.sample_rate)

    def start(self, mode: CaptureMode) -> None:
        if self.is_recording:
            raise RuntimeError("Audio capture is already running.")

        self._mode = mode
        self._started_at = time.perf_counter()
        self._chunks = {"microphone": [], "speakers": []}
        self._recorders = []
        self._device_labels = {}

        if mode in {CaptureMode.MICROPHONE, CaptureMode.BOTH}:
            microphone = sc.default_microphone()
            self._device_labels["microphone"] = str(microphone)
            self._recorders.append(
                _RecorderThread(
                    label="microphone",
                    device=microphone,
                    on_chunk=self._on_chunk,
                    start_time=self._started_at,
                    sample_rate=self.sample_rate,
                    chunk_frames=self.chunk_frames,
                    blocksize=self.blocksize,
                )
            )

        if mode in {CaptureMode.SPEAKERS, CaptureMode.BOTH}:
            speakers = resolve_default_loopback_microphone()
            self._device_labels["speakers"] = str(speakers)
            self._recorders.append(
                _RecorderThread(
                    label="speakers",
                    device=speakers,
                    on_chunk=self._on_chunk,
                    start_time=self._started_at,
                    sample_rate=self.sample_rate,
                    chunk_frames=self.chunk_frames,
                    blocksize=self.blocksize,
                )
            )

        for recorder in self._recorders:
            recorder.start()

    def stop(self) -> CaptureResult:
        if not self.is_recording:
            raise RuntimeError("Audio capture is not running.")

        recorders = list(self._recorders)
        self._recorders = []

        for recorder in recorders:
            recorder.stop()
        for recorder in recorders:
            recorder.join(timeout=3)
            if recorder.error:
                raise recorder.error

        with self._lock:
            microphone_chunks = list(self._chunks["microphone"])
            speaker_chunks = list(self._chunks["speakers"])

        duration_frames = _duration_frames(microphone_chunks, speaker_chunks)
        duration_seconds = max(
            max(0.0, time.perf_counter() - self._started_at),
            duration_frames / self.sample_rate if duration_frames else 0.0,
        )

        microphone_audio = _render_timeline_audio(microphone_chunks, duration_frames)
        speaker_audio = _render_timeline_audio(speaker_chunks, duration_frames)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_directory = self.output_directory / f"capture_{timestamp}"
        capture_directory.mkdir(parents=True, exist_ok=True)

        microphone_wav = None
        speaker_wav = None
        mixed_wav = None

        if microphone_audio.size:
            microphone_wav = capture_directory / "microphone.wav"
            _write_mono_wav(microphone_wav, microphone_audio, self.sample_rate)

        if speaker_audio.size:
            speaker_wav = capture_directory / "speakers.wav"
            _write_mono_wav(speaker_wav, speaker_audio, self.sample_rate)

        mixed_audio = _mix_audio(microphone_audio, speaker_audio, self.sample_rate)
        if mixed_audio.size:
            mixed_wav = capture_directory / "mixed.wav"
            _write_mono_wav(mixed_wav, mixed_audio, self.sample_rate)

        timeline_path = capture_directory / "timeline.json"
        timeline = {
            "mode": self._mode.value,
            "sample_rate": self.sample_rate,
            "transcription_audio_path": str(mixed_wav) if mixed_wav else None,
            "mix_strategy": "dominant-source-windowed",
            "sources": {
                "microphone": _serialize_chunks(microphone_chunks),
                "speakers": _serialize_chunks(speaker_chunks),
            },
        }
        timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")

        metadata_path = capture_directory / "capture.json"
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": self._mode.value,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(duration_seconds, 3),
            "devices": self._device_labels,
            "transcription_audio_path": str(mixed_wav) if mixed_wav else None,
            "files": {
                "microphone_wav": str(microphone_wav) if microphone_wav else None,
                "speaker_wav": str(speaker_wav) if speaker_wav else None,
                "mixed_wav": str(mixed_wav) if mixed_wav else None,
                "timeline_path": str(timeline_path),
            },
            "frame_counts": {
                "microphone": int(microphone_audio.shape[0]),
                "speakers": int(speaker_audio.shape[0]),
                "mixed": int(mixed_audio.shape[0]),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return CaptureResult(
            output_directory=capture_directory,
            metadata_path=metadata_path,
            timeline_path=timeline_path,
            microphone_wav=microphone_wav,
            speaker_wav=speaker_wav,
            mixed_wav=mixed_wav,
            duration_seconds=duration_seconds,
            sample_rate=self.sample_rate,
            mode=self._mode,
        )

    def _on_chunk(self, source: str, mono: np.ndarray, event: ChunkEvent) -> None:
        with self._lock:
            start_frame = int(round(event.start_seconds * self.sample_rate))
            end_frame = start_frame + mono.shape[0]
            self._chunks[source].append(
                RecordedChunk(
                    source=source,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_seconds=event.start_seconds,
                    end_seconds=event.end_seconds,
                    peak=event.peak,
                    rms=event.rms,
                    audio=mono.copy(),
                )
            )

        for listener in self.listeners:
            listener(event)


def _to_mono(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return frames.astype(np.float32, copy=False)
    return frames.mean(axis=1).astype(np.float32, copy=False)


def _duration_frames(*groups: list[RecordedChunk]) -> int:
    return max((chunk.end_frame for group in groups for chunk in group), default=0)


def _render_timeline_audio(chunks: list[RecordedChunk], duration_frames: int) -> np.ndarray:
    return _render_timeline_audio_window(chunks, start_frame=0, end_frame=duration_frames)


def _render_timeline_audio_window(
    chunks: list[RecordedChunk],
    *,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)

    duration_frames = max(0, end_frame - start_frame)
    if duration_frames <= 0:
        return np.array([], dtype=np.float32)

    timeline = np.zeros(duration_frames, dtype=np.float32)
    counts = np.zeros(duration_frames, dtype=np.float32)

    for chunk in chunks:
        chunk_start = max(start_frame, chunk.start_frame)
        chunk_end = min(end_frame, chunk.end_frame)
        if chunk_end <= chunk_start:
            continue

        target_start = chunk_start - start_frame
        target_end = chunk_end - start_frame
        source_start = chunk_start - chunk.start_frame
        usable_frames = chunk_end - chunk_start
        timeline[target_start:target_end] += chunk.audio[source_start : source_start + usable_frames]
        counts[target_start:target_end] += 1.0

    counts[counts == 0.0] = 1.0
    return (timeline / counts).astype(np.float32, copy=False)


def _mix_audio(microphone_audio: np.ndarray, speaker_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if microphone_audio.size == 0 and speaker_audio.size == 0:
        return np.array([], dtype=np.float32)
    if microphone_audio.size == 0:
        return _normalize_transcription_track(speaker_audio)
    if speaker_audio.size == 0:
        return _normalize_transcription_track(microphone_audio)

    target = max(microphone_audio.shape[0], speaker_audio.shape[0])
    mic = np.pad(microphone_audio, (0, target - microphone_audio.shape[0]))
    speakers = np.pad(speaker_audio, (0, target - speaker_audio.shape[0]))
    mic = _normalize_transcription_track(mic)
    speakers = _normalize_transcription_track(speakers)
    return _windowed_dominant_mix(mic, speakers, sample_rate)


def _normalize_transcription_track(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return np.array([], dtype=np.float32)

    magnitude = np.abs(audio)
    active = magnitude > 0.01
    if not np.any(active):
        return audio.astype(np.float32, copy=True)

    rms = float(np.sqrt(np.mean(np.square(audio[active]))))
    if rms <= 1e-6:
        return audio.astype(np.float32, copy=True)

    gain = min(max(0.12 / rms, 0.75), 2.5)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32, copy=False)


def _windowed_dominant_mix(
    microphone_audio: np.ndarray,
    speaker_audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    if microphone_audio.size != speaker_audio.size:
        raise ValueError("Tracks must be equal length before mixing.")

    mixed = np.zeros_like(microphone_audio, dtype=np.float32)
    window_size = max(1, int(sample_rate * TRANSCRIPTION_MIX_WINDOW_SECONDS))

    for start in range(0, microphone_audio.shape[0], window_size):
        end = min(microphone_audio.shape[0], start + window_size)
        microphone_chunk = microphone_audio[start:end]
        speaker_chunk = speaker_audio[start:end]
        microphone_rms = _rms(microphone_chunk)
        speaker_rms = _rms(speaker_chunk)

        if (
            microphone_rms < TRANSCRIPTION_MIX_SILENCE_RMS
            and speaker_rms < TRANSCRIPTION_MIX_SILENCE_RMS
        ):
            mixed[start:end] = 0.0
            continue

        if microphone_rms >= speaker_rms * TRANSCRIPTION_MIX_DOMINANCE_RATIO:
            mixed[start:end] = microphone_chunk
            continue

        if speaker_rms >= microphone_rms * TRANSCRIPTION_MIX_DOMINANCE_RATIO:
            mixed[start:end] = speaker_chunk
            continue

        mixed[start:end] = microphone_chunk if microphone_rms >= speaker_rms else speaker_chunk

    return np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _serialize_chunks(chunks: list[RecordedChunk]) -> list[dict[str, float | int | str]]:
    return [
        {
            "source": chunk.source,
            "start_frame": chunk.start_frame,
            "end_frame": chunk.end_frame,
            "start_seconds": round(chunk.start_seconds, 3),
            "end_seconds": round(chunk.end_seconds, 3),
            "peak": round(chunk.peak, 4),
            "rms": round(chunk.rms, 4),
        }
        for chunk in chunks
    ]


def _write_mono_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
