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

from local_granola.config import BLOCKSIZE, OUTPUT_DIR, SAMPLE_RATE

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
    elapsed_seconds: float


@dataclass(frozen=True)
class CaptureResult:
    output_directory: Path
    metadata_path: Path
    microphone_wav: Path | None
    speaker_wav: Path | None
    mixed_wav: Path | None
    duration_seconds: float
    sample_rate: int
    mode: CaptureMode


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
                    event = ChunkEvent(
                        source=self.label,
                        frames=int(mono.shape[0]),
                        peak=float(np.max(np.abs(mono))) if mono.size else 0.0,
                        rms=float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0,
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
        self._chunks: dict[str, list[np.ndarray]] = {"microphone": [], "speakers": []}
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

    def add_listener(self, listener: Callable[[ChunkEvent], None]) -> None:
        self.listeners.append(listener)

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

        duration_seconds = max(0.0, time.perf_counter() - self._started_at)

        with self._lock:
            microphone_audio = _concat_chunks(self._chunks["microphone"])
            speaker_audio = _concat_chunks(self._chunks["speakers"])

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

        mixed_audio = _mix_audio(microphone_audio, speaker_audio)
        if mixed_audio.size:
            mixed_wav = capture_directory / "mixed.wav"
            _write_mono_wav(mixed_wav, mixed_audio, self.sample_rate)

        metadata_path = capture_directory / "capture.json"
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": self._mode.value,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(duration_seconds, 3),
            "devices": self._device_labels,
            "files": {
                "microphone_wav": str(microphone_wav) if microphone_wav else None,
                "speaker_wav": str(speaker_wav) if speaker_wav else None,
                "mixed_wav": str(mixed_wav) if mixed_wav else None,
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
            microphone_wav=microphone_wav,
            speaker_wav=speaker_wav,
            mixed_wav=mixed_wav,
            duration_seconds=duration_seconds,
            sample_rate=self.sample_rate,
            mode=self._mode,
        )

    def _on_chunk(self, source: str, mono: np.ndarray, event: ChunkEvent) -> None:
        with self._lock:
            self._chunks[source].append(mono.copy())

        for listener in self.listeners:
            listener(event)


def _to_mono(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return frames.astype(np.float32, copy=False)
    return frames.mean(axis=1).astype(np.float32, copy=False)


def _concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _mix_audio(microphone_audio: np.ndarray, speaker_audio: np.ndarray) -> np.ndarray:
    if microphone_audio.size == 0 and speaker_audio.size == 0:
        return np.array([], dtype=np.float32)
    if microphone_audio.size == 0:
        return speaker_audio.copy()
    if speaker_audio.size == 0:
        return microphone_audio.copy()

    target = max(microphone_audio.shape[0], speaker_audio.shape[0])
    mic = np.pad(microphone_audio, (0, target - microphone_audio.shape[0]))
    speakers = np.pad(speaker_audio, (0, target - speaker_audio.shape[0]))
    return np.clip((mic + speakers) * 0.5, -1.0, 1.0).astype(np.float32, copy=False)


def _write_mono_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
