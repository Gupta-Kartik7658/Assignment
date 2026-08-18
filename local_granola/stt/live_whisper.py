from __future__ import annotations

import multiprocessing as mp
import queue as pyqueue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from local_granola.audio import AudioCaptureService, CaptureResult
from local_granola.config import (
    DEFAULT_CHUNK_WINDOW_SECONDS,
    LIVE_TRANSCRIBE_ACCEPT_OVERLAP_SECONDS,
    LIVE_TRANSCRIBE_BEAM_SIZE,
    LIVE_TRANSCRIBE_BEST_OF,
    LIVE_TRANSCRIBE_MIN_AUDIO_SECONDS,
    LIVE_TRANSCRIBE_POLL_SECONDS,
    LIVE_TRANSCRIBE_PREVIEW_LOOKBACK_SECONDS,
    LIVE_TRANSCRIBE_STABILIZATION_SECONDS,
    LIVE_TRANSCRIBE_STEP_SECONDS,
    LIVE_TRANSCRIBE_VAD_FILTER,
    LIVE_TRANSCRIBE_WINDOW_SECONDS,
    MODELS_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_DOWNLOAD_ROOT,
    WHISPER_LANGUAGE,
    WHISPER_LOCAL_MODEL_DIR,
    WHISPER_MODEL_CANDIDATES,
    WHISPER_MODEL_NAME,
)
from local_granola.meeting import (
    TranscriptDocument,
    TranscriptSegment,
    generate_transcript_artifacts_from_document,
)

try:  # pragma: no cover - import availability depends on environment
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - import availability depends on environment
    WhisperModel = None  # type: ignore[assignment]


TRANSCRIPT_SOURCES = ("microphone", "speakers")
PREVIEW_LOOKBACK_SECONDS = LIVE_TRANSCRIBE_PREVIEW_LOOKBACK_SECONDS


@dataclass(frozen=True)
class TranscriptUpdate:
    segment: TranscriptSegment
    is_preview: bool = False


@dataclass(frozen=True)
class TranscriptArtifactsResult:
    transcript_json_path: Path
    transcript_text_path: Path
    chunks_json_path: Path
    chunks_text_path: Path
    chunk_count: int
    source_json_paths: dict[str, Path] = field(default_factory=dict)
    source_text_paths: dict[str, Path] = field(default_factory=dict)


def _stt_worker_main(
    source: str,
    model_reference: str,
    model_label: str,
    command_queue,
    result_queue,
) -> None:
    if WhisperModel is None:
        result_queue.put({"type": "error", "source": source, "error": "faster-whisper is not installed"})
        return

    try:
        model = WhisperModel(
            model_reference,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=str(WHISPER_DOWNLOAD_ROOT),
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        result_queue.put({"type": "error", "source": source, "error": str(exc)})
        return

    result_queue.put({"type": "ready", "source": source, "model_label": model_label})

    while True:
        command = command_queue.get()
        if command["type"] == "stop":
            return
        if command["type"] != "decode":
            continue

        try:
            segments, info = model.transcribe(
                command["audio"],
                language=WHISPER_LANGUAGE,
                task="transcribe",
                beam_size=LIVE_TRANSCRIBE_BEAM_SIZE,
                best_of=LIVE_TRANSCRIBE_BEST_OF,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=LIVE_TRANSCRIBE_VAD_FILTER,
                word_timestamps=False,
            )
            result_queue.put(
                {
                    "type": "decoded",
                    "source": source,
                    "request_id": command["request_id"],
                    "total_seconds": command["total_seconds"],
                    "window_start": command["window_start"],
                    "language": getattr(info, "language", WHISPER_LANGUAGE),
                    "segments": [
                        {
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "text": str(segment.text).strip(),
                        }
                        for segment in segments
                        if str(segment.text).strip()
                    ],
                }
            )
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            result_queue.put(
                {
                    "type": "error",
                    "source": source,
                    "error": str(exc),
                }
            )


class LiveTranscriptionService:
    def __init__(self, capture_service: AudioCaptureService) -> None:
        self.capture_service = capture_service
        self.sample_rate = capture_service.sample_rate
        self._segment_listeners: list[Callable[[TranscriptUpdate], None]] = []
        self._status_listeners: list[Callable[[str], None]] = []
        self._lock = threading.Lock()
        self._manager_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._mp_context = mp.get_context("spawn")
        self._result_queue = None
        self._command_queues: dict[str, object] = {}
        self._worker_processes: dict[str, object] = {}
        self._model_label = WHISPER_MODEL_NAME
        self._model_reference = WHISPER_MODEL_NAME
        self._language: str | None = WHISPER_LANGUAGE
        self._source_transcripts: dict[str, TranscriptDocument] = {}
        self._merged_transcript = TranscriptDocument()
        self._source_committed_until: dict[str, float] = {}
        self._source_last_dispatched_total: dict[str, float] = {}
        self._source_inflight: dict[str, bool] = {}
        self._source_request_id: dict[str, int] = {}
        self._source_preview_signature: dict[str, tuple[float, float, str] | None] = {}
        self._ready_sources: set[str] = set()
        self._reset_state()

    def add_segment_listener(self, listener: Callable[[TranscriptUpdate], None]) -> None:
        self._segment_listeners.append(listener)

    def add_status_listener(self, listener: Callable[[str], None]) -> None:
        self._status_listeners.append(listener)

    def start(self) -> None:
        if self._manager_thread and self._manager_thread.is_alive():
            return

        self._stop_event.clear()
        self._reset_state()
        self._notify_status("Loading local Whisper model...")
        self._model_reference, self._model_label = self._resolve_model_reference()
        self._result_queue = self._mp_context.Queue()

        for source in self.capture_service.active_sources:
            command_queue = self._mp_context.Queue()
            process = self._mp_context.Process(
                target=_stt_worker_main,
                args=(source, self._model_reference, self._model_label, command_queue, self._result_queue),
                daemon=True,
            )
            process.start()
            self._command_queues[source] = command_queue
            self._worker_processes[source] = process

        self._manager_thread = threading.Thread(target=self._run_manager, daemon=True, name="stt-manager")
        self._manager_thread.start()

    def stop(self, capture_result: CaptureResult | None = None) -> TranscriptArtifactsResult | None:
        self._stop_event.set()
        if self._manager_thread:
            self._manager_thread.join(timeout=10)
            self._manager_thread = None

        self._shutdown_workers()

        if capture_result is None:
            return None

        return self._build_final_artifacts(capture_result)

    def _run_manager(self) -> None:
        while not self._stop_event.is_set():
            self._drain_results(max_messages=32)
            self._dispatch_jobs()
            time.sleep(LIVE_TRANSCRIBE_POLL_SECONDS)

        self._drain_results(max_messages=256)

    def _dispatch_jobs(self) -> None:
        for source in self.capture_service.active_sources:
            self._dispatch_job_for_source(source)

    def _dispatch_job_for_source(self, source: str) -> bool:
        if source not in self._command_queues:
            return False
        if self._source_inflight[source]:
            return False

        snapshot = self.capture_service.get_source_transcription_window(
            source,
            window_seconds=LIVE_TRANSCRIBE_WINDOW_SECONDS,
            normalized=True,
        )
        if snapshot.audio.size == 0:
            return False

        total_seconds = snapshot.total_seconds
        if total_seconds < LIVE_TRANSCRIBE_MIN_AUDIO_SECONDS:
            return False

        last_dispatched_total = self._source_last_dispatched_total[source]
        if total_seconds - last_dispatched_total < LIVE_TRANSCRIBE_STEP_SECONDS:
            return False

        request_id = self._source_request_id[source] + 1
        self._source_request_id[source] = request_id
        self._source_last_dispatched_total[source] = total_seconds
        self._source_inflight[source] = True

        self._command_queues[source].put(
            {
                "type": "decode",
                "request_id": request_id,
                "window_start": snapshot.window_start_seconds,
                "total_seconds": total_seconds,
                "audio": snapshot.audio,
            }
        )
        return True

    def _drain_results(self, *, max_messages: int) -> None:
        if self._result_queue is None:
            return

        for _ in range(max_messages):
            try:
                message = self._result_queue.get_nowait()
            except pyqueue.Empty:
                break

            message_type = message.get("type")
            if message_type == "ready":
                self._handle_ready(message)
            elif message_type == "decoded":
                self._handle_decoded(message)
            elif message_type == "error":
                self._handle_error(message)

    def _handle_ready(self, message: dict) -> None:
        source = message["source"]
        with self._lock:
            self._ready_sources.add(source)
            ready_sources = len(self._ready_sources)
            total_sources = len(self.capture_service.active_sources)

        if ready_sources < total_sources:
            self._notify_status(
                f"Whisper ready for {source}. Model: {self._model_label}. Waiting for other sources..."
            )
        else:
            self._notify_status(
                f"Whisper ready. Parallel low-latency transcription is on. "
                f"Language locked to {WHISPER_LANGUAGE}. Model: {self._model_label}."
            )
        self._dispatch_job_for_source(source)

    def _handle_decoded(self, message: dict) -> None:
        source = message["source"]
        self._source_inflight[source] = False
        self._language = WHISPER_LANGUAGE or message.get("language")

        total_seconds = float(message["total_seconds"])
        stable_cutoff = total_seconds - LIVE_TRANSCRIBE_STABILIZATION_SECONDS
        window_start = float(message["window_start"])
        decoded_segments = [
            TranscriptSegment(
                start_seconds=window_start + float(item["start"]),
                end_seconds=window_start + float(item["end"]),
                text=str(item["text"]).strip(),
                source=source,
            )
            for item in message.get("segments", [])
            if str(item.get("text", "")).strip()
        ]

        changed = self._emit_preview(source, decoded_segments, stable_cutoff)
        for candidate in decoded_segments:
            if candidate.end_seconds > stable_cutoff:
                continue
            changed = self._ingest_candidate(source, candidate) or changed

        if changed:
            counts = self._snapshot_counts()
            self._notify_status(
                "Merged transcript live - "
                f"{counts['merged']} merged segments "
                f"(mic {counts['microphone']}, speaker {counts['speakers']}) - "
                f"model={self._model_label}"
            )

        if not self._stop_event.is_set():
            self._dispatch_job_for_source(source)

    def _handle_error(self, message: dict) -> None:
        source = message.get("source", "unknown")
        self._source_inflight[source] = False
        self._notify_status(f"Transcription error on {source}: {message.get('error', 'unknown error')}")

    def _emit_preview(
        self,
        source: str,
        decoded_segments: list[TranscriptSegment],
        stable_cutoff: float,
    ) -> bool:
        preview_segments = [
            segment
            for segment in decoded_segments
            if segment.end_seconds > max(0.0, stable_cutoff - PREVIEW_LOOKBACK_SECONDS)
        ]

        if not preview_segments:
            preview_signature = None
        else:
            preview_start = preview_segments[0].start_seconds
            preview_end = preview_segments[-1].end_seconds
            preview_text = " ".join(segment.text.strip() for segment in preview_segments).strip()
            preview_signature = (round(preview_start, 3), round(preview_end, 3), preview_text)

        with self._lock:
            previous_signature = self._source_preview_signature[source]
            if preview_signature == previous_signature:
                return False
            self._source_preview_signature[source] = preview_signature

        if preview_signature is None:
            self._emit_segment(TranscriptSegment(0.0, 0.0, "", source=source), is_preview=True)
            return True

        preview_start, preview_end, preview_text = preview_signature
        self._emit_segment(
            TranscriptSegment(preview_start, preview_end, preview_text, source=source),
            is_preview=True,
        )
        return True

    def _ingest_candidate(self, source: str, candidate: TranscriptSegment) -> bool:
        with self._lock:
            source_document = self._source_transcripts[source]
            recent_source_segments = source_document.segments[-6:]
            committed_until = self._source_committed_until[source]

            if not _accept_source_candidate(candidate, recent_source_segments, committed_until):
                return False

            source_document.add_segment(candidate)
            self._source_committed_until[source] = max(committed_until, candidate.end_seconds)

            if not _accept_merged_candidate(candidate, self._merged_transcript.segments[-8:]):
                return False

            self._merged_transcript.add_segment(candidate)

        self._emit_segment(candidate, is_preview=False)
        return True

    def _build_final_artifacts(self, capture_result: CaptureResult) -> TranscriptArtifactsResult | None:
        if WhisperModel is None:
            self._notify_status("Transcript finalization unavailable: faster-whisper is not installed.")
            return None

        model = WhisperModel(
            self._model_reference,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=str(WHISPER_DOWNLOAD_ROOT),
            local_files_only=True,
        )

        source_documents: dict[str, TranscriptDocument] = {}
        merged_document = TranscriptDocument(
            source_audio_path=str(capture_result.mixed_wav) if capture_result.mixed_wav else None,
            language=WHISPER_LANGUAGE,
        )

        for source in self.capture_service.active_sources:
            source_path = _source_audio_path(capture_result, source)
            if source_path is None:
                continue

            segments, info = model.transcribe(
                source_path,
                language=WHISPER_LANGUAGE,
                task="transcribe",
                beam_size=WHISPER_BEAM_SIZE,
                best_of=WHISPER_BEST_OF,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                word_timestamps=False,
            )
            document = TranscriptDocument(source_audio_path=source_path, language=getattr(info, "language", "en"))
            for segment in segments:
                text = str(segment.text).strip()
                if not text:
                    continue
                document.add_segment(
                    TranscriptSegment(
                        start_seconds=float(segment.start),
                        end_seconds=float(segment.end),
                        text=text,
                        source=source,
                    )
                )
            source_documents[source] = document

        for segment in sorted(
            [segment for document in source_documents.values() for segment in document.segments],
            key=lambda item: (item.start_seconds, item.end_seconds, item.source),
        ):
            if _accept_merged_candidate(segment, merged_document.segments[-8:]):
                merged_document.add_segment(segment)

        if not merged_document.segments:
            self._notify_status("Transcript ready, but no speech segments were detected.")
            return None

        source_json_paths: dict[str, Path] = {}
        source_text_paths: dict[str, Path] = {}
        for source, document in source_documents.items():
            source_json_paths[source] = document.save_json(
                capture_result.output_directory / f"transcript_{source}.json"
            )
            source_text_paths[source] = document.save_text(
                capture_result.output_directory / f"transcript_{source}.txt"
            )

        artifacts = generate_transcript_artifacts_from_document(
            transcript=merged_document,
            output_directory=capture_result.output_directory,
            window_seconds=DEFAULT_CHUNK_WINDOW_SECONDS,
        )
        self._notify_status(
            f"Merged transcript saved with {artifacts.chunk_count} chunks using model {self._model_label}."
        )
        return TranscriptArtifactsResult(
            transcript_json_path=artifacts.transcript_json_path,
            transcript_text_path=artifacts.transcript_text_path,
            chunks_json_path=artifacts.chunks_json_path,
            chunks_text_path=artifacts.chunks_text_path,
            chunk_count=artifacts.chunk_count,
            source_json_paths=source_json_paths,
            source_text_paths=source_text_paths,
        )

    def _emit_segment(self, segment: TranscriptSegment, *, is_preview: bool) -> None:
        update = TranscriptUpdate(segment=segment, is_preview=is_preview)
        for listener in self._segment_listeners:
            listener(update)

    def _notify_status(self, status: str) -> None:
        for listener in self._status_listeners:
            listener(status)

    def _snapshot_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "microphone": len(self._source_transcripts["microphone"].segments),
                "speakers": len(self._source_transcripts["speakers"].segments),
                "merged": len(self._merged_transcript.segments),
            }

    def _shutdown_workers(self) -> None:
        for queue in self._command_queues.values():
            try:
                queue.put({"type": "stop"})
            except Exception:
                pass

        for process in self._worker_processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

        self._worker_processes = {}
        self._command_queues = {}
        self._result_queue = None

    def _reset_state(self) -> None:
        self._language = WHISPER_LANGUAGE
        self._source_transcripts = {
            source: TranscriptDocument(language=WHISPER_LANGUAGE) for source in TRANSCRIPT_SOURCES
        }
        self._merged_transcript = TranscriptDocument(language=WHISPER_LANGUAGE)
        self._source_committed_until = {source: 0.0 for source in TRANSCRIPT_SOURCES}
        self._source_last_dispatched_total = {source: 0.0 for source in TRANSCRIPT_SOURCES}
        self._source_inflight = {source: False for source in TRANSCRIPT_SOURCES}
        self._source_request_id = {source: 0 for source in TRANSCRIPT_SOURCES}
        self._source_preview_signature = {source: None for source in TRANSCRIPT_SOURCES}
        self._ready_sources = set()

    @staticmethod
    def _resolve_model_reference() -> tuple[str, str]:
        if _is_complete_model_dir(WHISPER_LOCAL_MODEL_DIR):
            return str(WHISPER_LOCAL_MODEL_DIR), "custom-local"

        for model_name in WHISPER_MODEL_CANDIDATES:
            snapshot_root = MODELS_DIR / f"models--Systran--faster-whisper-{model_name}" / "snapshots"
            if snapshot_root.exists():
                snapshots = sorted(
                    [path for path in snapshot_root.iterdir() if path.is_dir()],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                for snapshot in snapshots:
                    if _is_complete_model_dir(snapshot):
                        return str(snapshot), model_name

        raise FileNotFoundError(
            "No complete local Whisper model found in models/. "
            "tiny.en should already be there. Do not download small.en."
        )


def _accept_source_candidate(
    candidate: TranscriptSegment,
    recent_segments: list[TranscriptSegment],
    committed_until: float,
) -> bool:
    if candidate.end_seconds <= committed_until + 0.05:
        return False

    if candidate.start_seconds < committed_until - LIVE_TRANSCRIBE_ACCEPT_OVERLAP_SECONDS:
        return False

    normalized_candidate = _normalize_text(candidate.text)
    for existing_segment in recent_segments:
        normalized_existing = _normalize_text(existing_segment.text)
        if (
            normalized_candidate == normalized_existing
            and abs(candidate.start_seconds - existing_segment.start_seconds) <= 3.0
        ):
            return False

    return True


def _accept_merged_candidate(
    candidate: TranscriptSegment,
    recent_segments: list[TranscriptSegment],
) -> bool:
    normalized_candidate = _normalize_text(candidate.text)
    for existing_segment in recent_segments:
        overlap = min(candidate.end_seconds, existing_segment.end_seconds) - max(
            candidate.start_seconds,
            existing_segment.start_seconds,
        )
        min_duration = min(
            candidate.end_seconds - candidate.start_seconds,
            existing_segment.end_seconds - existing_segment.start_seconds,
        )
        overlap_ratio = (overlap / min_duration) if min_duration > 0 and overlap > 0 else 0.0
        normalized_existing = _normalize_text(existing_segment.text)

        if (
            overlap_ratio >= 0.6
            and (
                normalized_candidate == normalized_existing
                or normalized_candidate in normalized_existing
                or normalized_existing in normalized_candidate
            )
        ):
            return False

    return True


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _source_audio_path(capture_result: CaptureResult, source: str) -> str | None:
    if source == "microphone" and capture_result.microphone_wav is not None:
        return str(capture_result.microphone_wav)
    if source == "speakers" and capture_result.speaker_wav is not None:
        return str(capture_result.speaker_wav)
    return None


def _is_complete_model_dir(path: Path) -> bool:
    required_files = ("config.json", "tokenizer.json", "vocabulary.txt", "model.bin")
    return path.exists() and all((path / name).exists() for name in required_files)
