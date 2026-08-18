from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from local_granola.audio import AudioCaptureService, CaptureMode, ChunkEvent, list_audio_devices
from local_granola.config import APP_NAME, OUTPUT_DIR
from local_granola.llm import ChunkIngestionService
from local_granola.stt import LiveTranscriptionService, TranscriptArtifactsResult, TranscriptUpdate


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("980x760")
        self.root.minsize(860, 640)

        self.capture_service = AudioCaptureService(output_directory=OUTPUT_DIR)
        self.capture_service.add_listener(self._queue_event)
        self.events: queue.Queue[ChunkEvent] = queue.Queue()
        self.ingestion = ChunkIngestionService()
        self.ingestion.add_outline_listener(self._queue_outline)
        self.ingestion.add_status_listener(self._queue_transcript_status)
        self.transcription_service = LiveTranscriptionService(self.capture_service)
        self.transcription_service.add_segment_listener(self._queue_transcript)
        self.transcription_service.add_segment_listener(self.ingestion.handle_update)
        self.transcription_service.add_status_listener(self._queue_transcript_status)
        self.transcript_events: queue.Queue[TranscriptUpdate] = queue.Queue()
        self.transcript_status_events: queue.Queue[str] = queue.Queue()
        self.outline_events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.preview_by_source: dict[str, str] = {}
        self.committed_transcript_lines: list[str] = []

        self.mode_var = tk.StringVar(value=CaptureMode.BOTH.value)
        self.status_var = tk.StringVar(value="Ready to record")
        self.timer_var = tk.StringVar(value="00:00")
        self.output_var = tk.StringVar(value=str(Path.cwd() / OUTPUT_DIR))
        self.mic_var = tk.StringVar(value="Loading…")
        self.speaker_var = tk.StringVar(value="Loading…")
        self.loopback_var = tk.StringVar(value="Loading…")
        self.level_var = tk.StringVar(value="No audio yet")
        self.transcript_status_var = tk.StringVar(value="Transcript idle")
        self.preview_var = tk.StringVar(value="Preview will appear here as you speak.")

        self._build()
        self._refresh_devices()
        self._schedule_ui_refresh()

    def _build(self) -> None:
        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root_frame)
        header.pack(fill=tk.X)

        ttk.Label(header, text=APP_NAME, font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Live transcript + topic outline from local nomic + Qwen",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(4, 0))

        controls = ttk.LabelFrame(root_frame, text="Capture controls", padding=12)
        controls.pack(fill=tk.X, pady=(16, 12))

        ttk.Label(controls, text="Audio source").grid(row=0, column=0, sticky=tk.W)
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            state="readonly",
            values=[mode.value for mode in CaptureMode],
            width=18,
        )
        mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(12, 16))

        ttk.Button(controls, text="Refresh devices", command=self._refresh_devices).grid(
            row=0, column=2, sticky=tk.W
        )

        ttk.Button(controls, text="Start recording", command=self._start_recording).grid(
            row=1, column=0, pady=(14, 0), sticky=tk.W
        )
        ttk.Button(controls, text="Stop recording", command=self._stop_recording).grid(
            row=1, column=1, pady=(14, 0), sticky=tk.W, padx=(12, 0)
        )

        ttk.Label(controls, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).grid(
            row=1, column=2, sticky=tk.W, pady=(14, 0), padx=(16, 0)
        )
        ttk.Label(controls, textvariable=self.timer_var).grid(row=1, column=3, sticky=tk.W, pady=(14, 0))

        devices = ttk.LabelFrame(root_frame, text="Detected devices", padding=12)
        devices.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(devices, text="Default microphone").grid(row=0, column=0, sticky=tk.NW)
        ttk.Label(devices, textvariable=self.mic_var, wraplength=620).grid(
            row=0, column=1, sticky=tk.W, padx=(12, 0)
        )
        ttk.Label(devices, text="Default speaker").grid(row=1, column=0, sticky=tk.NW, pady=(8, 0))
        ttk.Label(devices, textvariable=self.speaker_var, wraplength=620).grid(
            row=1, column=1, sticky=tk.W, padx=(12, 0), pady=(8, 0)
        )
        ttk.Label(devices, text="Loopback source").grid(row=2, column=0, sticky=tk.NW, pady=(8, 0))
        ttk.Label(devices, textvariable=self.loopback_var, wraplength=620).grid(
            row=2, column=1, sticky=tk.W, padx=(12, 0), pady=(8, 0)
        )

        transcript_frame = ttk.LabelFrame(root_frame, text="Live transcript", padding=12)
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        ttk.Label(transcript_frame, textvariable=self.transcript_status_var).pack(anchor=tk.W)
        ttk.Label(
            transcript_frame,
            textvariable=self.preview_var,
            wraplength=760,
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(6, 6))

        self.transcript_box = tk.Text(transcript_frame, height=12, wrap=tk.WORD)
        self.transcript_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.transcript_box.configure(state=tk.DISABLED)

        outline_frame = ttk.LabelFrame(root_frame, text="Live topic outline", padding=12)
        outline_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.outline_box = tk.Text(outline_frame, height=10, wrap=tk.WORD)
        self.outline_box.pack(fill=tk.BOTH, expand=True)
        self.outline_box.configure(state=tk.DISABLED)

        live = ttk.LabelFrame(root_frame, text="Capture diagnostics", padding=12)
        live.pack(fill=tk.BOTH, expand=True)

        ttk.Label(live, textvariable=self.level_var).pack(anchor=tk.W)

        self.log = tk.Text(live, height=8, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log.configure(state=tk.DISABLED)

        footer = ttk.Frame(root_frame)
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(footer, text="Output directory").pack(anchor=tk.W)
        ttk.Label(footer, textvariable=self.output_var, foreground="#555555").pack(anchor=tk.W)

    def _start_recording(self) -> None:
        if self.capture_service.is_recording:
            return

        try:
            mode = CaptureMode(self.mode_var.get())
            self.capture_service.start(mode)
            self.ingestion.start()
            self.transcription_service.start()
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not start recording:\n\n{exc}")
            return

        self.status_var.set("● Recording")
        self.transcript_status_var.set("Starting live transcript…")
        self._clear_transcript()
        self.committed_transcript_lines = []
        self.preview_by_source = {}
        self.preview_var.set("Preview will appear here as you speak.")
        self._append_log(f"Started recording from {mode.value}.")

    def _stop_recording(self) -> None:
        if not self.capture_service.is_recording:
            return

        try:
            result = self.capture_service.stop()
        except Exception as exc:
            self.status_var.set("Capture failed")
            messagebox.showerror(APP_NAME, f"Could not stop recording cleanly:\n\n{exc}")
            return

        try:
            transcript_artifacts = self.transcription_service.stop(capture_result=result)
        except Exception as exc:
            transcript_artifacts = None
            self._append_log(f"Transcript finalization failed: {exc}")
            self.transcript_status_var.set(f"Transcript finalization failed: {exc}")

        try:
            outline = self.ingestion.stop()
            self._set_outline(outline)
        except Exception as exc:
            self._append_log(f"Summary tree flush failed: {exc}")

        self.status_var.set("Capture saved")
        self.timer_var.set("00:00")

        files = [
            result.microphone_wav,
            result.speaker_wav,
            result.mixed_wav,
            result.timeline_path,
            result.metadata_path,
        ]
        if transcript_artifacts is not None:
            files.extend(
                [
                    transcript_artifacts.transcript_json_path,
                    transcript_artifacts.transcript_text_path,
                    transcript_artifacts.chunks_json_path,
                    transcript_artifacts.chunks_text_path,
                ]
            )
            files.extend(transcript_artifacts.source_json_paths.values())
            files.extend(transcript_artifacts.source_text_paths.values())
        visible_files = "\n".join(str(path) for path in files if path is not None)
        self._append_log(
            f"Stopped recording after {result.duration_seconds:.1f}s.\nSaved:\n{visible_files}"
        )
        self.preview_by_source = {}
        self.preview_var.set("Preview cleared.")
        self._render_transcript_box()

    def _refresh_devices(self) -> None:
        try:
            snapshot = list_audio_devices()
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not enumerate audio devices:\n\n{exc}")
            return

        self.mic_var.set(snapshot.default_microphone)
        self.speaker_var.set(snapshot.default_speaker)
        self.loopback_var.set(snapshot.loopback_microphone)
        self._append_log(
            "Detected devices refreshed. "
            f"microphones={snapshot.microphone_count}, "
            f"speakers={snapshot.speaker_count}, "
            f"loopback inputs={snapshot.loopback_count}"
        )

    def _queue_event(self, event: ChunkEvent) -> None:
        self.events.put(event)

    def _queue_transcript(self, update: TranscriptUpdate) -> None:
        self.transcript_events.put(update)

    def _queue_transcript_status(self, status: str) -> None:
        self.transcript_status_events.put(status)

    def _queue_outline(self, outline: str, reason: str) -> None:
        self.outline_events.put((outline, reason))

    def _schedule_ui_refresh(self) -> None:
        self._drain_events()
        self._drain_transcript_events()
        self._drain_outline_events()
        self._refresh_timer()
        self.root.after(200, self._schedule_ui_refresh)

    def _drain_events(self) -> None:
        processed = 0
        while processed < 25:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            self.level_var.set(
                f"Latest chunk: {event.source} | frames={event.frames} | "
                f"peak={event.peak:.3f} | rms={event.rms:.3f}"
            )
            self._append_log(
                f"{event.elapsed_seconds:6.2f}s  {event.source:<10}  "
                f"frames={event.frames:<5} peak={event.peak:.3f} rms={event.rms:.3f}"
            )
            processed += 1

    def _drain_transcript_events(self) -> None:
        processed = 0
        while processed < 25:
            try:
                status = self.transcript_status_events.get_nowait()
            except queue.Empty:
                break
            self.transcript_status_var.set(status)
            processed += 1

        processed = 0
        while processed < 25:
            try:
                update = self.transcript_events.get_nowait()
            except queue.Empty:
                break

            self._append_transcript_segment(update)
            processed += 1

    def _drain_outline_events(self) -> None:
        latest = None
        while True:
            try:
                latest = self.outline_events.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._set_outline(latest[0])

    def _set_outline(self, text: str) -> None:
        self.outline_box.configure(state=tk.NORMAL)
        self.outline_box.delete("1.0", tk.END)
        self.outline_box.insert(tk.END, text.rstrip() + "\n")
        self.outline_box.see(tk.END)
        self.outline_box.configure(state=tk.DISABLED)

    def _refresh_timer(self) -> None:
        if not self.capture_service.is_recording:
            return

        elapsed = max(0, int(self.capture_service.elapsed_seconds))
        minutes, seconds = divmod(elapsed, 60)
        self.timer_var.set(f"{minutes:02d}:{seconds:02d}")

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"{message}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _append_transcript_segment(self, update: TranscriptUpdate) -> None:
        segment = update.segment
        if update.is_preview:
            if segment.text.strip():
                self.preview_by_source[segment.source] = segment.to_text_line()
            else:
                self.preview_by_source.pop(segment.source, None)

            if self.preview_by_source:
                ordered_sources = ["microphone", "speakers"]
                lines = [
                    self.preview_by_source[source]
                    for source in ordered_sources
                    if source in self.preview_by_source
                ]
                self.preview_var.set("Live preview " + " | ".join(lines))
            else:
                self.preview_var.set("Preview will appear here as you speak.")
            self._render_transcript_box()
            return

        line = f"{segment.to_text_line()}\n"
        self.committed_transcript_lines.append(line.rstrip("\n"))
        self._render_transcript_box()

    def _clear_transcript(self) -> None:
        self.committed_transcript_lines = []
        self.transcript_box.configure(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.configure(state=tk.DISABLED)

    def _render_transcript_box(self) -> None:
        lines = list(self.committed_transcript_lines)

        preview_lines = []
        ordered_sources = ["microphone", "speakers"]
        for source in ordered_sources:
            if source not in self.preview_by_source:
                continue
            preview_lines.append(f"{self.preview_by_source[source]}  [live]")

        if preview_lines:
            if lines:
                lines.append("")
            lines.extend(preview_lines)

        body = "\n".join(lines)
        if body:
            body += "\n"

        self.transcript_box.configure(state=tk.NORMAL)
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, body)
        self.transcript_box.see(tk.END)
        self.transcript_box.configure(state=tk.DISABLED)


def run() -> None:
    import os
    import sys
    import traceback
    from app_paths import app_dir
    from local_models import runtime_problems

    os.chdir(app_dir())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (app_dir() / "models").mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    try:
        problems = runtime_problems()
        if problems:
            messagebox.showwarning(APP_NAME, "\n\n".join(problems))
        MainWindow(root)
        root.mainloop()
    except Exception:
        text = traceback.format_exc()
        (app_dir() / "granola.log").write_text(text, encoding="utf-8")
        messagebox.showerror(APP_NAME, f"The app crashed.\nDetails saved to granola.log\n\n{text[-800:]}")
        raise
