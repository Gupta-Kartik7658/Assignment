from __future__ import annotations

import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from local_granola.audio import AudioCaptureService, CaptureMode, ChunkEvent, list_audio_devices
from local_granola.config import APP_NAME, OUTPUT_DIR


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("860x620")
        self.root.minsize(760, 520)

        self.capture_service = AudioCaptureService(output_directory=OUTPUT_DIR)
        self.capture_service.add_listener(self._queue_event)
        self.events: queue.Queue[ChunkEvent] = queue.Queue()

        self.mode_var = tk.StringVar(value=CaptureMode.BOTH.value)
        self.status_var = tk.StringVar(value="Ready to record")
        self.timer_var = tk.StringVar(value="00:00")
        self.output_var = tk.StringVar(value=str(Path.cwd() / OUTPUT_DIR))
        self.mic_var = tk.StringVar(value="Loading…")
        self.speaker_var = tk.StringVar(value="Loading…")
        self.loopback_var = tk.StringVar(value="Loading…")
        self.level_var = tk.StringVar(value="No audio yet")

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
            text="Prototype slice: capture microphone and speaker audio locally",
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

        live = ttk.LabelFrame(root_frame, text="Live capture events", padding=12)
        live.pack(fill=tk.BOTH, expand=True)

        ttk.Label(live, textvariable=self.level_var).pack(anchor=tk.W)

        self.log = tk.Text(live, height=18, wrap=tk.WORD)
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
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not start recording:\n\n{exc}")
            return

        self.status_var.set("● Recording")
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

        self.status_var.set("Capture saved")
        self.timer_var.set("00:00")

        files = [
            result.microphone_wav,
            result.speaker_wav,
            result.mixed_wav,
            result.metadata_path,
        ]
        visible_files = "\n".join(str(path) for path in files if path is not None)
        self._append_log(
            f"Stopped recording after {result.duration_seconds:.1f}s.\nSaved:\n{visible_files}"
        )

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

    def _schedule_ui_refresh(self) -> None:
        self._drain_events()
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


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    MainWindow(root)
    root.mainloop()
