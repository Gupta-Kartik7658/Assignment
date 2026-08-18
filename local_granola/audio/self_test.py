from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import soundcard as sc

from .devices import list_audio_devices, resolve_default_loopback_microphone


@dataclass(frozen=True)
class SelfTestResult:
    microphone_peak: float
    microphone_rms: float
    speaker_peak: float
    speaker_rms: float


def run_self_test() -> SelfTestResult:
    sample_rate = 16_000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    stereo_tone = np.column_stack([tone, tone])

    speaker = sc.default_speaker()
    loopback = resolve_default_loopback_microphone()
    microphone = sc.default_microphone()
    speaker_box: dict[str, np.ndarray] = {}

    def capture_loopback() -> None:
        with loopback.recorder(samplerate=sample_rate, channels=2, blocksize=2048) as recorder:
            speaker_box["frames"] = recorder.record(numframes=sample_rate)

    thread = threading.Thread(target=capture_loopback, daemon=True)
    thread.start()
    time.sleep(0.1)
    speaker.play(stereo_tone, samplerate=sample_rate)
    thread.join()

    microphone_data = microphone.record(samplerate=sample_rate, numframes=sample_rate // 2)
    speaker_data = speaker_box["frames"]

    result = SelfTestResult(
        microphone_peak=float(np.max(np.abs(microphone_data))),
        microphone_rms=float(np.sqrt(np.mean(np.square(microphone_data)))),
        speaker_peak=float(np.max(np.abs(speaker_data))),
        speaker_rms=float(np.sqrt(np.mean(np.square(speaker_data)))),
    )
    print(
        json.dumps(
            {
                "devices": asdict(list_audio_devices()),
                "results": asdict(result),
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    run_self_test()
