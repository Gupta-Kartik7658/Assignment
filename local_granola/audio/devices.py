from __future__ import annotations

from dataclasses import dataclass

import soundcard as sc


@dataclass(frozen=True)
class AudioDeviceSnapshot:
    default_microphone: str
    default_speaker: str
    loopback_microphone: str
    microphone_count: int
    speaker_count: int
    loopback_count: int


def _device_label(device: object | None) -> str:
    return str(device) if device is not None else "Unavailable"


def list_audio_devices() -> AudioDeviceSnapshot:
    microphones = sc.all_microphones()
    loopback_microphones = sc.all_microphones(include_loopback=True)
    speakers = sc.all_speakers()

    default_microphone = sc.default_microphone()
    default_speaker = sc.default_speaker()
    loopback_microphone = resolve_default_loopback_microphone()

    return AudioDeviceSnapshot(
        default_microphone=_device_label(default_microphone),
        default_speaker=_device_label(default_speaker),
        loopback_microphone=_device_label(loopback_microphone),
        microphone_count=len(microphones),
        speaker_count=len(speakers),
        loopback_count=len(loopback_microphones),
    )


def resolve_default_loopback_microphone():
    default_speaker = sc.default_speaker()
    speaker_name = getattr(default_speaker, "name", "")
    loopbacks = sc.all_microphones(include_loopback=True)

    for device in loopbacks:
        device_name = getattr(device, "name", "")
        if speaker_name and speaker_name == device_name:
            return device

    for device in loopbacks:
        label = str(device)
        if "Loopback" in label and speaker_name and speaker_name in label:
            return device

    try:
        return sc.get_microphone(speaker_name, include_loopback=True)
    except Exception:
        pass

    for device in loopbacks:
        if "Loopback" in str(device):
            return device

    raise RuntimeError("No loopback microphone was found for the default speaker.")
