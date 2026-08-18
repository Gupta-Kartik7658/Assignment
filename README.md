# Local Granola Prototype

This repository now starts with the first MVP slice:

- capture the default microphone
- capture the default speaker output through Windows loopback
- capture both at the same time
- mix both sources into one canonical transcription audio file
- show transcript lines in the UI while recording
- save `microphone.wav`, `speakers.wav`, `mixed.wav`, `timeline.json`, and `capture.json`

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

On first transcription run, Whisper may download the configured local model into `models/`.

## Recommended Whisper model downloads

The app now prefers English-only models in this order:

1. `small.en`
2. `base.en`
3. `tiny.en`
4. `tiny`

Recommended manual downloads:

```powershell
cd D:\Assignment\Assignment
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8', download_root='models'); print('base.en ready')"
python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8', download_root='models'); print('small.en ready')"
```

Or use the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_whisper_models.ps1
```

If a model download is incomplete, the app will ignore it and fall back to the next complete model.

## Quick diagnostic

If you want a fast hardware sanity check before opening the UI:

```powershell
python -m local_granola.audio.self_test
```

That prints device discovery plus simple microphone and speaker-capture levels.

## Current output

Each recording creates a new folder inside `output/`:

```text
output/
  capture_YYYYMMDD_HHMMSS/
    microphone.wav
    speakers.wav
    mixed.wav
    timeline.json
    capture.json
```

`mixed.wav` is the file we should feed into local STT.

`timeline.json` preserves chunk timing from capture so the transcription layer has a clean timestamp spine to build on.

If live transcription is enabled successfully, the same capture folder will also receive:

```text
transcript.json
transcript.txt
chunks.json
chunks.txt
```

## Transcript + chunking scaffolding

Once STT returns timestamped segments, we can turn them into transcript artifacts with:

```powershell
python -m local_granola.meeting.demo
```

That currently generates:

```text
output/
  demo_transcript/
    transcript.json
    transcript.txt
    chunks.json
    chunks.txt
```

The default chunk window is 30 seconds. I chose 30 seconds over 60 because it keeps the later LLM pass more responsive during a live meeting while still giving enough context for meeting-note extraction.

## Why this shape

This keeps the app on a packaging-friendly path:

- local desktop UI
- local audio capture
- no cloud dependency
- files written to a predictable folder

That makes the next steps straightforward:

1. feed the mixed stream into local STT
2. chunk transcript text
3. send chunks to the local quantized Qwen runtime
4. package the UI app as a Windows distributable
