# Local Granola Prototype

This repository now starts with the first MVP slice:

- capture the default microphone
- capture the default speaker output through Windows loopback
- capture both at the same time
- save `microphone.wav`, `speakers.wav`, `mixed.wav`, and `capture.json`

## Run locally

```powershell
python -m pip install -r requirements.txt
python app.py
```

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
    capture.json
```

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
