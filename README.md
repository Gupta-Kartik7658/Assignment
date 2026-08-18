# Local Granola

Live meeting capture with a topic tree instead of one flat summary.
Everything runs locally: Whisper STT, nomic embeddings, quantized Qwen.

## Run

Ollama must be running. Weights stay in `models/`.

```powershell
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python test_tree.py
.venv\Scripts\python main.py
.venv\Scripts\python app.py
```

- `test_tree.py` — offline tree checks (no models)
- `main.py` — fake transcript through nomic + Qwen
- `app.py` — live UI: mic/speaker capture → Whisper → summary tree

## Local models

Put these in `models/` (already wired):

- `nomic-embed-text-v1.5.f16.gguf` — topic embeddings
- `qwen2.5-1.5b-instruct-q4_k_m.gguf` — summaries + SAME/SUB/NEW check

Whisper models also live under `models/` (faster-whisper snapshots).

## Live path

1. `AudioCaptureService` captures mic and/or speakers.
2. `LiveTranscriptionService` emits `TranscriptUpdate` events.
3. Preview text (`is_preview=True`) is UI-only.
4. Stable segments go to `ChunkIngestionService`, which batches ~18 words
   then calls `Tree.insert_chunk`.
5. Nomic places the chunk. If it looks like a new topic, Qwen answers
   SAME / SUB / NEW. Node summaries fire only when that node's buffer is full.
6. The UI outline panel shows the tree. On Stop, buffers flush and ancestors roll up.

## Output

Each recording writes `output/capture_<timestamp>/`:

```
microphone.wav, speakers.wav, mixed.wav
transcript.json, transcript.txt, chunks.json, chunks.txt
```

## Binary later

Pack the Python app, keep `models/` next to the exe. First run registers the
two GGUFs with local Ollama as `nomic-local` and `qwen-local`. No cloud, no keys.
