# Local Granola

Local meeting app: live transcript + a **topic tree** (not one flat summary).
No cloud. Ollama must be running.

## Run

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

`python main.py` — fake transcript, no mic. `python test_tree.py` — tree logic only.

## Models (`models/`)

| file | used for |
|---|---|
| `nomic-embed-text-v1.5.f16.gguf` | is this the same topic? |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | SAME/SUB/NEW + node summaries |
| faster-whisper `tiny.en` snapshot | speech → text |

## End to end

```
mic / speakers
    → Whisper (tiny.en)          live transcript
    → stable segments only       previews stay in the UI
    → nomic embedding            stick to current node, or flag a split
    → Qwen (only if unsure)      SAME / SUB / NEW  (max depth 4)
    → Qwen (per node, lazy)      that node's summary, not the whole tree
    → UI "Live topic outline"
```

On **Stop**: leftover buffers flush, ancestors roll up once, files land in `output/capture_<timestamp>/`.

LLM budget: 0–1 Qwen calls per chunk (placement or that node's summary). Rollups are not per-chunk.

## Layout

- `app.py` — UI
- `summary_tree.py` — topic tree
- `embedder.py` / `llm.py` / `local_models.py` — nomic + Qwen via Ollama
- `local_granola/stt/` — live Whisper
- `local_granola/llm/chunk_ingestion.py` — STT → tree
- `local_granola/audio/` — capture
