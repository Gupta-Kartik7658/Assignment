# Local Granola

Local “Granola clone”: capture a meeting, transcribe it live, and grow a **topic tree** instead of one flat summary.

Nothing leaves the machine. Whisper does STT. Nomic does topic similarity. Quantized Qwen writes summaries and decides SAME / SUB / NEW. Ollama only runs the two GGUFs already in `models/`.

## Setup

1. Install [Ollama](https://ollama.com) and leave it running.
2. Python 3.13, venv already in `.venv`.

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Other entry points:

| command | what it does |
|---|---|
| `python app.py` | live UI (mic + speakers → transcript → tree) |
| `python main.py` | fake standup through nomic + Qwen, no mic |
| `python test_tree.py` | tree tests, no models |
| `python calibrate.py` | prints nomic cosine scores (threshold tuning) |

Ollama must be up for embeddings and summaries. The app tries `ollama serve` if it is not.

## Models (`models/`)

Keep these next to the code. Do not delete them.

| path | role |
|---|---|
| `nomic-embed-text-v1.5.f16.gguf` | 768-d topic vectors (`clustering:` prefix) |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | SAME/SUB/NEW + per-node summaries |
| `models--Systran--faster-whisper-tiny.en/` | local STT (`tiny.en`, offline) |

First run registers the GGUFs with Ollama as `nomic-local` and `qwen-local`. Whisper is loaded with `local_files_only=True` so it does not hit Hugging Face.

## End-to-end (live)

```
mic and/or speakers
        │
        ▼
AudioCaptureService          16 kHz, mixed.wav + per-source wavs
        │
        ▼
LiveTranscriptionService     faster-whisper tiny.en, CPU int8
        │
        ├─ preview (is_preview=True)  → UI only, never the tree
        └─ stable segment             → ChunkIngestionService
                                              │
                                              │ batch ~8 words / ~4s
                                              ▼
                                        Tree.insert_chunk
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │ nomic cosine vs active / parent / siblings / kids │
                    │                                                     │
                    │  ≥ 0.75          → same node, no Qwen               │
                    │  2 “away” votes  → Qwen: SAME | SUB | NEW           │
                    │  depth cap 4     → extra text piles on that node    │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                                    that node's raw_chunks
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │ if buffer ≥ ~25 words or ~15s:                    │
                    │   one Qwen call = prior summary + new raw text    │
                    │   then clear raw_chunks                           │
                    │ ancestors only get dirty=True (no LLM yet)        │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                               UI “Live topic outline”
```

On **Stop recording**:

1. Final Whisper pass over the saved WAVs (cleaner than the live pass).
2. Tree flushes leftover buffers and rolls up dirty ancestors **once**, bottom-up.
3. Files are written under `output/capture_<timestamp>/`.

## Why a tree

A 30-minute meeting must not resend the whole transcript to Qwen on every update.

- **Placement** is embeddings first. Qwen is asked only when nomic is unsure.
- **Node summary** sees only that node’s previous summary + its new raw chunks. Cost is O(1) in session length.
- **Subtree rollup** (`subtree_summary`) is lazy: on Stop, or if the UI asks. Each rollup call is that node’s summary + its **direct children’s** summaries, never grandchildren, never raw audio.

Exactly one node is `active` (last append). The UI can highlight “what we are on right now.”

Root = whole session. Children = topics. Children of those = drill-downs. Branching is unbounded; this is not a binary tree. Max height is 5 levels (depth 0–4).

## Placement rules

Nomic cosine vs the active node, its parent, siblings, children, and root’s children.

| cosine vs best existing node | action |
|---|---|
| ≥ 0.75 | stay there, skip Qwen |
| below 0.75, first vote | debounce (need 2 in a row) |
| 2 away votes, Qwen = SAME | stay on active |
| Qwen = SUB | new child of active |
| Qwen = NEW | new child of active’s parent (sibling-level shift) |

Without Qwen loaded, NEW is guessed only if cosine < 0.55; otherwise SAME. Calibration showed SAME / SUB / NEW overlap in ~0.63–0.81, which is why Qwen owns the split.

## LLM budget

- Per live chunk: 0 Qwen calls (nomic sure) or 1 (verify a split) plus 0 or 1 (that node’s summary if the buffer is full).
- Per rollup cycle: 1 call per dirty node, never on every chunk.
- No call ever gets the full tree.

## UI

`python app.py` opens Local Granola.

- **Capture controls** — Microphone / Speakers / Both, Start / Stop.
- **Live transcript** — committed lines + grey `[live]` preview.
- **Live topic outline** — indented tree (`*` = active node). Summaries appear after a node’s buffer fills; until then you may see `[+N words buffered]`.
- **Stop** — outline flushes and `output/capture_…` is written.

## Output

Each session:

```
output/capture_YYYYMMDD_HHMMSS/
  microphone.wav
  speakers.wav
  mixed.wav
  transcript.json / .txt      merged stable transcript
  chunks.json / .txt          60s windows (post-recording)
  transcript_microphone.*     mic-only
  transcript_speakers.*       speaker-only
  timeline.json
  capture.json
```

Live notes come from in-memory stable segments, not from `chunks.json`. `chunks.json` is the after-the-fact export.

## Repo map

| path | job |
|---|---|
| `app.py` | desktop entry |
| `summary_tree.py` | `SummaryNode`, `Tree`, placement, summaries, rollup |
| `embedder.py` | nomic embed + cosine (`HIGH=0.75`, `LOW=0.55`) |
| `llm.py` | Qwen generate |
| `local_models.py` | GGUF paths, Ollama register |
| `app_paths.py` | project / exe directory |
| `local_granola/audio/` | capture, devices |
| `local_granola/stt/live_whisper.py` | live + final Whisper |
| `local_granola/llm/chunk_ingestion.py` | stable STT → `insert_chunk` |
| `local_granola/meeting/` | transcript docs, 60s chunker |
| `local_granola/ui/main_window.py` | Tk UI |
| `models/` | all weights |

## Packaging

`python build_exe.py` writes `dist/LocalGranola/LocalGranola.exe` and copies `models/` next to it. Keep the whole folder together. Ollama is still required at runtime.
