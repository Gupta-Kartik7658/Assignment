# Tree-Based Real-Time Transcript Summarizer

Keeps a live outline of a conversation as a tree of topic summaries instead of one
flat summary, while keeping LLM calls to a minimum. Everything runs locally.

## Files

- `summary_tree.py` - `SummaryNode`, `Tree`, two-stage placement
- `local_models.py` - points at GGUFs in `models/` and registers them with Ollama
- `embedder.py` - `models/nomic-embed-text-v1.5.f16.gguf`
- `llm.py` - `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `main.py` - fake live session, no hints, real embed + Qwen
- `test_tree.py` - offline sanity checks (no models needed)

## Setup

Ollama must be running (it only executes the GGUFs, it does not download them).
Weights stay in `models/`:

- `models/nomic-embed-text-v1.5.f16.gguf`
- `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

```bash
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python test_tree.py
.venv\Scripts\python main.py
```

## Binary later

Pack the Python code, keep `models/` next to the exe. First run registers the
two GGUFs with local Ollama as `nomic-local` and `qwen-local`. No cloud, no keys.

## How a chunk is placed

1. Embed the last 1-3 chunks (nomic).
2. Compare to the active node, its parent, siblings, children, and root children.
3. If cosine >= 0.75 against some existing node → append there. No Qwen.
4. Otherwise wait for 2 votes in a row, then one short Qwen call:
   `SAME` / `SUB` / `NEW`. That is the only time placement uses the LLM.
5. Depth cap still applies: no nodes below depth 4.

Summaries and rollups are unchanged: one Qwen call scoped to a single node
when its buffer hits ~150 words, and lazy ancestor rollups.

## Depth cap

Root is depth 0, `MAX_DEPTH = 4` (5 levels). Covered by `test_depth_cap`.
