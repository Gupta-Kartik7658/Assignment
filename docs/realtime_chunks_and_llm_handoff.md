# Realtime Chunks and LLM Handoff

## Goal

This document explains:

- where transcript chunks are created
- where they are saved
- how they should be passed to the local LLM
- what should be used for realtime access vs post-recording export

## Current transcript flow

1. `AudioCaptureService` captures microphone and/or speaker audio.
2. `LiveTranscriptionService` runs live STT and emits `TranscriptUpdate` events.
3. Stable transcript segments are accumulated in memory.
4. When recording stops, the final transcript is rebuilt from the saved WAV files.
5. The final merged transcript is chunked into fixed windows and saved to disk.

## Where chunks come from

### 1. Realtime source in memory

Realtime transcript data comes from:

- `local_granola/stt/live_whisper.py`
- event type: `TranscriptUpdate`
- stable events: `is_preview=False`
- preview events: `is_preview=True`

Important:

- preview text is only for UI display
- preview text should not be treated as the authoritative LLM input
- realtime LLM ingestion should use only stable transcript segments

The stable segment objects are `TranscriptSegment` values with:

- `start_seconds`
- `end_seconds`
- `text`
- `source`

### 2. Post-recording chunk files on disk

After recording stops, the app writes artifacts under:

`output/capture_<timestamp>/`

The important files are:

- `transcript.json` — final merged transcript
- `transcript.txt` — final merged transcript as readable text
- `chunks.json` — final merged 60-second chunks
- `chunks.txt` — final merged chunks as readable text
- `transcript_microphone.json` — final mic-only transcript
- `transcript_microphone.txt`
- `transcript_speakers.json` — final speaker-only transcript
- `transcript_speakers.txt`
- `timeline.json` — capture timeline metadata
- `capture.json` — capture metadata and file paths

Important:

- `chunks.json` is not realtime
- `chunks.json` is generated only after recording stops

## Where chunking is implemented

Chunking is currently implemented in:

- `local_granola/meeting/chunker.py`
- function: `build_chunks(...)`

The default chunk window comes from:

- `local_granola/config.py`
- `DEFAULT_CHUNK_WINDOW_SECONDS = 60.0`

The final chunk files are created through:

- `local_granola/meeting/pipeline.py`
- `generate_transcript_artifacts_from_document(...)`

The final STT service calls this from:

- `local_granola/stt/live_whisper.py`
- method: `_build_final_artifacts(...)`

## What should be sent to the LLM

Yes — the chunks should be the main input to the local LLM.

Recommended rule:

- use stable transcript chunks for reasoning
- use preview text only for UX

Each LLM chunk payload should contain:

```json
{
  "chunk_index": 3,
  "start_seconds": 120.0,
  "end_seconds": 180.0,
  "text": "merged readable text for this chunk",
  "segments": [
    {
      "start_seconds": 123.2,
      "end_seconds": 129.7,
      "text": "Hello everyone, welcome to the review meeting.",
      "source": "microphone"
    }
  ],
  "language": "en",
  "is_final": false
}
```

## Realtime access: what it should mean

Realtime access should not wait for `chunks.json`.

Instead, realtime access should work like this:

1. subscribe to stable transcript events from `LiveTranscriptionService`
2. collect stable segments into an in-memory active chunk
3. once the active chunk crosses 60 seconds, finalize it
4. immediately send that finalized chunk to the LLM
5. start a new active chunk

This means:

- UI stays live through preview events
- LLM receives only stable text
- saved disk files remain the post-recording source of truth

## Recommended architecture for LLM ingestion

Add a new service, for example:

- `local_granola/llm/chunk_ingestion.py`

Suggested responsibility:

- listen to `TranscriptUpdate`
- ignore `is_preview=True`
- accumulate stable `TranscriptSegment` entries
- emit a chunk payload every 60 seconds
- optionally flush early after a silence timeout

Suggested internal state:

```python
current_chunk_index: int
current_chunk_start_seconds: float
current_chunk_segments: list[TranscriptSegment]
last_emitted_chunk_index: int
```

Suggested emit rules:

- normal flush: when `segment.end_seconds >= current_chunk_start_seconds + 60`
- idle flush: if no new stable segment arrives for 5 to 10 seconds
- final flush: when recording stops

## Best source for the LLM

Use this priority:

1. stable realtime in-memory chunks for live notes
2. final post-recording chunks for cleanup, correction, and final summaries

This is useful because:

- the live STT is lower latency
- the final STT pass is higher quality
- the LLM can first work live, then refine from final chunks later

## Recommended LLM workflow

### Live phase

For each stable 60-second chunk:

- send the chunk text to the local LLM
- ask for rolling notes, action items, decisions, and risks
- keep the last 1 to 3 chunks as context

### Finalization phase

After recording stops:

- load `chunks.json`
- re-run the LLM over the final chunk sequence
- replace or refine the live notes with higher-quality final notes

## Important caveats

- do not feed raw preview text into the final notes pipeline
- do not rely on `chunks.json` for live processing
- do not rely on the UI text box as the data source
- use `TranscriptSegment` / stable event data as the canonical realtime source

## Suggested next implementation step

The next code step should be:

1. create a `ChunkIngestionService`
2. subscribe it to `LiveTranscriptionService.add_segment_listener(...)`
3. emit 60-second stable chunks in memory
4. hand those chunks to the local LLM immediately

That gives us proper realtime LLM access without waiting for the saved transcript files.
