"""
Print cosine similarities from nomic-embed-text so we can pick thresholds.

We already know which chunks are SAME / SUB / NEW (from the demo transcript).
The script just measures what the embedder thinks, it does not place nodes.

Needs Ollama running. Uses models/nomic-embed-text-v1.5.f16.gguf.

Run:  .venv\\Scripts\\python calibrate.py
"""

from embedder import cosine, embed

# labeled like the later two-stage check: SAME stay, SUB drill down, NEW topic shift
CHUNKS = [
    ("Alright let's start the standup, quick round of updates.", None),
    ("First up the payments migration, we moved 60 percent of traffic.", "SUB"),
    ("Error rate stayed flat at 0.2 percent so far.", "SAME"),
    ("We still need to migrate refunds, that's next week.", "SAME"),
    ("On refunds specifically, the old provider api is being shut off in May.", "SUB"),
    ("So we have a hard deadline there, no room to slip.", "SAME"),
    ("Switching topics, the mobile release got rejected by review.", "NEW"),
    ("They flagged the microphone permission description.", "SAME"),
    ("We resubmitted yesterday with a longer explanation.", "SAME"),
    ("Last thing, hiring. Two onsites this week for the backend role.", "NEW"),
    ("Both candidates look strong, decision by Friday.", "SAME"),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    print("embedding chunks...")
    vecs = [embed(text) for text, _ in CHUNKS]

    print("\n--- consecutive chunk vs previous chunk ---")
    print(f"{'rel':<6} {'cos':>6}  chunk")
    by_rel = {"SAME": [], "SUB": [], "NEW": []}
    for i in range(1, len(CHUNKS)):
        rel = CHUNKS[i][1]
        score = cosine(vecs[i], vecs[i - 1])
        by_rel[rel].append(score)
        print(f"{rel:<6} {score:6.3f}  {CHUNKS[i][0][:70]}")

    print("\n--- chunk vs running topic blob (closer to real node embedding) ---")
    print(f"{'rel':<6} {'cos':>6}  chunk")
    blob_rel = {"SAME": [], "SUB": [], "NEW": []}
    blob = CHUNKS[0][0]
    blob_vec = vecs[0]
    for i in range(1, len(CHUNKS)):
        rel = CHUNKS[i][1]
        score = cosine(vecs[i], blob_vec)
        blob_rel[rel].append(score)
        print(f"{rel:<6} {score:6.3f}  {CHUNKS[i][0][:70]}")
        if rel == "SAME":
            blob = blob + " " + CHUNKS[i][0]
            blob_vec = embed(blob)
        else:
            blob = CHUNKS[i][0]
            blob_vec = vecs[i]

    print("\n--- ranges (chunk vs previous chunk) ---")
    for rel in ("SAME", "SUB", "NEW"):
        xs = by_rel[rel]
        print(f"{rel:<6} n={len(xs)}  min={min(xs):.3f}  mean={mean(xs):.3f}  max={max(xs):.3f}")

    print("\n--- ranges (chunk vs topic blob) ---")
    for rel in ("SAME", "SUB", "NEW"):
        xs = blob_rel[rel]
        print(f"{rel:<6} n={len(xs)}  min={min(xs):.3f}  mean={mean(xs):.3f}  max={max(xs):.3f}")

    same_min = min(blob_rel["SAME"])
    shift_max = max(blob_rel["SUB"] + blob_rel["NEW"])
    print("\n--- first-pass threshold guess (from topic-blob numbers) ---")
    print(f"SAME min = {same_min:.3f}")
    print(f"SUB/NEW max = {shift_max:.3f}")
    if same_min > shift_max:
        high = (same_min + shift_max) / 2
        print(f"no overlap. high threshold ~ {high:.3f}  (above this = SAME, skip Qwen)")
        print(f"low threshold  ~ {shift_max - 0.05:.3f}  (below this = likely NEW, still ask Qwen)")
    else:
        print("overlap. cosine alone cannot tell SAME vs SUB vs NEW on this sample.")
        print("that is why the next step is the Qwen check, not a hard cutoff.")
        print("a skip-Qwen bar would need to sit above every SUB/NEW score,")
        print(f"so high ~ {shift_max + 0.02:.3f}  (almost nothing will skip the check)")
        print("low ~ 0.55  (we never saw a score that low here; keep it as a safety net)")


if __name__ == "__main__":
    main()
