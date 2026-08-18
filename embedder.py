"""
Embeddings from models/nomic-embed-text-v1.5.f16.gguf (via local Ollama).

nomic wants a task prefix. For "is this the same topic" we use clustering:
on both the chunk and the node, not search_query / search_document.
"""

import numpy as np
import requests

from local_models import NOMIC_NAME, OLLAMA, ensure_embedder

PREFIX = "clustering: "

# from calibrate.py: SAME/SUB/NEW overlap around 0.63-0.81
HIGH = 0.75  # this similar to an existing node -> stay, skip Qwen
LOW = 0.55  # safety net; we still ask Qwen, we do not auto-split


def embed(text: str) -> list[float]:
    ensure_embedder()
    payload = {"model": NOMIC_NAME, "input": PREFIX + (text or " ")}
    r = requests.post(f"{OLLAMA}/api/embed", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def cosine(a, b) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
