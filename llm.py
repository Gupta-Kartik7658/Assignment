"""
Quantized Qwen from models/qwen2.5-1.5b-instruct-q4_k_m.gguf (via local Ollama).
"""

import requests

from local_models import OLLAMA, QWEN_NAME, ensure_qwen

llm_call_count = 0


def qwen_call(prompt: str, max_tokens: int = 120) -> str:
    global llm_call_count
    llm_call_count += 1
    ensure_qwen()
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": QWEN_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens},
        },
        timeout=180,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def classify_placement(chunk: str, current_topic: str, sibling_titles: list[str]) -> str:
    siblings = ", ".join(sibling_titles) if sibling_titles else "(none)"
    prompt = (
        "You are labeling one new sentence in a meeting.\n"
        "Reply with ONLY one word: SAME, SUB, or NEW.\n"
        "SAME = continues the current topic.\n"
        "SUB = a more specific sub-topic of the current topic.\n"
        "NEW = a different topic at the same level.\n\n"
        f"Current topic: {current_topic or '(start of meeting)'}\n"
        f"Other topics at this level: {siblings}\n"
        f"New sentence: {chunk}\n\n"
        "Answer:"
    )
    raw = qwen_call(prompt, max_tokens=8).upper()
    for word in ("SAME", "SUB", "NEW"):
        if word in raw:
            return word
    return "SAME"
