"""
GGUFs live in models/. Ollama only runs them; it does not fetch anything.
"""

import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OLLAMA = "http://localhost:11434"

QWEN_GGUF = MODELS / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
NOMIC_GGUF = MODELS / "nomic-embed-text-v1.5.f16.gguf"

QWEN_NAME = "qwen-local"
NOMIC_NAME = "nomic-local"


def _has(name: str) -> bool:
    r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
    r.raise_for_status()
    return any(m["name"].split(":")[0] == name for m in r.json().get("models", []))


def _is_gguf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def _create(name: str, gguf: Path) -> None:
    if not gguf.exists():
        raise FileNotFoundError(f"put the GGUF here: {gguf}")
    if gguf.stat().st_size < 10_000_000:
        raise FileNotFoundError(f"file too small, re-download: {gguf}")
    if not _is_gguf(gguf):
        raise FileNotFoundError(
            f"{gguf.name} is not a GGUF file (wrong or corrupt download).\n"
            f"Delete it and save the real file as:\n  {gguf}"
        )
    if _has(name):
        return
    path = str(gguf.resolve()).replace("\\", "/")
    modelfile = ROOT / f"Modelfile.{name}"
    extra = "PARAMETER num_ctx 2048\n" if name == QWEN_NAME else ""
    modelfile.write_text(
        f"FROM {path}\nPARAMETER temperature 0.2\n{extra}",
        encoding="utf-8",
    )
    print(f"registering {gguf.name} as {name} (one-time, from models/)...")
    subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile)],
        check=True,
        cwd=str(ROOT),
    )


def ensure_embedder() -> None:
    _create(NOMIC_NAME, NOMIC_GGUF)


def qwen_ready() -> bool:
    return QWEN_GGUF.exists() and QWEN_GGUF.stat().st_size > 10_000_000 and _is_gguf(QWEN_GGUF)


def ensure_qwen() -> None:
    _create(QWEN_NAME, QWEN_GGUF)


def ensure_models() -> None:
    ensure_embedder()
    ensure_qwen()
