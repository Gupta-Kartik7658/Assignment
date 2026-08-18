"""
Download the local Qwen GGUF into models/ (this folder only).

Run:  .venv\\Scripts\\python download_models.py
"""

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ~1.12 GB, quantized Qwen2.5-1.5B Instruct. Stays in this folder.
URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEST = MODELS / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EXPECTED = 1117320736


def main():
    if DEST.exists() and DEST.stat().st_size == EXPECTED:
        print(f"already have {DEST} ({DEST.stat().st_size} bytes)")
        return
    if DEST.exists() and DEST.stat().st_size != EXPECTED:
        print(f"partial file {DEST.stat().st_size} bytes, re-downloading")
        DEST.unlink()

    print(f"downloading to {DEST}")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or EXPECTED)
        done = 0
        last_print = 0
        with open(DEST, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if done - last_print >= 50 * 1024 * 1024 or done == total:
                    pct = 100.0 * done / total
                    print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)")
                    last_print = done

    print(f"done: {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
