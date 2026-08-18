"""
Build dist/LocalGranola/LocalGranola.exe plus models/ next to it.

Run:  .venv\\Scripts\\python build_exe.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist" / "LocalGranola"
MODELS_SRC = ROOT / "models"
MODELS_DST = DIST / "models"


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def download_tiny_whisper() -> None:
    print("ensuring tiny.en whisper model is in models/ ...")
    code = (
        "from faster_whisper import WhisperModel;"
        "WhisperModel('tiny.en', device='cpu', compute_type='int8', download_root='models');"
        "print('tiny.en ready')"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(ROOT))


def main() -> None:
    download_tiny_whisper()
    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--windowed",
            "--name",
            "LocalGranola",
            "--collect-all",
            "faster_whisper",
            "--collect-all",
            "ctranslate2",
            "--collect-all",
            "tokenizers",
            "--collect-all",
            "soundcard",
            "--collect-submodules",
            "local_granola",
            "--hidden-import",
            "summary_tree",
            "--hidden-import",
            "embedder",
            "--hidden-import",
            "llm",
            "--hidden-import",
            "local_models",
            "--hidden-import",
            "app_paths",
            "--hidden-import",
            "numpy",
            "--hidden-import",
            "requests",
            "--hidden-import",
            "certifi",
            "--hidden-import",
            "multiprocessing",
            str(ROOT / "app.py"),
        ]
    )

    MODELS_DST.mkdir(parents=True, exist_ok=True)
    print(f"copying models -> {MODELS_DST}")
    shutil.copytree(MODELS_SRC, MODELS_DST, dirs_exist_ok=True)

    readme = DIST / "README.txt"
    readme.write_text(
        "Local Granola\n\n"
        "1. Keep this whole folder together. Do not move only the .exe.\n"
        "2. Keep the models folder next to LocalGranola.exe.\n"
        "3. Ollama must be installed. The app will try to start it.\n"
        "4. Double-click LocalGranola.exe\n"
        "5. If something fails, open granola.log in this folder.\n",
        encoding="utf-8",
    )
    print(f"\nDone. Run: {DIST / 'LocalGranola.exe'}")


if __name__ == "__main__":
    main()
