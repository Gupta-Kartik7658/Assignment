import os

# Use local Whisper only. Do not talk to Hugging Face on startup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

from local_granola.ui.main_window import run


if __name__ == "__main__":
    mp.freeze_support()
    run()
