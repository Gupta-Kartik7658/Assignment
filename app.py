import multiprocessing as mp

from local_granola.ui.main_window import run


if __name__ == "__main__":
    mp.freeze_support()
    run()
