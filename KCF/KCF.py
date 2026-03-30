from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from tracker_framework import KCFTrackerAdapter, run_tracker_app


def main():
    run_tracker_app(KCFTrackerAdapter())


if __name__ == "__main__":
    main()