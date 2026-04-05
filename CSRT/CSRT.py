from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from tracker_framework import run_tracker_app_by_name


def main():
    run_tracker_app_by_name("CSRT")


if __name__ == "__main__":
    main()
