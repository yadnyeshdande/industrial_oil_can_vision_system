"""
main.py
=======
Industrial Multi-Camera Vision Detection System v2.0
Main entry point.

Usage:
    python main.py [--config config/config.yaml]

Windows Service (NSSM):
    nssm install VisionSystem python main.py
    nssm set VisionSystem AppDirectory C:\\VisionSystem
    nssm start VisionSystem
"""

from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Industrial Multi-Camera Vision Detection System v2.0"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main():
    # Required for Windows multiprocessing
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    config_path = str(Path(args.config).resolve())

    if not Path(config_path).exists():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    # Import and run supervisor
    from supervisor.supervisor import run_supervisor
    run_supervisor(config_path)


if __name__ == "__main__":
    main()
