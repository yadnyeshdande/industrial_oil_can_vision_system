"""
scripts/health_dashboard.py
============================
Command-line health dashboard for the running system.
Reads log files and displays real-time status.

Usage: python scripts/health_dashboard.py
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

LOG_DIR = _ROOT / "logs"


def read_last_lines(filepath: Path, n: int = 20) -> list:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except FileNotFoundError:
        return [f"[Log not found: {filepath}]"]
    except Exception as e:
        return [f"[Error reading log: {e}]"]


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def format_section(title: str, lines: list) -> str:
    out = [f"\n  {'─'*56}", f"  {title}", f"  {'─'*56}"]
    for line in lines[-5:]:
        out.append(f"  {line}")
    return "\n".join(out)


def main():
    print("Industrial Vision System — Health Dashboard")
    print("Press Ctrl+C to exit\n")

    log_files = {
        "Supervisor":    LOG_DIR / "supervisor.log",
        "Camera 0":      LOG_DIR / "camera_0.log",
        "Camera 1":      LOG_DIR / "camera_1.log",
        "Camera 2":      LOG_DIR / "camera_2.log",
        "Detection 0":   LOG_DIR / "detection_0.log",
        "Detection 1":   LOG_DIR / "detection_1.log",
        "Detection 2":   LOG_DIR / "detection_2.log",
        "Relay":         LOG_DIR / "relay.log",
        "GPU Monitor":   LOG_DIR / "gpu_monitor.log",
    }

    try:
        while True:
            clear_screen()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{'='*60}")
            print(f"  Industrial Vision System — Health Dashboard  [{now}]")
            print(f"{'='*60}")

            for name, path in log_files.items():
                lines = read_last_lines(path, n=3)
                print(format_section(name, lines))

            print(f"\n{'='*60}")
            print("  [Refreshing every 5s — Ctrl+C to exit]")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nDashboard closed.")


if __name__ == "__main__":
    main()
