"""
tests/simulate_failures.py
===========================
Failure simulation tests for supervisor recovery validation.
Run AFTER the system is deployed and running.

Tests:
1. Camera disconnect simulation
2. Detection process kill
3. Relay process kill
4. GUI crash
5. Memory pressure simulation

Usage:
    python tests/simulate_failures.py [--test camera|detection|relay|gui|memory]
"""

from __future__ import annotations
import argparse
import os
import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def find_process_by_name(name: str) -> list:
    """Find processes matching a name pattern."""
    try:
        import psutil
        result = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if name.lower() in cmdline.lower():
                    result.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return result
    except Exception as e:
        print(f"[ERROR] psutil error: {e}")
        return []


def simulate_camera_disconnect(camera_id: int = 0):
    """Kill a camera process to simulate camera disconnect."""
    print(f"\n[SIM] Simulating Camera {camera_id} disconnect...")
    procs = find_process_by_name(f"camera_{camera_id}")
    if not procs:
        print(f"[SIM] Camera {camera_id} process not found (may already be restarted)")
        return

    for proc in procs:
        print(f"[SIM] Killing Camera {camera_id} process PID={proc.pid}")
        try:
            proc.terminate()
            time.sleep(2)
            if proc.is_running():
                proc.kill()
        except Exception as e:
            print(f"[SIM] Error: {e}")

    print(f"[SIM] Camera {camera_id} process killed. Watching for supervisor recovery...")
    print("[SIM] Supervisor should restart camera within 10-15 seconds.")


def simulate_detection_crash(camera_id: int = 0):
    """Kill a detection process to simulate inference crash."""
    print(f"\n[SIM] Simulating Detection {camera_id} crash...")
    procs = find_process_by_name(f"detection_{camera_id}")
    if not procs:
        print(f"[SIM] Detection {camera_id} process not found")
        return

    for proc in procs:
        print(f"[SIM] Killing Detection {camera_id} PID={proc.pid}")
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except Exception as e:
            print(f"[SIM] Error: {e}")

    print("[SIM] Detection process killed. Supervisor should restart within 10-15 seconds.")


def simulate_relay_failure():
    """Kill the relay process to simulate USB failure."""
    print("\n[SIM] Simulating Relay process failure...")
    procs = find_process_by_name("relay_process")
    if not procs:
        procs = find_process_by_name("relay")

    if not procs:
        print("[SIM] Relay process not found")
        return

    for proc in procs:
        print(f"[SIM] Killing Relay PID={proc.pid}")
        try:
            proc.terminate()
        except Exception as e:
            print(f"[SIM] Error: {e}")

    print("[SIM] Relay process killed. Supervisor should restart it.")
    print("[SIM] Production (detection) should continue unaffected.")


def simulate_gui_crash():
    """Kill the GUI process to simulate display crash."""
    print("\n[SIM] Simulating GUI crash...")
    procs = find_process_by_name("gui_process")
    if not procs:
        procs = find_process_by_name("gui")

    if not procs:
        print("[SIM] GUI process not found")
        return

    for proc in procs:
        print(f"[SIM] Killing GUI PID={proc.pid}")
        try:
            proc.terminate()
        except Exception as e:
            print(f"[SIM] Error: {e}")

    print("[SIM] GUI killed. Production (detection + relay) should continue.")
    print("[SIM] Supervisor may restart GUI.")


def simulate_all_sequentially():
    """Run all failure simulations with recovery time between each."""
    print("\n" + "=" * 60)
    print("FULL FAILURE SIMULATION SEQUENCE")
    print("=" * 60)

    tests = [
        ("GUI crash", simulate_gui_crash),
        ("Relay failure", simulate_relay_failure),
        ("Camera 0 disconnect", lambda: simulate_camera_disconnect(0)),
        ("Detection 0 crash", lambda: simulate_detection_crash(0)),
    ]

    for test_name, test_fn in tests:
        print(f"\n[SIM] === {test_name} ===")
        test_fn()
        print(f"[SIM] Waiting 30 seconds for supervisor recovery...")
        for i in range(30, 0, -5):
            print(f"[SIM] {i}s remaining...")
            time.sleep(5)
        print(f"[SIM] Recovery window complete for: {test_name}")

    print("\n[SIM] All failure simulations complete.")
    print("[SIM] Review logs/supervisor.log for recovery events.")


def main():
    parser = argparse.ArgumentParser(description="Vision System Failure Simulator")
    parser.add_argument("--test", choices=["camera", "detection", "relay", "gui", "all"],
                        default="all", help="Which failure to simulate")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID for camera/detection tests")
    args = parser.parse_args()

    print("=" * 60)
    print("Industrial Vision System — Failure Simulator")
    print("=" * 60)
    print(f"Test: {args.test}")
    print("WARNING: This will intentionally kill production processes.")
    print("Supervisor MUST be running for recovery to work.")
    print()
    input("Press ENTER to continue, Ctrl+C to abort...")

    if args.test == "camera":
        simulate_camera_disconnect(args.camera_id)
    elif args.test == "detection":
        simulate_detection_crash(args.camera_id)
    elif args.test == "relay":
        simulate_relay_failure()
    elif args.test == "gui":
        simulate_gui_crash()
    elif args.test == "all":
        simulate_all_sequentially()


if __name__ == "__main__":
    main()
