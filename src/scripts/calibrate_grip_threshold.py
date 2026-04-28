#!/usr/bin/env python3
"""Calibrate GRIP_LOAD_SUCCESS_THRESHOLD interactively.

Procedure:
  1. Operator clears the gripper jaws.
  2. Script closes the gripper on empty air, samples normalised_load
     a few times, and records ``empty_load``.
  3. Operator inserts a 2-inch cube; script closes again and records
     ``block_load``.
  4. Recommends a threshold midway (with a tunable margin) and warns
     if the signal-to-noise ratio is too low to trust.

Usage:
    python scripts/calibrate_grip_threshold.py
    python scripts/calibrate_grip_threshold.py --write-env
    python scripts/calibrate_grip_threshold.py --margin 0.4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Calibrate gripper load threshold")
    parser.add_argument("--samples", type=int, default=5, help="Reads per state")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="Fraction of (block-empty) above empty for the threshold")
    parser.add_argument("--write-env", action="store_true",
                        help="Update .env with the recommended threshold")
    parser.add_argument("--mock", action="store_true", help="Use mock gripper")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    from qa_cell_edge_agent.drivers.gripper import Gripper

    port = os.environ.get("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.environ.get("MYCOBOT_BAUD", "115200"))
    grip = Gripper(port=port, baud=baud, mock=args.mock)

    print("\n=== Gripper Load Threshold Calibration ===\n")

    # ── Empty close ─────────────────────────────────────────────
    input("Clear the gripper jaws of any object, then press ENTER... ")
    grip.open_gripper()
    time.sleep(0.5)
    grip.close_gripper()
    empty_loads = _sample(grip, args.samples)
    grip.open_gripper()
    empty_avg = sum(empty_loads) / len(empty_loads)
    print(f"  Empty close loads: {[round(v, 3) for v in empty_loads]}")
    print(f"  → empty_load mean = {empty_avg:.3f}\n")

    # ── Block close ─────────────────────────────────────────────
    input("Place a 2-inch cube between the jaws, then press ENTER... ")
    grip.close_gripper()
    block_loads = _sample(grip, args.samples)
    grip.open_gripper()
    block_avg = sum(block_loads) / len(block_loads)
    print(f"  Block close loads: {[round(v, 3) for v in block_loads]}")
    print(f"  → block_load mean = {block_avg:.3f}\n")

    # ── Threshold computation ──────────────────────────────────
    delta = block_avg - empty_avg
    print(f"  Δ (block - empty) = {delta:+.3f}")

    if delta < 0.10:
        print(f"\n  {FAIL} Signal too weak (Δ < 0.10).")
        print("    Check gripper firmware, cable, or calibrate gripper itself.")
        sys.exit(1)

    threshold = empty_avg + args.margin * delta
    print(f"  Recommended GRIP_LOAD_SUCCESS_THRESHOLD = {threshold:.3f} "
          f"(margin = {args.margin:.0%} of Δ)\n")

    if args.write_env:
        _write_env_var("GRIP_LOAD_SUCCESS_THRESHOLD", f"{threshold:.3f}")
        print(f"  {PASS} Updated {ENV_PATH}")
    else:
        print("  Add to .env:")
        print(f"    GRIP_LOAD_SUCCESS_THRESHOLD={threshold:.3f}\n")


def _sample(grip, n: int):
    samples = []
    for _ in range(n):
        time.sleep(0.15)
        samples.append(float(grip.read().normalized_load))
    return samples


def _write_env_var(key: str, value: str) -> None:
    if not os.path.isfile(ENV_PATH):
        with open(ENV_PATH, "w") as f:
            f.write(f"{key}={value}\n")
        return
    with open(ENV_PATH) as f:
        lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
