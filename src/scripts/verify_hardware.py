#!/usr/bin/env python3
"""Pre-flight hardware verification.

Run before calibration or starting the agent to confirm that the myCobot
serial connection and USB camera are working.

Usage:
    python scripts/verify_hardware.py
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

PASS = "\033[92m\u2713\033[0m"
FAIL = "\033[91m\u2717\033[0m"
WARN = "\033[93m!\033[0m"


def check_serial():
    """Check myCobot serial connection."""
    port = os.environ.get("MYCOBOT_PORT")
    baud = int(os.environ.get("MYCOBOT_BAUD", "115200"))

    # ── Discovery ────────────────────────────────────────────────
    if not port:
        try:
            import serial.tools.list_ports
        except ImportError:
            print(f"  {FAIL} pyserial not installed (pip install pyserial)")
            return False

        candidates = []
        known_ids = {(0x10C4, 0xEA60), (0x1A86, 0x7523), (0x0403, 0x6001)}
        for p in serial.tools.list_ports.comports():
            if p.vid is not None and (p.vid, p.pid) in known_ids:
                candidates.append((p.device, f"VID:PID {p.vid:04X}:{p.pid:04X}", p.description))

        if not candidates:
            import glob
            for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
                for dev in sorted(glob.glob(pattern)):
                    candidates.append((dev, "unknown VID:PID", "generic serial"))

        if not candidates:
            print(f"  {FAIL} No serial port found")
            print(f"      Fix: Plug in the myCobot via USB")
            print(f"      Fix: Check 'ls /dev/ttyUSB*' or 'ls /dev/ttyACM*'")
            return False

        if len(candidates) > 1:
            print(f"  {WARN} Multiple serial ports found:")
            for dev, vid_pid, desc in candidates:
                print(f"        {dev} ({vid_pid}, {desc})")
            print(f"      Set MYCOBOT_PORT in .env to choose explicitly")

        port = candidates[0][0]
        vid_info = candidates[0][1]
        print(f"  Serial port:  {port} ({vid_info})")
    else:
        print(f"  Serial port:  {port} (from MYCOBOT_PORT env)")

    # ── Verify communication ─────────────────────────────────────
    try:
        from pymycobot import MyCobot280
    except ImportError:
        print(f"  {FAIL} pymycobot not installed (pip install 'pymycobot>=3.6.0')")
        return False

    try:
        mc = MyCobot280(port, baud)
        time.sleep(0.5)
    except Exception as exc:
        print(f"  {FAIL} Cannot open {port}: {exc}")
        if "Permission denied" in str(exc) or "Errno 13" in str(exc):
            print(f"      Fix: sudo usermod -aG dialout $USER && logout/login")
        elif "No such file" in str(exc):
            print(f"      Fix: Check cable connection, try 'ls /dev/ttyUSB*'")
        return False

    # Test angles
    try:
        angles = mc.get_angles()
        if angles and len(angles) == 6:
            angles_str = ", ".join(f"{a:.1f}" for a in angles)
            print(f"  {PASS} myCobot:    responding (angles: [{angles_str}])")
        else:
            print(f"  {WARN} myCobot:    connected but get_angles() returned {angles}")
            print(f"      This may mean firmware needs updating or arm is in error state")
    except Exception as exc:
        print(f"  {FAIL} myCobot:    get_angles() failed: {exc}")
        return False

    # Test gripper
    try:
        grip_val = mc.get_gripper_value()
        print(f"  {PASS} Gripper:    responding (value: {grip_val})")
    except Exception as exc:
        print(f"  {WARN} Gripper:    get_gripper_value() failed: {exc}")

    # Test temperatures
    try:
        temps = mc.get_joints_temperature()
        if temps and len(temps) >= 6:
            temps_str = ", ".join(f"{t:.1f}" for t in temps[:6])
            print(f"  {PASS} Temps:      [{temps_str}] \u00b0C")
        else:
            print(f"  {WARN} Temps:      get_joints_temperature() returned {temps}")
    except Exception as exc:
        print(f"  {WARN} Temps:      get_joints_temperature() failed: {exc}")

    return True


def check_camera():
    """Check USB camera."""
    try:
        import cv2
    except ImportError:
        print(f"  {FAIL} OpenCV not installed (pip install opencv-python-headless)")
        return False

    idx_env = os.environ.get("CAMERA_DEVICE_INDEX")
    if idx_env is not None:
        indices = [int(idx_env)]
        print(f"  Camera index: {idx_env} (from CAMERA_DEVICE_INDEX env)")
    else:
        import glob
        video_devs = sorted(glob.glob("/dev/video*"))
        indices = []
        for dev in video_devs:
            try:
                indices.append(int(dev.replace("/dev/video", "")))
            except ValueError:
                pass
        if not indices:
            indices = [0]

    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  {PASS} Camera:     /dev/video{idx} ({w}x{h}, working)")
            return True

    print(f"  {FAIL} No working camera found")
    print(f"      Fix: Check USB camera is plugged in")
    print(f"      Fix: Try 'ls /dev/video*' and set CAMERA_DEVICE_INDEX in .env")
    return False


def check_calibration():
    """Check if calibration files exist."""
    waypoints_path = os.path.join(
        os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json"
    )
    camera_cal_path = os.path.join(
        os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "camera_calibration.json"
    )

    if os.path.isfile(waypoints_path):
        print(f"  {PASS} Waypoints:  {os.path.basename(waypoints_path)} found")
    else:
        print(f"  {WARN} Waypoints:  not calibrated (will use defaults)")
        print(f"      Run: python scripts/calibrate_arm.py")

    if os.path.isfile(camera_cal_path):
        print(f"  {PASS} Camera cal: {os.path.basename(camera_cal_path)} found")
    else:
        print(f"  {WARN} Camera cal: not calibrated (vision-guided pick disabled)")
        print(f"      Run: python scripts/calibrate_camera.py (optional)")


def check_model():
    """Check if model file exists."""
    model_path = os.environ.get("MODEL_PATH", "./models/yolov5n.onnx")
    if os.path.isfile(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  {PASS} Model:      {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"  {WARN} Model:      {model_path} not found (will use mock inference)")
        print(f"      Run: python scripts/download_model.py")


def main():
    load_dotenv()

    print(f"\n  Hardware Verification")
    print("=" * 60)

    serial_ok = check_serial()
    print()
    camera_ok = check_camera()
    print()
    check_calibration()
    print()
    check_model()

    print("=" * 60)
    if serial_ok and camera_ok:
        print(f"  {PASS} All hardware checks passed.")
    elif not serial_ok and not camera_ok:
        print(f"  {FAIL} Serial and camera both failed — check connections.")
    elif not serial_ok:
        print(f"  {FAIL} Serial failed — camera OK. Fix serial before calibrating.")
    else:
        print(f"  {FAIL} Camera failed — serial OK. Fix camera before running agent.")
    print()


if __name__ == "__main__":
    main()
