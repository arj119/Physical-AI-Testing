#!/usr/bin/env python3
"""Camera-to-robot calibration for vision-guided picking.

Supports two methods:
  1. Homography (default) — point the camera at the workspace, place a marker
     at 4+ known robot positions, record pixel↔robot correspondences, and
     compute a 2D homography.  Best for flat-surface pick-and-place.
  2. Hand-eye — attach an ArUco marker to the end-effector, move to 10+
     poses, and solve the AX=XB problem via cv2.calibrateHandEye.

Usage:
    python scripts/calibrate_camera.py                  # homography (default)
    python scripts/calibrate_camera.py --method handeye # full hand-eye
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "camera_calibration.json"
)


def _connect_robot():
    """Connect to the myCobot 280 and return the instance."""
    try:
        from pymycobot import MyCobot280
    except ImportError:
        print("ERROR: pymycobot not installed. Run: pip install 'pymycobot>=3.6.0,<5.0.0'")
        sys.exit(1)

    port = os.getenv("MYCOBOT_PORT", "/dev/ttyUSB0")
    baud = int(os.getenv("MYCOBOT_BAUD", "115200"))
    print(f"Connecting to myCobot on {port} @ {baud}...")
    try:
        mc = MyCobot280(port, baud)
        time.sleep(0.5)
    except Exception as exc:
        print(f"ERROR: Cannot open serial port {port}: {exc}")
        if "Permission denied" in str(exc):
            print("  Fix: sudo usermod -aG dialout $USER && logout/login")
        print("  Run 'python scripts/verify_hardware.py' first.")
        sys.exit(1)

    # Validate communication
    try:
        angles = mc.get_angles()
        if not angles or len(angles) != 6:
            print(f"WARNING: myCobot returned angles={angles} — may not be responding")
    except Exception as exc:
        print(f"ERROR: Cannot communicate with myCobot: {exc}")
        sys.exit(1)

    return mc


def _open_camera():
    """Open the USB camera and verify it captures frames."""
    idx = int(os.getenv("CAMERA_DEVICE_INDEX", "0"))
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {idx}")
        print("  Run 'python scripts/verify_hardware.py' to check camera.")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        print(f"ERROR: Camera at index {idx} opened but cannot capture frames")
        print("  Fix: Check camera USB connection. Try a different CAMERA_DEVICE_INDEX.")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Camera opened: index {idx}, {w}x{h}")
    return cap


# ── Homography calibration ────────────────────────────────────────────


def calibrate_homography(num_points: int = 6) -> dict:
    """Interactive homography calibration.

    The user places a visible marker at known robot positions and clicks
    the corresponding pixel location in the camera feed.
    """
    mc = _connect_robot()
    cap = _open_camera()

    print(f"\n=== Homography Calibration ({num_points} points, min 4 required) ===")
    print("For each point:")
    print("  1. Move the arm so the end-effector tip touches the workspace surface")
    print("  2. Press ENTER to record the robot position")
    print("  3. Click the corresponding point in the camera window")
    print("Press Ctrl+C to abort without saving.\n")

    robot_points = []  # [[x, y], ...]
    pixel_points = []  # [[cx, cy], ...]
    clicked = []

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", _on_click)

    for i in range(num_points):
        input(f"  [{i+1}/{num_points}] Move arm to point, press ENTER...")
        coords = mc.get_coords()
        if not coords or len(coords) < 3:
            print("    WARNING: Could not read robot coords — skipping")
            continue
        robot_points.append([coords[0], coords[1]])
        print(f"    Robot: x={coords[0]:.1f}, y={coords[1]:.1f}, z={coords[2]:.1f}")

        print("    Now click the same point in the camera image...")
        clicked.clear()
        while not clicked:
            ret, frame = cap.read()
            if ret:
                display = frame.copy()
                for px, py in pixel_points:
                    cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
                cv2.imshow("Calibration", display)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC to abort
                print("Aborted.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)

        px, py = clicked[0]
        pixel_points.append([px, py])
        print(f"    Pixel: ({px}, {py})\n")

    cap.release()
    cv2.destroyAllWindows()

    if len(robot_points) < 4:
        print("ERROR: Need at least 4 points for homography")
        sys.exit(1)

    # Compute homography: pixel → robot
    src = np.array(pixel_points, dtype=np.float64)
    dst = np.array(robot_points, dtype=np.float64)
    H, mask = cv2.findHomography(src, dst)

    z_pick = float(input("Enter the Z height of the pick surface (mm): "))
    rx = float(input("Enter approach Rx angle (degrees, typically 180 for downward): ") or "180")
    ry = float(input("Enter approach Ry angle (degrees, typically 0): ") or "0")
    rz = float(input("Enter approach Rz angle (degrees, typically 0): ") or "0")

    return {
        "mode": "homography",
        "homography_matrix": H.tolist(),
        "z_pick_mm": z_pick,
        "approach_angles": [rx, ry, rz],
        "num_calibration_points": len(robot_points),
        "pixel_points": pixel_points,
        "robot_points": robot_points,
    }


# ── Hand-eye calibration ─────────────────────────────────────────────


def calibrate_hand_eye(num_poses: int = 12) -> dict:
    """Eye-to-hand calibration using an ArUco marker on the end-effector."""
    mc = _connect_robot()
    cap = _open_camera()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()

    print(f"\n=== Hand-Eye Calibration ({num_poses} poses) ===")
    print("Attach an ArUco marker (DICT_4X4_50, ID 0) to the end-effector.")
    print("Move the arm to different poses. The marker must be visible in each.\n")

    marker_size = float(input("Enter ArUco marker size (mm): ") or "30")

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    # Get camera intrinsics (approximate if not calibrated)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    fx = fy = w  # rough estimate
    cx_cam, cy_cam = w / 2, h / 2
    camera_matrix = np.array([
        [fx, 0, cx_cam],
        [0, fy, cy_cam],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(5)

    print(f"Using approximate camera matrix (fx=fy={fx}, cx={cx_cam}, cy={cy_cam})")
    print("For better accuracy, provide a proper camera calibration.\n")

    for i in range(num_poses):
        input(f"  [{i+1}/{num_poses}] Move arm to a new pose, press ENTER...")
        coords = mc.get_coords()
        if not coords or len(coords) < 6:
            print("    WARNING: Could not read robot coords — skipping")
            continue

        ret, frame = cap.read()
        if not ret:
            print("    WARNING: Camera read failed — skipping")
            continue

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=detector_params)
        if ids is None or 0 not in ids:
            print("    WARNING: ArUco marker not detected — skipping")
            continue

        idx = list(ids.flatten()).index(0)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[idx:idx+1], marker_size, camera_matrix, dist_coeffs
        )

        R_target, _ = cv2.Rodrigues(rvecs[0])
        t_target = tvecs[0].reshape(3, 1)

        # Robot gripper pose in base frame
        x, y, z, rx_deg, ry_deg, rz_deg = coords
        R_gripper = _euler_to_rotation_matrix(
            np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
        )
        t_gripper = np.array([[x], [y], [z]], dtype=np.float64)

        R_gripper2base_list.append(R_gripper)
        t_gripper2base_list.append(t_gripper)
        R_target2cam_list.append(R_target)
        t_target2cam_list.append(t_target)

        print(f"    Pose recorded (robot: [{x:.1f}, {y:.1f}, {z:.1f}], marker detected)")

    cap.release()

    if len(R_gripper2base_list) < 3:
        print("ERROR: Need at least 3 valid poses")
        sys.exit(1)

    print(f"\nComputing hand-eye transform from {len(R_gripper2base_list)} poses...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.HAND_EYE_TSAI,
    )

    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()

    z_pick = float(input("Enter the Z height of the pick surface (mm): "))
    rx = float(input("Enter approach Rx angle (degrees): ") or "180")
    ry = float(input("Enter approach Ry angle (degrees): ") or "0")
    rz = float(input("Enter approach Rz angle (degrees): ") or "0")

    return {
        "mode": "hand_eye",
        "T_camera_to_base": T.tolist(),
        "camera_matrix": camera_matrix.tolist(),
        "z_pick_mm": z_pick,
        "approach_angles": [rx, ry, rz],
        "num_calibration_poses": len(R_gripper2base_list),
    }


def _euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (radians, XYZ convention) to a 3×3 rotation matrix."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Camera-to-robot calibration")
    parser.add_argument(
        "--method", choices=["homography", "handeye"], default="homography",
        help="Calibration method (default: homography)",
    )
    parser.add_argument(
        "--points", type=int, default=6,
        help="Number of calibration points/poses (default: 6)",
    )
    args = parser.parse_args()

    if args.method == "homography":
        result = calibrate_homography(num_points=args.points)
    else:
        result = calibrate_hand_eye(num_poses=args.points)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibration saved to {OUTPUT_FILE}")
    print("The arm driver will use this for vision-guided picking.")


if __name__ == "__main__":
    main()
