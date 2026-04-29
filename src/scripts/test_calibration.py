#!/usr/bin/env python3
"""Test camera calibration accuracy.

Places the arm at known positions and shows where the camera thinks
they are (and vice versa). Helps diagnose if the homography is correct.

Usage:
    python scripts/test_calibration.py
"""

import os
import sys
import time
import json

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qa_cell_edge_agent.config.settings import Settings
from qa_cell_edge_agent.drivers.transforms import CameraTransform
from qa_cell_edge_agent.drivers.block_detector import BlockDetector
from qa_cell_edge_agent.drivers.workspace import WorkspaceMonitor


def _compute_rz(detected_angle: float, offset: float) -> float:
    """Compute gripper rz: negate for overhead mirror, normalize to [-45, 45]."""
    rz = -detected_angle + offset
    while rz > 45:
        rz -= 90
    while rz < -45:
        rz += 90
    return rz


def main():
    load_dotenv()
    settings = Settings()

    # Load calibration
    transform = CameraTransform()
    if not transform.is_calibrated:
        print("ERROR: No camera calibration found. Run calibrate_camera.py first.")
        sys.exit(1)

    print(f"Calibration mode: {transform._mode}")
    if transform._mode == "homography":
        print(f"Homography matrix:\n{transform._H}")
    print(f"Z pick height: {transform._z_pick} mm")
    print(f"Approach angles: {transform._approach_angles}")
    print()

    # Connect to robot
    try:
        from pymycobot import MyCobot280
    except ImportError:
        print("ERROR: pymycobot not installed")
        sys.exit(1)

    port = os.getenv("MYCOBOT_PORT", "/dev/ttyTHS1")
    baud = int(os.getenv("MYCOBOT_BAUD", "1000000"))
    mc = MyCobot280(port, baud)
    time.sleep(0.5)

    # Open camera
    cap = cv2.VideoCapture(settings.camera_device_index)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    print("=" * 60)
    print("  Camera Calibration Test")
    print("=" * 60)
    print()
    print("This test does two things:")
    print("  1. You click a point in the camera → shows robot coords it maps to")
    print("  2. Move arm to a position → shows where camera thinks it is")
    print()
    print("Instructions:")
    print("  - Click anywhere in the camera view to see predicted robot XY")
    print("  Testing:")
    print("    'm' — move arm to last clicked point")
    print("    'g' — go to detected block automatically")
    print("    'r' — read current robot coords")
    print("  Rotation:")
    print("    '+'/'-' — adjust offset by 1 degree")
    print("    ']'/'[' — adjust offset by 5 degrees")
    print("  Calibration:")
    print("    'c' — record current arm position as calibration point")
    print("           (move arm to a spot, press 'c', click that spot in camera)")
    print("    'w' — write new calibration from recorded points (need 4+)")
    print("  Other:")
    print("    'h' — move arm to SAFE_ABOVE (home safe)")
    print("    's' — release servos")
    print("    'q' — quit")
    print()

    block_detector = BlockDetector()
    workspace = WorkspaceMonitor()
    rotation_offset = float(os.environ.get("CAMERA_ROTATION_OFFSET", "0"))
    print(f"  Current CAMERA_ROTATION_OFFSET: {rotation_offset}°")

    clicked_points = []
    last_target = None
    cal_points_robot = []   # [[x, y], ...] from robot
    cal_points_pixel = []   # [[px, py], ...] from clicks
    waiting_for_click = False  # True when 'c' was pressed and waiting for click

    def on_click(event, x, y, flags, param):
        nonlocal last_target, waiting_for_click
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))

            if waiting_for_click:
                # Recording calibration point
                cal_points_pixel.append([x, y])
                waiting_for_click = False
                n = len(cal_points_robot)
                print(f"  Calibration point {n} recorded: pixel ({x}, {y}) → robot {cal_points_robot[-1]}")
                print(f"  Total points: {n} (need 4+ to write calibration)")
                return

            target = transform.pixel_to_robot(x, y)
            last_target = target
            print(f"  Pixel ({x}, {y}) → Robot ({target.coords[0]:.1f}, {target.coords[1]:.1f}) "
                  f"dist={target.distance_from_base:.1f}mm reachable={target.reachable}")

    cv2.namedWindow("Calibration Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Test", 960, 720)
    cv2.setMouseCallback("Calibration Test", on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Draw clicked points and their robot coords
        for px, py in clicked_points[-10:]:
            target = transform.pixel_to_robot(px, py)
            color = (0, 255, 0) if target.reachable else (0, 0, 255)
            cv2.circle(display, (px, py), 6, color, -1)
            label = f"({target.coords[0]:.0f}, {target.coords[1]:.0f})"
            cv2.putText(display, label, (px + 10, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Detect block and show rotation
        zone_mask = workspace._zone_mask if workspace.is_configured else None
        detection = block_detector.detect(frame, zone_mask=zone_mask)
        if detection is not None:
            # Draw rotated bbox
            box_points = cv2.boxPoints(detection.rotated_bbox).astype(int)
            cv2.drawContours(display, [box_points], 0, (0, 255, 0), 2)
            # Show detected angle + corrected angle
            corrected = _compute_rz(detection.rotation_angle, rotation_offset)
            cv2.putText(display, f"Detected: {detection.rotation_angle:.0f} deg | Corrected (rz): {corrected:.0f} deg | Offset: {rotation_offset:.0f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw recorded calibration points
        for i, (px, py) in enumerate(cal_points_pixel):
            cv2.circle(display, (px, py), 8, (255, 0, 255), -1)
            cv2.putText(display, str(i+1), (px+10, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        if waiting_for_click:
            cv2.putText(display, "CLICK the gripper tip position now!",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            info = f"G:go | M:click-move | C:calibrate | W:write({len(cal_points_robot)}pts) | +/-:rot | Q:quit"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Calibration Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            if last_target and last_target.reachable:
                x, y = last_target.coords[0], last_target.coords[1]
                z = transform._z_pick
                rx, ry, rz = transform._approach_angles
                move_coords = [x, y, z, rx, ry, rz]

                # Mimic the pick-and-place approach sequence
                import json as _json
                wp_file = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json")
                approach_z = float(os.environ.get("APPROACH_HEIGHT_MM", "160"))

                # 1. Go to SAFE_ABOVE
                if os.path.isfile(wp_file):
                    with open(wp_file) as _f:
                        _wp = _json.load(_f)
                    if "SAFE_ABOVE" in _wp:
                        print(f"\n  Step 1: SAFE_ABOVE...")
                        mc.sync_send_angles(_wp["SAFE_ABOVE"]["angles"], 30, timeout=15)

                # 2. Approach above target (high Z) — use detected rotation if available
                grip_rz = rz
                if detection is not None:
                    grip_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                    print(f"  Gripper rz: {grip_rz:.1f}° (detected={detection.rotation_angle:.1f} + offset={rotation_offset:.1f})")
                approach_coords = [x, y, approach_z, rx, ry, grip_rz]
                print(f"  Step 2: Approach above ({x:.1f}, {y:.1f}, z={approach_z:.0f})...")
                mc.send_coords(approach_coords, 40, 0)
                deadline = time.time() + 10
                while time.time() < deadline:
                    if mc.is_in_position(approach_coords, 1) == 1:
                        break
                    time.sleep(0.1)

                # 3. Descend to pick height
                move_coords = [x, y, z, rx, ry, grip_rz]
                print(f"  Step 3: Descend to z={z:.0f}...")
                mc.send_coords(move_coords, 25, 0)
                deadline = time.time() + 10
                while time.time() < deadline:
                    if mc.is_in_position(move_coords, 1) == 1:
                        break
                    time.sleep(0.1)

                actual = mc.get_coords()
                if isinstance(actual, list) and len(actual) >= 3:
                    err_x = abs(actual[0] - x)
                    err_y = abs(actual[1] - y)
                    print(f"  Arrived: x={actual[0]:.1f}, y={actual[1]:.1f}, z={actual[2]:.1f}")
                    print(f"  Error: dx={err_x:.1f}mm, dy={err_y:.1f}mm")
                    if err_x < 5 and err_y < 5:
                        print(f"  GOOD — within 5mm accuracy")
                    else:
                        print(f"  CHECK — look if gripper tip is at the clicked spot")
                else:
                    print(f"  Could not verify position (timeout or IK failure)")
                print()
            elif last_target:
                print(f"  Cannot move — target is outside workspace")
            else:
                print(f"  Click a point first, then press 'm'")
        elif key == ord('g'):
            if detection is not None:
                bbox = detection.bounding_box
                px = int(bbox[0] + bbox[2] / 2)
                py = int(bbox[1] + bbox[3] / 2)
                target = transform.pixel_to_robot(px, py)

                if not target.reachable:
                    print(f"  Block at pixel ({px}, {py}) → robot ({target.coords[0]:.1f}, {target.coords[1]:.1f}) — UNREACHABLE")
                else:
                    x, y = target.coords[0], target.coords[1]
                    z = transform._z_pick
                    rx, ry, rz = transform._approach_angles
                    grip_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                    approach_z = float(os.environ.get("APPROACH_HEIGHT_MM", "160"))

                    print(f"\n  Block: {detection.dominant_color} at pixel ({px}, {py})")
                    print(f"  → Robot ({x:.1f}, {y:.1f}) rz={grip_rz:.1f}°")

                    import json as _json
                    wp_file = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json")

                    # 1. Lift to current position's Z at safe height first
                    current_coords = mc.get_coords()
                    if isinstance(current_coords, list) and len(current_coords) == 6:
                        lift_z = approach_z + 40  # extra clearance
                        lift_coords = [current_coords[0], current_coords[1], lift_z,
                                       current_coords[3], current_coords[4], current_coords[5]]
                        print(f"  Step 1: Lifting to z={lift_z:.0f}...")
                        mc.send_coords(lift_coords, 30, 0)
                        time.sleep(2)

                    # 2. SAFE_ABOVE
                    if os.path.isfile(wp_file):
                        with open(wp_file) as _f:
                            _wp = _json.load(_f)
                        if "SAFE_ABOVE" in _wp:
                            print(f"  Step 2: SAFE_ABOVE...")
                            mc.sync_send_angles(_wp["SAFE_ABOVE"]["angles"], 30, timeout=15)

                    # 3. Approach above target
                    approach_coords = [x, y, approach_z, rx, ry, grip_rz]
                    print(f"  Step 3: Above target ({x:.1f}, {y:.1f}, z={approach_z:.0f}) rz={grip_rz:.0f}°...")
                    mc.send_coords(approach_coords, 40, 0)
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        if mc.is_in_position(approach_coords, 1) == 1:
                            break
                        time.sleep(0.1)

                    print(f"  PAUSED above target.")
                    print(f"  Controls: +/-: rotate 1° | ]/[: rotate 5°")
                    print(f"            Arrow keys: nudge XY by 5mm")
                    print(f"            'd': descend | 'q': abort")

                    # Wait for user to adjust and press 'd' to descend
                    NUDGE_MM = 5.0
                    while True:
                        ret2, frame2 = cap.read()
                        if ret2:
                            disp2 = frame2.copy()
                            cv2.putText(disp2, "ABOVE TARGET: d=descend | +/-=rot | arrows=nudge XY",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(disp2, f"pos=({x:.1f}, {y:.1f}) rz={grip_rz:.0f} offset={rotation_offset:.0f}",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.imshow("Calibration Test", disp2)

                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('d'):
                            break
                        elif k == ord('q'):
                            print("  Aborted descent")
                            break
                        elif k == ord('+') or k == ord('='):
                            rotation_offset += 1
                            grip_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  rz={grip_rz:.0f}° (offset={rotation_offset:.0f})")
                        elif k == ord('-'):
                            rotation_offset -= 1
                            grip_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  rz={grip_rz:.0f}° (offset={rotation_offset:.0f})")
                        elif k == ord(']'):
                            rotation_offset += 5
                            grip_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  rz={grip_rz:.0f}° (offset={rotation_offset:.0f})")
                        elif k == ord('['):
                            rotation_offset -= 5
                            grip_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  rz={grip_rz:.0f}° (offset={rotation_offset:.0f})")
                        # Arrow keys (OpenCV key codes)
                        elif k == 82 or k == 0:  # Up arrow
                            x += NUDGE_MM
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  nudge X+ → ({x:.1f}, {y:.1f})")
                        elif k == 84 or k == 1:  # Down arrow
                            x -= NUDGE_MM
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  nudge X- → ({x:.1f}, {y:.1f})")
                        elif k == 81 or k == 2:  # Left arrow
                            y += NUDGE_MM
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  nudge Y+ → ({x:.1f}, {y:.1f})")
                        elif k == 83 or k == 3:  # Right arrow
                            y -= NUDGE_MM
                            mc.send_coords([x, y, approach_z, rx, ry, grip_rz], 20, 0)
                            print(f"  nudge Y- → ({x:.1f}, {y:.1f})")

                    if k == ord('d'):
                        # 4. Descend
                        move_coords = [x, y, z, rx, ry, grip_rz]
                        print(f"  Step 4: Descending to z={z:.0f}...")
                        mc.send_coords(move_coords, 25, 0)
                        deadline = time.time() + 10
                        while time.time() < deadline:
                            if mc.is_in_position(move_coords, 1) == 1:
                                break
                            time.sleep(0.1)

                    actual = mc.get_coords()
                    if isinstance(actual, list) and len(actual) >= 3:
                        print(f"  Arrived: x={actual[0]:.1f}, y={actual[1]:.1f}, z={actual[2]:.1f}")
                    print()
            else:
                print("  No block detected — place a block in the zone first")
        elif key == ord(']'):
            rotation_offset += 5
            print(f"  CAMERA_ROTATION_OFFSET = {rotation_offset:.0f}° (+5)")
        elif key == ord('['):
            rotation_offset -= 5
            print(f"  CAMERA_ROTATION_OFFSET = {rotation_offset:.0f}° (-5)")
        elif key == ord('+') or key == ord('='):
            rotation_offset += 1
            print(f"  CAMERA_ROTATION_OFFSET = {rotation_offset:.0f}°")
            # If arm is at a position, rotate J6 to preview the new angle
            try:
                current = mc.get_angles()
                if isinstance(current, list) and len(current) == 6 and detection is not None:
                    new_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                    current_coords = mc.get_coords()
                    if isinstance(current_coords, list) and len(current_coords) == 6:
                        current_coords[5] = new_rz
                        mc.send_coords(current_coords, 20, 0)
                        print(f"  Rotating gripper to rz={new_rz:.0f}° (preview)")
            except Exception:
                pass
            print(f"  Set in .env: CAMERA_ROTATION_OFFSET={rotation_offset:.0f}")
        elif key == ord('-'):
            rotation_offset -= 1
            print(f"  CAMERA_ROTATION_OFFSET = {rotation_offset:.0f}°")
            try:
                current = mc.get_angles()
                if isinstance(current, list) and len(current) == 6 and detection is not None:
                    new_rz = _compute_rz(detection.rotation_angle, rotation_offset)
                    current_coords = mc.get_coords()
                    if isinstance(current_coords, list) and len(current_coords) == 6:
                        current_coords[5] = new_rz
                        mc.send_coords(current_coords, 20, 0)
                        print(f"  Rotating gripper to rz={new_rz:.0f}° (preview)")
            except Exception:
                pass
            print(f"  Set in .env: CAMERA_ROTATION_OFFSET={rotation_offset:.0f}")
        elif key == ord('c'):
            # Record robot position, then wait for click to record pixel
            coords = mc.get_coords()
            if isinstance(coords, list) and len(coords) >= 2:
                cal_points_robot.append([coords[0], coords[1]])
                waiting_for_click = True
                print(f"\n  Robot at ({coords[0]:.1f}, {coords[1]:.1f}) — now CLICK the gripper tip in the camera")
            else:
                print(f"  Could not read robot coords: {coords}")
        elif key == ord('w'):
            if len(cal_points_robot) < 4:
                print(f"  Need at least 4 points, have {len(cal_points_robot)}. Press 'c' to add more.")
            else:
                # Compute homography and save
                src = np.array(cal_points_pixel, dtype=np.float64)
                dst = np.array(cal_points_robot, dtype=np.float64)
                H, mask = cv2.findHomography(src, dst)

                z_pick = transform._z_pick
                approach = transform._approach_angles

                cal_data = {
                    "mode": "homography",
                    "homography_matrix": H.tolist(),
                    "z_pick_mm": z_pick,
                    "approach_angles": approach,
                    "num_calibration_points": len(cal_points_robot),
                    "pixel_points": cal_points_pixel,
                    "robot_points": cal_points_robot,
                }

                cal_file = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "camera_calibration.json")
                with open(cal_file, "w") as f:
                    json.dump(cal_data, f, indent=2)

                # Reload transform
                transform._H = np.array(H, dtype=np.float64)
                print(f"\n  NEW CALIBRATION SAVED with {len(cal_points_robot)} points!")
                print(f"  File: {cal_file}")
                print(f"  Homography matrix:\n{H}")
                print(f"  Test by clicking or pressing 'g' to verify accuracy.")
                print()
        elif key == ord('h'):
            import json as _json
            wp_file = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "waypoints.json")
            if os.path.isfile(wp_file):
                with open(wp_file) as _f:
                    _wp = _json.load(_f)
                if "SAFE_ABOVE" in _wp:
                    print("  Moving to SAFE_ABOVE...")
                    mc.sync_send_angles(_wp["SAFE_ABOVE"]["angles"], 30, timeout=15)
                    print("  Done.")
                else:
                    print("  SAFE_ABOVE waypoint not found")
            else:
                print("  waypoints.json not found")
        elif key == ord('s'):
            mc.release_all_servos()
            print("  Servos released — arm is free to move manually")
            print("  Press 'r' to read position after repositioning")
            print()
        elif key == ord('r'):
            coords = mc.get_coords()
            angles = mc.get_angles()
            if isinstance(coords, list) and len(coords) >= 3:
                print(f"\n  Robot position: x={coords[0]:.1f}, y={coords[1]:.1f}, z={coords[2]:.1f}")
                print(f"  Robot angles: {[f'{a:.1f}' for a in angles]}")
                print(f"  Distance from base: {(coords[0]**2 + coords[1]**2)**0.5:.1f} mm")
                print()
            else:
                print(f"  Could not read coords: {coords}")

    cap.release()
    cv2.destroyAllWindows()

    # ── Save settings on exit ────────────────────────────────────
    original_offset = float(os.environ.get("CAMERA_ROTATION_OFFSET", "0"))
    changes = []

    if rotation_offset != original_offset:
        changes.append(f"CAMERA_ROTATION_OFFSET={rotation_offset:.0f}")

    if changes:
        print("\n" + "=" * 60)
        print("  Settings changed during this session:")
        for c in changes:
            print(f"    {c}")

        # Update .env file
        env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.isfile(env_file):
            with open(env_file) as f:
                env_lines = f.readlines()

            updated = set()
            new_lines = []
            for line in env_lines:
                replaced = False
                for change in changes:
                    key = change.split("=")[0]
                    if line.strip().startswith(key + "=") or line.strip().startswith("# " + key):
                        new_lines.append(change + "\n")
                        updated.add(key)
                        replaced = True
                        break
                if not replaced:
                    new_lines.append(line)

            # Append any keys not found in .env
            for change in changes:
                key = change.split("=")[0]
                if key not in updated:
                    new_lines.append("\n" + change + "\n")

            with open(env_file, "w") as f:
                f.writelines(new_lines)
            print(f"\n  Updated {env_file}")
        else:
            print(f"\n  No .env file found — add these manually:")
            for c in changes:
                print(f"    {c}")

        print("=" * 60)
    else:
        print("\n  No settings changed.")

    # Summary: show calibration points
    cal_file = os.path.join(os.path.dirname(__file__), "..", "qa_cell_edge_agent", "drivers", "camera_calibration.json")
    if os.path.isfile(cal_file):
        with open(cal_file) as f:
            cal = json.load(f)
        if "pixel_points" in cal and "robot_points" in cal:
            print(f"\n  Camera calibration: {len(cal['pixel_points'])} points")
            for i, (px, ry) in enumerate(zip(cal["pixel_points"], cal["robot_points"])):
                print(f"    {i+1}. Pixel {px} → Robot {ry}")


if __name__ == "__main__":
    main()
