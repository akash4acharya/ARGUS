#!/usr/bin/env python3
"""
ARGUS CCTV Speed Monitor — Standalone Homography & Physics Verification Script.
Implements Skill 5: Modular Verification & Testing Protocol.

Verifies:
  1. Perspective transform matrix accuracy (floating-point error < 1e-4 meters).
  2. Inverse projection accuracy back into pixel coordinate space.
  3. Point-in-polygon zone boundary gating.
  4. Rolling window speed calculation under simulated 60 km/h vehicle trajectory.
  5. Robustness to bounding-box pixel jitter (median filter stability).
  6. Optional live/file stream ingestion test with --source argument.
"""

import sys
import time
import argparse
import numpy as np
import cv2

from core.homography_speed import HomographySpeedEngine
from core.stream_reader import UniversalCCTVStream


def test_homography_matrix_accuracy():
    print("=" * 70)
    print("TEST 1: 4-Point Homography Matrix DLT Precision (< 1e-4 m)")
    print("=" * 70)

    src_points = [
        [480.0, 260.0],   # P1: Top-Left
        [800.0, 260.0],   # P2: Top-Right
        [1120.0, 680.0],  # P3: Bottom-Right
        [160.0, 680.0],   # P4: Bottom-Left
    ]
    real_width_m = 8.0
    real_length_m = 35.0

    engine = HomographySpeedEngine(src_points, real_width_m, real_length_m)

    expected_corners = [
        (0.0, 0.0),
        (real_width_m, 0.0),
        (real_width_m, real_length_m),
        (0.0, real_length_m),
    ]

    max_forward_err = 0.0
    max_inverse_err = 0.0

    for i, (px, exp_m) in enumerate(zip(src_points, expected_corners)):
        calc_x_m, calc_y_m = engine.pixel_to_meters(tuple(px))
        err_m = float(np.hypot(calc_x_m - exp_m[0], calc_y_m - exp_m[1]))
        if err_m > max_forward_err:
            max_forward_err = err_m

        # Inverse test
        inv_u, inv_v = engine.meters_to_pixel(exp_m)
        err_px = float(np.hypot(inv_u - px[0], inv_v - px[1]))
        if err_px > max_inverse_err:
            max_inverse_err = err_px

        print(f"  Corner P{i+1}: Source px {px} -> Projected ({calc_x_m:8.5f} m, {calc_y_m:8.5f} m) [Error: {err_m:.2e} m]")

    print(f"  Max Forward Transformation Error: {max_forward_err:.3e} meters")
    print(f"  Max Inverse Transformation Error: {max_inverse_err:.3e} pixels")

    assert max_forward_err < 1e-4, f"Forward error {max_forward_err} exceeds 1e-4 tolerance!"
    assert max_inverse_err < 1e-3, f"Inverse error {max_inverse_err} exceeds 1e-3 tolerance!"
    print("  >>> PASSED: Matrix DLT precision verified.")
    return engine


def test_zone_boundary_gating(engine: HomographySpeedEngine):
    print("\n" + "=" * 70)
    print("TEST 2: Point-In-Polygon Zone Boundary Gating")
    print("=" * 70)

    # Center of trapezoid should be inside
    center_px = (640.0, 470.0)
    inside = engine.is_inside_zone(center_px)
    print(f"  Point {center_px} (Center): inside={inside} (Expected: True)")
    assert inside, f"Point {center_px} should be inside calibration polygon!"

    # Corner points should be inside/on boundary
    for pt in engine.src_polygon:
        in_corner = engine.is_inside_zone(tuple(pt))
        print(f"  Corner Point {tuple(pt)}: inside={in_corner} (Expected: True)")
        assert in_corner, f"Corner {tuple(pt)} should be inside/boundary!"

    # Out-of-bounds point (top-left sky)
    sky_px = (100.0, 50.0)
    outside = engine.is_inside_zone(sky_px)
    print(f"  Point {sky_px} (Sky/Outside): inside={outside} (Expected: False)")
    assert not outside, f"Point {sky_px} should be outside calibration polygon!"

    print("  >>> PASSED: Zone boundary gating verified.")


def test_simulated_speed_physics(engine: HomographySpeedEngine):
    print("\n" + "=" * 70)
    print("TEST 3: Trajectory Velocity Estimation (Simulated 60 km/h Vehicle)")
    print("=" * 70)

    # Target speed: 60 km/h = 16.6667 m/s
    target_kmh = 60.0
    target_mps = target_kmh / 3.6
    fps = 30.0
    dt = 1.0 / fps

    track_id = 42
    engine.trajectories.clear()
    engine.speed_buffer.clear()

    # Move vehicle along center lane (X = 4.0 m) from Y = 2.0 m to Y = 32.0 m
    # Total distance = 30 meters
    current_time = 1000.0
    y_current = 2.0
    x_center = 4.0

    calculated_speeds = []

    for step in range(35):
        current_time += dt
        y_current += target_mps * dt

        if y_current > engine.real_length_m:
            break

        # Convert simulated real metric position back to pixel coordinates
        px_u, px_v = engine.meters_to_pixel((x_center, y_current))

        speed = engine.compute_speed_kmh(track_id, (px_u, px_v), current_time)
        if speed is not None:
            calculated_speeds.append(speed)
            print(f"  Step {step:02d} (t={current_time:.3f}s): Y={y_current:5.2f}m -> Speed = {speed:5.2f} km/h")

    assert len(calculated_speeds) > 0, "No speed calculated! (Should have >= 8 samples)"
    steady_speed = calculated_speeds[-1]
    speed_error = abs(steady_speed - target_kmh)
    print(f"\n  Final Steady Speed: {steady_speed:.2f} km/h (Target: {target_kmh:.2f} km/h)")
    print(f"  Absolute Error: {speed_error:.3f} km/h ({(speed_error / target_kmh) * 100:.2f}%)")

    assert speed_error < 0.5, f"Speed error {speed_error} km/h is greater than 0.5 km/h tolerance!"
    print("  >>> PASSED: Physics velocity calculation verified.")


def test_jitter_rejection(engine: HomographySpeedEngine):
    print("\n" + "=" * 70)
    print("TEST 4: Jitter Rejection with Rolling Median Filter")
    print("=" * 70)

    track_id = 99
    engine.trajectories.clear()
    engine.speed_buffer.clear()

    target_kmh = 50.0
    target_mps = target_kmh / 3.6
    dt = 1.0 / 30.0
    current_time = 2000.0
    y_current = 3.0
    x_center = 4.0

    # Simulate random 1-2 pixel detector noise
    np.random.seed(42)
    speeds_with_jitter = []

    for step in range(30):
        current_time += dt
        y_current += target_mps * dt
        if y_current > engine.real_length_m:
            break

        clean_u, clean_v = engine.meters_to_pixel((x_center, y_current))
        # Add +- 1.5 pixel jitter
        jitter_u = clean_u + np.random.uniform(-1.5, 1.5)
        jitter_v = clean_v + np.random.uniform(-1.5, 1.5)

        spd = engine.compute_speed_kmh(track_id, (jitter_u, jitter_v), current_time)
        if spd is not None:
            speeds_with_jitter.append(spd)

    avg_speed = float(np.mean(speeds_with_jitter))
    std_speed = float(np.std(speeds_with_jitter))
    print(f"  Simulated 50.0 km/h with +-1.5px jitter:")
    print(f"  Mean Speed: {avg_speed:.2f} km/h | Std Dev: {std_speed:.2f} km/h")

    assert abs(avg_speed - target_kmh) < 2.5, f"Median filter variation {abs(avg_speed - target_kmh):.2f} exceeded 2.5 km/h!"
    print("  >>> PASSED: Jitter rejection verified.")


def test_optional_source_stream(source: str):
    print("\n" + "=" * 70)
    print(f"TEST 5: Stream Ingestion Protocol for source: {source}")
    print("=" * 70)

    stream = UniversalCCTVStream(source_url=source)
    try:
        stream.start()
        print("  Waiting 2 seconds for frames...")
        time.sleep(2.0)

        frame_obj = stream.get_latest()
        if frame_obj is None:
            print("  [WARN] No frames received from stream source.")
        else:
            print(f"  Received Frame #{frame_obj.frame_id}")
            print(f"  Capture Timestamp: {frame_obj.capture_time:.4f}")
            print(f"  Resolution: {frame_obj.image.shape[1]}x{frame_obj.image.shape[0]}")
            print("  >>> PASSED: Stream ingestion verified.")
    finally:
        stream.stop()


def main():
    parser = argparse.ArgumentParser(description="Homography & Speed Physics Test Suite")
    parser.add_argument("--source", type=str, default=None, help="Optional RTSP URL or video path to test ingestion")
    args = parser.parse_args()

    print("Running ARGUS CCTV Verification Suite...\n")
    engine = test_homography_matrix_accuracy()
    test_zone_boundary_gating(engine)
    test_simulated_speed_physics(engine)
    test_jitter_rejection(engine)

    if args.source:
        test_optional_source_stream(args.source)

    print("\n" + "#" * 70)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY (EXIT 0)")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
