"""
test_smoke_pipeline.py
End-to-end smoke verification test for ARGUS pipeline on live sample video.
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.stream_reader import UniversalCCTVStream
from core.detector import YOLOv8VehicleDetector
from core.tracker import DeepSORTVehicleTracker
from core.homography_speed import HomographySpeedEngine
from core.hud_renderer import ExceptionFirstHudRenderer


def run_vision_smoke_test(video_path: str, max_frames: int = 120) -> None:
    print(f"\n[TEST 1] Testing Vision Pipeline on '{video_path}'...")
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    # Setup calibration points scaled to video resolution (768x432)
    src_points = [
        [150.0, 120.0],  # Top-left
        [620.0, 120.0],  # Top-right
        [720.0, 400.0],  # Bottom-right
        [80.0, 400.0],   # Bottom-left
    ]
    real_width_m = 10.0
    real_length_m = 40.0

    stream = UniversalCCTVStream(source_url=video_path).start()
    detector = YOLOv8VehicleDetector(confidence_threshold=0.25)
    tracker = DeepSORTVehicleTracker(max_age=30)
    speed_engine = HomographySpeedEngine(src_points, real_width_m, real_length_m)
    hud_renderer = ExceptionFirstHudRenderer(speed_limit_kmh=40.0)

    frames_processed = 0
    total_detections = 0
    total_tracks = 0
    last_frame_id = -1
    start_time = time.monotonic()

    try:
        # Give stream thread a brief moment to grab initial frame
        time.sleep(0.1)

        while frames_processed < max_frames:
            ts_frame = stream.get_latest()
            if ts_frame is None or ts_frame.image is None or ts_frame.frame_id == last_frame_id:
                time.sleep(0.005)
                continue

            last_frame_id = ts_frame.frame_id
            frame = ts_frame.image.copy()
            timestamp = ts_frame.capture_time

            # 1. Detection
            detections = detector.detect(frame)
            total_detections += len(detections)

            # 2. Tracking
            tracked_vehicles = tracker.update(detections, frame)
            total_tracks += len(tracked_vehicles)

            # 3. Homography & Speed
            speed_map = {}
            for v in tracked_vehicles:
                speed_kmh = speed_engine.compute_speed_kmh(
                    track_id=v.track_id,
                    tire_px=v.tire_point,
                    capture_time=timestamp,
                )
                speed_map[v.track_id] = speed_kmh

            # 4. Complete HUD Render
            rendered_frame = hud_renderer.render_overlay(
                frame=frame,
                src_points=src_points,
                vehicles=tracked_vehicles,
                speed_map=speed_map,
            )

            assert rendered_frame is not None
            assert rendered_frame.shape == frame.shape

            frames_processed += 1
            if frames_processed % 30 == 0 or frames_processed == max_frames:
                print(f"  Processed {frames_processed}/{max_frames} frames | Detected: {len(detections)} | Active Tracks: {len(tracked_vehicles)}")

    finally:
        stream.stop()

    elapsed = time.monotonic() - start_time
    fps = frames_processed / elapsed if elapsed > 0 else 0
    print(f"[OK] Vision pipeline verified: {frames_processed} frames processed in {elapsed:.2f}s ({fps:.1f} FPS)")
    print(f"[OK] Total cumulative detections: {total_detections}, Total track instances: {total_tracks}")
    assert frames_processed >= 30, f"Expected at least 30 frames, got {frames_processed}"


def run_fastapi_smoke_test(video_path: str) -> None:
    print(f"\n[TEST 2] Testing FastAPI Service Boot with CCTV_SOURCE='{video_path}'...")
    os.environ["CCTV_SOURCE"] = video_path

    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.status_code} {res_health.text}"
        health_data = res_health.json()
        print(f"[OK] /health -> {health_data}")
        assert health_data.get("status") == "healthy"

        # 2. Calibration endpoint
        res_calib = client.get("/api/calibration")
        assert res_calib.status_code == 200, f"Calibration get failed: {res_calib.status_code} {res_calib.text}"
        calib_data = res_calib.json()
        print(f"[OK] /api/calibration -> Source points: {len(calib_data.get('src_points', []))} points")
        assert "src_points" in calib_data
        assert "real_width_m" in calib_data

        # 3. Settings endpoint
        res_settings = client.get("/api/settings")
        assert res_settings.status_code == 200, f"Settings get failed: {res_settings.status_code}"
        print(f"[OK] /api/settings -> {res_settings.json()}")

        # 4. Telemetry endpoint
        res_telemetry = client.get("/api/telemetry")
        assert res_telemetry.status_code == 200, f"Telemetry get failed: {res_telemetry.status_code}"
        print(f"[OK] /api/telemetry -> {res_telemetry.json()}")

        # 5. Violations endpoint
        res_violations = client.get("/api/violations")
        assert res_violations.status_code == 200, f"Violations get failed: {res_violations.status_code} {res_violations.text}"
        print(f"[OK] /api/violations -> Status 200 OK")

    print("[OK] FastAPI endpoints verified successfully.")


if __name__ == "__main__":
    smoke_video = "data/smoke_test/sample_traffic.mp4"
    run_vision_smoke_test(smoke_video, max_frames=120)
    run_fastapi_smoke_test(smoke_video)
    print("\nALL SMOKE PIPELINE TESTS PASSED!")
