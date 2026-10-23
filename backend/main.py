import os
import cv2
import time
import yaml
import asyncio
import threading
from typing import List, Dict, Any, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from core.stream_reader import UniversalCCTVStream
from core.detector import YOLOv8VehicleDetector
from core.tracker import DeepSORTVehicleTracker
from core.homography_speed import HomographySpeedEngine
from core.hud_renderer import ExceptionFirstHudRenderer
from backend.database import init_db
from backend.models import CalibrationSchema, SettingsSchema, TelemetryStatusSchema
from backend.worker import AsyncViolationWorker, ViolationEvent


# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")


def load_settings() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to read {CONFIG_PATH}: {e}")
    return {}


settings = load_settings()

# Default settings fallbacks
CAMERA_CONF = settings.get("camera", {})
CALIB_CONF = settings.get("calibration", {})
DET_CONF = settings.get("detector", {})
TRACK_CONF = settings.get("tracker", {})
STORAGE_CONF = settings.get("storage", {})

CAMERA_SOURCE = CAMERA_CONF.get("source", 0)
CAMERA_WIDTH = CAMERA_CONF.get("width", 1280)
CAMERA_HEIGHT = CAMERA_CONF.get("height", 720)
SPEED_LIMIT_KMH = float(CAMERA_CONF.get("speed_limit_kmh", 40.0))

SRC_POINTS = CALIB_CONF.get(
    "src_points",
    [[480.0, 260.0], [800.0, 260.0], [1120.0, 680.0], [160.0, 680.0]],
)
REAL_WIDTH_M = float(CALIB_CONF.get("real_width_m", 8.0))
REAL_LENGTH_M = float(CALIB_CONF.get("real_length_m", 35.0))
EVIDENCE_DIR = STORAGE_CONF.get("evidence_dir", "evidence_crops")

# Global pipeline instances
stream_reader: Optional[UniversalCCTVStream] = None
detector: Optional[YOLOv8VehicleDetector] = None
tracker: Optional[DeepSORTVehicleTracker] = None
homography_engine: Optional[HomographySpeedEngine] = None
hud_renderer: Optional[ExceptionFirstHudRenderer] = None
violation_worker: Optional[AsyncViolationWorker] = None

# Thread-safe frame and telemetry storage for MJPEG & WebSockets
latest_annotated_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()

telemetry_data = {
    "fps": 0.0,
    "delta_t_ms": 0.0,
    "active_tracks": 0,
    "speed_limit_kmh": SPEED_LIMIT_KMH,
    "camera_source": str(CAMERA_SOURCE),
    "stream_status": "CONNECTING",
}
telemetry_lock = threading.Lock()

# WebSocket active client registry
active_connections: List[WebSocket] = []
main_loop: Optional[asyncio.AbstractEventLoop] = None
vision_thread_running = False


async def broadcast_message(message: dict) -> None:
    """Broadcasts a JSON packet to all connected WebSocket clients."""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_connections:
            active_connections.remove(ws)


def vision_processing_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Dedicated AI Vision Inference & Tracking Loop.
    Executes Skill 1, 2, 3, 4:
      - Reads newest frame with arrival timestamp from UniversalCCTVStream.
      - Executes YOLOv8 vehicle detection.
      - Maintains object tracks with DeepSORT.
      - Projects bottom-center tire contact dots through Homography matrix.
      - Computes smoothed km/h speed with rolling window and median filter.
      - Renders Exception-First HUD.
      - Enqueues speed violations into non-blocking worker queue.
    """
    global latest_annotated_jpeg, vision_thread_running

    prev_frame_time = time.monotonic()
    frame_count = 0
    fps_start_time = time.monotonic()
    fps_estimate = 30.0

    print("[INFO] Vision processing loop started.")

    while vision_thread_running:
        loop_start = time.monotonic()
        frame_obj = stream_reader.get_latest()

        if frame_obj is None or frame_obj.image is None:
            with telemetry_lock:
                telemetry_data["stream_status"] = "RECONNECTING"
            # Generate a standby frame with diagnostic banner
            standby = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                standby,
                "CONNECTING TO CCTV STREAM...",
                (CAMERA_WIDTH // 2 - 250, CAMERA_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            ret, encoded = cv2.imencode(".jpg", standby)
            if ret:
                with jpeg_lock:
                    latest_annotated_jpeg = encoded.tobytes()
            time.sleep(0.04)
            continue

        raw_bgr = frame_obj.image.copy()
        capture_time = frame_obj.capture_time

        # Calculate time delta from arrival time
        delta_t_ms = (capture_time - prev_frame_time) * 1000.0 if prev_frame_time > 0 else 33.3
        prev_frame_time = capture_time

        # 1. Detection
        detections = detector.detect(raw_bgr)

        # 2. Tracking
        tracked_vehicles = tracker.update(detections, raw_bgr)

        # 3. Speed Calculation
        speed_map = {}
        for vehicle in tracked_vehicles:
            spd = homography_engine.compute_speed_kmh(
                vehicle.track_id, vehicle.tire_point, capture_time
            )
            speed_map[vehicle.track_id] = spd

            # 4. Check for Speeding Violation
            if spd is not None and spd >= homography_engine.speed_limit_kmh if hasattr(homography_engine, "speed_limit_kmh") else (spd is not None and spd >= hud_renderer.speed_limit_kmh):
                # Crop vehicle for evidence
                x1, y1, x2, y2 = [int(v) for v in vehicle.bbox]
                h_img, w_img = raw_bgr.shape[:2]
                pad = 12
                crop_x1 = max(0, x1 - pad)
                crop_y1 = max(0, y1 - pad)
                crop_x2 = min(w_img, x2 + pad)
                crop_y2 = min(h_img, y2 + pad)

                crop = raw_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

                event = ViolationEvent(
                    camera_id=1,
                    track_id=vehicle.track_id,
                    vehicle_class=vehicle.class_name,
                    measured_speed_kmh=spd,
                    speed_limit_kmh=hud_renderer.speed_limit_kmh,
                    vehicle_crop_bgr=crop.copy(),
                )
                violation_worker.enqueue_from_vision_thread(loop, event)

        # 5. Render HUD Overlay
        annotated = hud_renderer.render_overlay(
            frame=raw_bgr,
            src_points=homography_engine.src_polygon.tolist(),
            vehicles=tracked_vehicles,
            speed_map=speed_map,
        )

        # Compress to JPEG for Web MJPEG Stream
        ret, encoded_jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with jpeg_lock:
                latest_annotated_jpeg = encoded_jpg.tobytes()

        # Update FPS telemetry
        frame_count += 1
        now = time.monotonic()
        if (now - fps_start_time) >= 1.0:
            fps_estimate = frame_count / (now - fps_start_time)
            frame_count = 0
            fps_start_time = now

            with telemetry_lock:
                telemetry_data["fps"] = round(fps_estimate, 1)
                telemetry_data["delta_t_ms"] = round(delta_t_ms, 1)
                telemetry_data["active_tracks"] = len(tracked_vehicles)
                telemetry_data["stream_status"] = "LIVE"

        # Pace loop if processing was faster than ~30 FPS
        elapsed = time.monotonic() - loop_start
        if elapsed < 0.025:
            time.sleep(0.025 - elapsed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_reader, detector, tracker, homography_engine, hud_renderer
    global violation_worker, vision_thread_running, main_loop

    main_loop = asyncio.get_running_loop()

    # 1. Initialize Database Schema
    await init_db()

    # 2. Initialize Core Components
    stream_reader = UniversalCCTVStream(
        source_url=CAMERA_SOURCE,
        target_width=CAMERA_WIDTH,
        target_height=CAMERA_HEIGHT,
    ).start()

    detector = YOLOv8VehicleDetector(
        model_weights=DET_CONF.get("model_weights", "yolov8n.pt"),
        confidence_threshold=float(DET_CONF.get("confidence_threshold", 0.35)),
        iou_threshold=float(DET_CONF.get("iou_threshold", 0.45)),
        target_classes=DET_CONF.get("classes", [2, 3, 5, 7]),
        device=DET_CONF.get("device", "cpu"),
    )

    tracker = DeepSORTVehicleTracker(
        max_age=int(TRACK_CONF.get("max_age", 30)),
        n_init=int(TRACK_CONF.get("n_init", 3)),
        nms_max_overlap=float(TRACK_CONF.get("nms_max_overlap", 1.0)),
        max_cosine_distance=float(TRACK_CONF.get("max_cosine_distance", 0.3)),
        nn_budget=int(TRACK_CONF.get("nn_budget", 100)),
    )

    homography_engine = HomographySpeedEngine(
        src_points=SRC_POINTS,
        real_width_m=REAL_WIDTH_M,
        real_length_m=REAL_LENGTH_M,
        window_size=15,
        min_samples=8,
    )

    hud_renderer = ExceptionFirstHudRenderer(speed_limit_kmh=SPEED_LIMIT_KMH)
    violation_worker = AsyncViolationWorker(evidence_dir=EVIDENCE_DIR)

    # 3. Start Background Violation Consumer
    worker_task = asyncio.create_task(violation_worker.consume_forever(broadcast_message))

    # 4. Start Vision Thread
    vision_thread_running = True
    vision_thread = threading.Thread(
        target=vision_processing_loop, args=(main_loop,), daemon=True, name="VisionInferenceThread"
    )
    vision_thread.start()

    yield

    # Clean shutdown
    vision_thread_running = False
    worker_task.cancel()
    if stream_reader:
        stream_reader.stop()


app = FastAPI(
    title="ARGUS CCTV Speed & Telemetry Suite",
    description="Real-time CCTV Speed Monitoring with Planar Homography, DeepSORT & Exception-First HUD",
    version="1.0.0",
    lifespan=lifespan,
)

# Static and Evidence Directories
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
EVIDENCE_ABS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), EVIDENCE_DIR)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(EVIDENCE_ABS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/evidence", StaticFiles(directory=EVIDENCE_ABS_DIR), name="evidence")


@app.get("/")
async def get_index():
    """Serves the single-page operator dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "ARGUS CCTV Speed Monitor is running. Static UI not found."})


def generate_mjpeg_stream():
    """Generator for continuous multipart/x-mixed-replace JPEG video streaming."""
    while True:
        with jpeg_lock:
            frame_bytes = latest_annotated_jpeg

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.033)


@app.get("/api/stream/video")
async def video_stream():
    """Zero-latency MJPEG video streaming endpoint."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pushing real-time speed violation alerts and telemetry ticks to the UI.
    """
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Send initial state
        with telemetry_lock:
            current_telemetry = dict(telemetry_data)
        await websocket.send_json({"type": "TELEMETRY", "payload": current_telemetry})

        # Send recent violations history
        if violation_worker:
            for item in reversed(violation_worker.recent_violations):
                await websocket.send_json({"type": "VIOLATION", "payload": item})

        while True:
            # Keep socket open and handle incoming ping or client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/api/calibration")
async def get_calibration():
    """Returns current 4-point homography calibration coordinates and metric dimensions."""
    if homography_engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {
        "src_points": homography_engine.src_polygon.tolist(),
        "real_width_m": homography_engine.real_width_m,
        "real_length_m": homography_engine.real_length_m,
    }


@app.post("/api/calibration")
async def update_calibration(calib: CalibrationSchema):
    """
    Dynamically updates the 4-point homography calibration without restarting the video stream.
    """
    if homography_engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    try:
        homography_engine.update_calibration(
            src_points=calib.src_points,
            real_width_m=calib.real_width_m,
            real_length_m=calib.real_length_m,
        )
        return {
            "status": "success",
            "message": "Calibration matrix updated successfully",
            "src_points": calib.src_points,
            "real_width_m": calib.real_width_m,
            "real_length_m": calib.real_length_m,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/settings")
async def get_settings():
    """Returns active speed limit and camera parameters."""
    with telemetry_lock:
        limit = telemetry_data.get("speed_limit_kmh", SPEED_LIMIT_KMH)
        src = telemetry_data.get("camera_source", str(CAMERA_SOURCE))
    return {
        "speed_limit_kmh": limit,
        "camera_source": src,
    }


@app.post("/api/settings")
async def update_settings(new_settings: SettingsSchema):
    """Updates speed limit threshold or camera source dynamically."""
    global SPEED_LIMIT_KMH
    if new_settings.speed_limit_kmh is not None:
        SPEED_LIMIT_KMH = new_settings.speed_limit_kmh
        if hud_renderer:
            hud_renderer.set_speed_limit(SPEED_LIMIT_KMH)
        with telemetry_lock:
            telemetry_data["speed_limit_kmh"] = SPEED_LIMIT_KMH

    return {
        "status": "success",
        "speed_limit_kmh": SPEED_LIMIT_KMH,
    }


@app.get("/api/telemetry", response_model=TelemetryStatusSchema)
async def get_telemetry():
    """Returns real-time processing statistics."""
    with telemetry_lock:
        return TelemetryStatusSchema(**telemetry_data)


@app.get("/api/violations")
async def get_violations():
    """Returns recent violations from memory buffer."""
    if violation_worker:
        return violation_worker.recent_violations
    return []


if __name__ == "__main__":
    import uvicorn
    host = settings.get("server", {}).get("host", "0.0.0.0")
    port = int(settings.get("server", {}).get("port", 8000))
    print(f"[INFO] Starting ARGUS CCTV Speed Monitor server on http://{host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
