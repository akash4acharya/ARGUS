import os
import io
import csv
import cv2
import time
import yaml
import asyncio
import threading
from collections import deque
from typing import List, Dict, Any, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from core.stream_reader import UniversalCCTVStream
from core.detector import YOLOv8VehicleDetector
from core.tracker import DeepSORTVehicleTracker
from core.homography_speed import HomographySpeedEngine
from core.hud_renderer import ExceptionFirstHudRenderer
from backend.database import init_db
from backend.models import CalibrationSchema, SettingsSchema, TelemetryStatusSchema, CameraSwitchSchema
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

CAMERA_SOURCE = os.getenv("CCTV_SOURCE", CAMERA_CONF.get("source", 0))
if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)
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

CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "highway_expressway": {
        "id": "highway_expressway",
        "label": "6-Lane Expressway (MVI_40191)",
        "source": "data/demo/highway_expressway.mp4",
        "src_points": [[705.0, 260.0], [1140.0, 260.0], [1255.0, 680.0], [655.0, 680.0]],
        "real_width_m": 10.5,
        "real_length_m": 28.0,
        "speed_limit_kmh": 50.0,
    },
    "urban_arterial": {
        "id": "urban_arterial",
        "label": "8-Lane Urban Intersection (MVI_40712)",
        "source": "data/demo/urban_arterial.mp4",
        "src_points": [[390.0, 250.0], [630.0, 250.0], [560.0, 680.0], [90.0, 680.0]],
        "real_width_m": 11.0,
        "real_length_m": 32.0,
        "speed_limit_kmh": 50.0,
    },
    "overhead_closeup": {
        "id": "overhead_closeup",
        "label": "Close-Up Overhead (Sample)",
        "source": "data/smoke_test/sample_traffic.mp4",
        "src_points": [[480.0, 260.0], [800.0, 260.0], [1120.0, 680.0], [160.0, 680.0]],
        "real_width_m": 8.0,
        "real_length_m": 22.0,
        "speed_limit_kmh": 40.0,
    },
}

active_preset_id = "highway_expressway"
for _pid, _p in CAMERA_PRESETS.items():
    if str(CAMERA_SOURCE) == str(_p["source"]):
        active_preset_id = _pid
        break

# Global pipeline instances
stream_reader: Optional[UniversalCCTVStream] = None
stream_lock = threading.Lock()
detector: Optional[YOLOv8VehicleDetector] = None
tracker: Optional[DeepSORTVehicleTracker] = None
homography_engine: Optional[HomographySpeedEngine] = None
hud_renderer: Optional[ExceptionFirstHudRenderer] = None
violation_worker: Optional[AsyncViolationWorker] = None

# Thread-safe frame and telemetry storage for MJPEG & WebSockets
latest_annotated_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()
recent_logs: deque = deque(maxlen=150)
log_lock = threading.Lock()
last_log_emit: Dict[int, float] = {}

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

    prev_frame_time = 0.0
    last_frame_id = -1
    frame_count = 0
    fps_start_time = time.monotonic()
    fps_estimate = 25.0

    print("[INFO] Vision processing loop started.")

    while vision_thread_running:
        loop_start = time.monotonic()
        with stream_lock:
            reader = stream_reader
        frame_obj = reader.get_latest() if reader is not None else None

        if frame_obj is None or frame_obj.image is None:
            with telemetry_lock:
                telemetry_data["stream_status"] = "RECONNECTING"
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

        # Deduplicate identical video frames so repeated frames never compress delta_t
        if frame_obj.frame_id == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = frame_obj.frame_id

        raw_bgr = frame_obj.image.copy()
        capture_time = frame_obj.capture_time

        # Calculate time delta from deterministic/monotonic frame stamp
        delta_t_ms = (capture_time - prev_frame_time) * 1000.0 if prev_frame_time > 0 else 40.0
        prev_frame_time = capture_time

        # 1. Detection
        detections = detector.detect(raw_bgr)

        # 2. Tracking
        tracked_vehicles = tracker.update(detections, raw_bgr)

        # 3. Speed Calculation & Live Vehicle Log
        speed_map = {}
        now_wall = time.monotonic()
        active_limit = hud_renderer.speed_limit_kmh
        cam_label = CAMERA_PRESETS.get(active_preset_id, {}).get("label", str(CAMERA_SOURCE))

        for vehicle in tracked_vehicles:
            spd = homography_engine.compute_speed_kmh(
                vehicle.track_id, vehicle.tire_point, capture_time
            )
            speed_map[vehicle.track_id] = spd

            if spd is not None:
                status_str = "VIOLATION" if spd >= active_limit else ("WARNING" if spd >= active_limit * 0.85 else "NOMINAL")
                prev_emit = last_log_emit.get(vehicle.track_id, 0.0)
                if (now_wall - prev_emit) >= 0.75 or status_str == "VIOLATION":
                    if status_str != "VIOLATION" or (now_wall - prev_emit) >= 0.75:
                        last_log_emit[vehicle.track_id] = now_wall
                        log_entry = {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "camera": cam_label,
                            "track_id": vehicle.track_id,
                            "vehicle_class": vehicle.class_name,
                            "speed_kmh": round(spd, 1),
                            "limit_kmh": round(active_limit, 1),
                            "status": status_str,
                        }
                        with log_lock:
                            recent_logs.appendleft(log_entry)
                        if active_connections:
                            loop.call_soon_threadsafe(
                                asyncio.create_task,
                                broadcast_message({"type": "TRACK_LOG", "payload": log_entry}),
                            )

            # 4. Check for Speeding Violation
            if spd is not None and spd >= active_limit:
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
                    speed_limit_kmh=active_limit,
                    vehicle_crop_bgr=crop.copy(),
                )
                violation_worker.enqueue_from_vision_thread(loop, event)

        # 5. Render HUD Overlay (vector SVG handles Speed Zone overlay in dashboard)
        annotated = hud_renderer.render_overlay(
            frame=raw_bgr,
            src_points=homography_engine.src_polygon.tolist(),
            vehicles=tracked_vehicles,
            speed_map=speed_map,
            draw_zone=False,
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
        use_embedder=bool(TRACK_CONF.get("use_embedder", True)),
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


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "ARGUS CCTV Speed Monitor"}


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

        # Send recent vehicle telemetry logs
        with log_lock:
            hist_logs = list(recent_logs)[:40]
        for entry in reversed(hist_logs):
            await websocket.send_json({"type": "TRACK_LOG", "payload": entry})

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


@app.get("/api/cameras")
async def list_cameras():
    """Returns available demo camera presets and the currently active preset ID."""
    return {
        "active_preset_id": active_preset_id,
        "presets": [
            {
                "id": p["id"],
                "label": p["label"],
                "source": p["source"],
                "exists": os.path.exists(str(p["source"])),
            }
            for p in CAMERA_PRESETS.values()
        ],
    }


@app.post("/api/camera")
async def switch_camera(req: CameraSwitchSchema):
    """Hot-swaps the active video source and loads its calibrated Speed Zone."""
    global stream_reader, CAMERA_SOURCE, SPEED_LIMIT_KMH, active_preset_id
    preset = CAMERA_PRESETS.get(req.preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{req.preset_id}'")
    if not os.path.exists(str(preset["source"])):
        raise HTTPException(status_code=404, detail=f"Video file missing: {preset['source']}")

    new_reader = UniversalCCTVStream(
        source_url=preset["source"],
        target_width=CAMERA_WIDTH,
        target_height=CAMERA_HEIGHT,
    ).start()

    with stream_lock:
        old_reader = stream_reader
        stream_reader = new_reader
    if old_reader:
        old_reader.stop()

    active_preset_id = req.preset_id
    CAMERA_SOURCE = preset["source"]
    SPEED_LIMIT_KMH = float(preset["speed_limit_kmh"])

    if homography_engine:
        homography_engine.update_calibration(
            src_points=preset["src_points"],
            real_width_m=preset["real_width_m"],
            real_length_m=preset["real_length_m"],
        )
    if hud_renderer:
        hud_renderer.set_speed_limit(SPEED_LIMIT_KMH)
    if tracker:
        tracker.fallback_tracker.tracks.clear()
    with log_lock:
        recent_logs.clear()
        last_log_emit.clear()
    with telemetry_lock:
        telemetry_data["camera_source"] = str(CAMERA_SOURCE)
        telemetry_data["speed_limit_kmh"] = SPEED_LIMIT_KMH

    return {
        "status": "success",
        "active_preset_id": active_preset_id,
        "calibration": {
            "src_points": preset["src_points"],
            "real_width_m": preset["real_width_m"],
            "real_length_m": preset["real_length_m"],
        },
        "speed_limit_kmh": SPEED_LIMIT_KMH,
    }


@app.post("/api/calibration/reset")
async def reset_calibration():
    """Resets the Speed Zone polygon to the active camera preset's default."""
    preset = CAMERA_PRESETS.get(active_preset_id, CAMERA_PRESETS["highway_expressway"])
    if homography_engine:
        homography_engine.update_calibration(
            src_points=preset["src_points"],
            real_width_m=preset["real_width_m"],
            real_length_m=preset["real_length_m"],
        )
    return {
        "status": "success",
        "src_points": preset["src_points"],
        "real_width_m": preset["real_width_m"],
        "real_length_m": preset["real_length_m"],
    }


@app.get("/api/logs")
async def get_live_logs():
    """Returns recent structured per-vehicle telemetry logs."""
    with log_lock:
        return list(recent_logs)


@app.get("/api/logs/export.csv")
async def export_logs_csv():
    """Downloads the structured telemetry and violation log as a clean CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Camera", "Track_ID", "Vehicle_Class", "Speed_kmh", "Limit_kmh", "Status"])
    with log_lock:
        rows = list(recent_logs)
    for r in rows:
        writer.writerow([
            r.get("timestamp", ""),
            r.get("camera", ""),
            f"#{int(r.get('track_id', 0)):03d}",
            r.get("vehicle_class", ""),
            r.get("speed_kmh", ""),
            r.get("limit_kmh", ""),
            r.get("status", ""),
        ])
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=argus_speed_log.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    host = settings.get("server", {}).get("host", "0.0.0.0")
    port = int(settings.get("server", {}).get("port", 8000))
    print(f"[INFO] Starting ARGUS CCTV Speed Monitor server on http://{host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
