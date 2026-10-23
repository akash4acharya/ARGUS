# ROADMAP: ARGUS CCTV Speed & Telemetry Suite

## Phase Structure

### Phase 1: Ingestion & Perception Engine (Completed)
- [x] Zero-buffer RTSP / CCTV stream ingestion (`core/stream_reader.py`)
- [x] YOLOv8 vehicle detection with class filtering (`core/detector.py`)
- [x] DeepSORT multi-object tracking with appearance embeddings (`core/tracker.py`)

### Phase 2: Planar Homography Math & Speed Physics (Completed)
- [x] 4-Point Homography Matrix DLT solver (`core/homography_speed.py`)
- [x] Bottom-center tire contact projection $(u, v) = ((x_1+x_2)/2, y_2)$
- [x] Point-in-polygon ray-casting zone gating
- [x] Rolling window ($\ge 8$ samples) + median filter trajectory speed calculator
- [x] Standalone verification test suite (`test_homography.py`)

### Phase 3: Exception-First HUD & Visualization (Completed)
- [x] Semantic color tokens and tabular monospace formatting
- [x] 20% corner-bracket bounding boxes for nominal traffic
- [x] Amber warning & crimson violation high-contrast state transitions
- [x] Ground-plane anchor dot and floating telemetry pill overlay (`core/hud_renderer.py`)
- [x] Semi-transparent 20% alpha calibration polygon overlay

### Phase 4: Async Backend & Persistence (Completed)
- [x] Asynchronous database engine with PostgreSQL & SQLite fallback (`backend/database.py`)
- [x] Speed violation ORM models and Pydantic schemas (`backend/models.py`)
- [x] Non-blocking worker queue for disk crops and database persistence (`backend/worker.py`)
- [x] FastAPI MJPEG streaming and WebSocket real-time broadcast (`backend/main.py`)

### Phase 5: Operator Dashboard UI (Completed)
- [x] Fixed 100vh 3-Zone layout (`static/index.html`)
- [x] Zone A: Top Telemetry Bar with live FPS, delta_t, limit adjust
- [x] Zone B: 70% Primary Video Canvas with interactive 4-point SVG drag handles
- [x] Zone C: 30% Live Violation Audit Feed with snapshot preview and excess badges
- [x] Dynamic calibration save & real-time WebSocket sync
