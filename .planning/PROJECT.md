# PROJECT: ARGUS — CCTV Speed & Telemetry Overlay Suite

## 1. Executive Summary
ARGUS is a real-time computer vision and telemetry suite designed to piggyback on pre-existing CCTV camera infrastructure (NVRs, RTSP/ONVIF streams, webcams, and video archives). It delivers high-accuracy vehicle speed detection, trajectory tracking, planar homography translation, and asynchronous violation persistence without frame drop or visual clutter.

## 2. Core Visual & Technical Mandates
- **Exception-First Visual Hierarchy:** Nominal traffic blends quietly into the background via 20% corner-bracket bounding boxes and muted emerald tones (`#10B981`); vehicles approaching or exceeding the speed limit trigger amber (`#F59E0B`) or solid crimson alerts (`#EF4444`).
- **Zero-Buffer RTSP Stream Ingestion:** Frame grabbing is decoupled from AI inference inside an isolated daemon thread, stamping every incoming frame with `time.monotonic()` to guarantee accurate $\Delta t$ time-delta calculation.
- **Tire-Contact Ground Anchor Projection:** Uses $(u = (x_1+x_2)/2, v = y_2)$ bottom-center projection through a $3 \times 3$ Direct Linear Transform (DLT) planar homography matrix $H$ onto the $Z=0$ asphalt plane, eliminating truck/bus 3D height parallax errors.
- **Rolling Window Trajectory Smoothing & Jitter Rejection:** Accumulates $\ge 8$ valid frame samples across a 15-frame deque, coupled with a 5-sample rolling median filter to suppress 1–2px bounding-box jitter and prevent artificial speed spikes.
- **Asynchronous Producer-Consumer Persistence:** Employs an `asyncio.Queue` worker with pre-warmed database connection pooling (PostgreSQL with SQLite fallback) and disk snapshot writer to prevent disk/DB I/O from stalling the 30 FPS vision loop.
- **Interactive 3-Zone Web Dashboard:** Built with Vanilla CSS/JS and WebSockets, rendering live MJPEG annotated video, draggable 4-point homography calibration handles, real-time FPS/$\Delta t$ telemetry, and instant violation card alerts.

## 3. Technology Stack
- **Language:** Python 3.10+
- **Vision & Deep Learning:** Ultralytics YOLOv8, DeepSORT Realtime, OpenCV, NumPy, SciPy
- **Backend & Networking:** FastAPI, Uvicorn, WebSockets, Multipart MJPEG Streaming
- **Database:** SQLAlchemy 2.0 (AsyncIO), Asyncpg (PostgreSQL), Aiosqlite (Local Dev Fallback)
- **Frontend:** Single-page HTML5/CSS3/ES6 dashboard with JetBrains Mono monospace readouts
