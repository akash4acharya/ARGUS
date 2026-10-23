# 03 — TECHNICAL SPECIFICATION SHEET

## 1. System Stack & Dependencies
- **Language:** Python 3.10+
- **Vision & Math:** `ultralytics>=8.2.0` (YOLOv8), `opencv-python>=4.9.0`, `deep-sort-realtime>=1.3.2`, `numpy>=1.26.0`, `scipy>=1.12.0`
- **Backend & Async I/O:** `fastapi>=0.110.0`, `uvicorn[standard]`, `sqlalchemy[asyncio]>=2.0`, `asyncpg`, `pydantic>=2.6`, `pyyaml`
- **Database:** PostgreSQL 15+ (with fallback for lightweight SQLite async in local testing)

---

## 2. Target Repository File Tree
```text
cctv-speed-monitor/
├── config/
│   └── settings.yaml            # RTSP URLs, 4-point pixel coords, metric dimensions, speed limit
├── core/
│   ├── __init__.py
│   ├── stream_reader.py         # Threaded RTSP/NVR frame grabber with monotonic timestamps
│   ├── detector.py              # YOLOv8 wrapper filtering vehicle classes [2, 3, 5, 7]
│   ├── tracker.py               # DeepSORT wrapper maintaining track_id continuity
│   ├── homography_speed.py      # IPM matrix solver, rolling deque buffer, km/h calculator
│   └── hud_renderer.py          # Exception-first HUD overlay (brackets, pill, polygon, anchor)
├── backend/
│   ├── __init__.py
│   ├── database.py              # Async SQLAlchemy engine & PostgreSQL session maker
│   ├── models.py                # ORM definitions for cameras, calibrations, and violations
│   ├── worker.py                # Background consumer saving crops & DB rows from queue
│   └── main.py                  # FastAPI server (MJPEG video stream, REST API, WebSockets)
├── static/
│   └── index.html               # Single-page operator dashboard & 4-point calibration UI
├── evidence_crops/              # Saved JPEG snapshots of speeding vehicles
├── test_homography.py           # Standalone homography verification script
└── requirements.txt
```
