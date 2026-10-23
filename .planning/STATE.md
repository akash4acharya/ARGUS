# STATE: Project Execution Memory

## Active Milestone: v1.0.0 — Initial Architecture Scaffold & Verification

### Status Summary
- **Scaffolding:** 100% Complete. All target modules defined in `03_SPEC_SHEET.md` and `06_EXPLAINER_PART3_ASYNC_BACKEND_AND_UI.md` have been generated and populated.
- **Documentation:** Standardized in `/docs/01_DESIGN_SYSTEM.md` through `/docs/06_EXPLAINER_PART3_ASYNC_BACKEND_AND_UI.md`.
- **Environment:** Dedicated `.venv` created via `uv` with Python 3.14.
- **Verification Script:** `test_homography.py` created to test DLT precision, inverse projection, zone boundary gating, speed physics, and jitter rejection.

### Key Architectural Decisions
1. **Zero-Buffer Capture:** Frame reader runs in isolated daemon thread; inference loop reads newest frame with `time.monotonic()` arrival stamp.
2. **Bottom-Center Projection:** Ground anchor projected strictly at $(u=(x_1+x_2)/2, v=y_2)$ to avoid 3D vehicle height parallax.
3. **Double-Stabilized Speed Engine:** 15-frame deque requiring $\ge 8$ samples with 5-sample median filtering.
4. **Database Resilience:** Async SQLAlchemy automatically falls back to SQLite (`cctv_speed_local.db`) if PostgreSQL is unavailable locally.
5. **WebSocket & MJPEG:** Zero-lag multipart JPEG streaming paired with WebSocket push for instantaneous violation notification.
