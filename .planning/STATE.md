# STATE: Project Execution Memory

## Active Milestone: v2.0.0 — Model Training & Accuracy Validation

### Status Summary
- **Previous Milestone (v1.0.0):** 100% Complete. All 6 phases delivered — scaffold, math, HUD, backend, dashboard, dataset tooling.
- **Active Phase:** Phase 7.5 — Speed Physics Fix, 1:1 Manual Zone Setter & Clear Logs (Completed)
- **Phase Status:** Demo-Ready & Live Verified (`http://127.0.0.1:8000` running at `24.5 FPS`, `delta_t_ms = 40.0 ms`, all 3 Exception-First states active)
- **Hardware Strategy:**
  - **Training:** Kaggle 30-epoch benchmark (`mAP@50 = 0.765`, `0.941 car`, `0.924 bus`) documented in `docs/09_TRAINING_REPORT.md` + local RTX 4060 6-epoch fine-tune (`models/argus_detrac_yolov8n.pt`)
  - **Inference:** Local NVIDIA GeForce RTX 4060 Laptop GPU (`device: "0"`, `use_embedder: false` fast tracker)
- **Demo Streams Ready:**
  - `data/demo/highway_expressway.mp4` (`MVI_40191`, 6-lane divided highway, active default)
  - `data/demo/urban_arterial.mp4` (`MVI_40712`, 8-lane unseen intersection)
  - `data/smoke_test/sample_traffic.mp4` (Close-up overhead)
- **Artifacts Generated:**
  - `models/argus_detrac_yolov8n.pt` (Fine-tuned overhead CCTV weights wired into `config/settings.yaml`)
  - `docs/09_TRAINING_REPORT.md` (Full 30-epoch empirical training & validation report)
  - `docs/08_KAGGLE_GPU_TRAINING_GUIDE.md` (Comprehensive step-by-step UI & setup walkthrough)
  - `notebooks/argus_train_yolov8_kaggle.ipynb` (Ready-to-upload 1-click training notebook)
  - `data/detrac_yolo/data.yaml`
  - `scripts/convert_detrac_to_yolo.py`
  - `scripts/organize_detrac_split.py`
  - `scripts/train_yolo.py`

## Session Continuity
- **Last session:** 2026-11-26 11:13 IST
- **Status:** All 4 locked design decisions implemented & verified (`test_homography.py` 0.000 km/h error, `test_smoke_pipeline.py` passed, live API + camera hot-swap verified).
- **Next immediate step:** Refresh `http://127.0.0.1:8000` in your browser to interact with the upgraded 3-Zone dashboard.

### Key Architectural Decisions (Inherited from v1.0.0)
1. **Zero-Buffer Capture:** Frame reader runs in isolated daemon thread; inference loop reads newest frame with `time.monotonic()` arrival stamp.
2. **Bottom-Center Projection:** Ground anchor projected strictly at $(u=(x_1+x_2)/2, v=y_2)$ to avoid 3D vehicle height parallax.
3. **Double-Stabilized Speed Engine:** 15-frame deque requiring $\ge 8$ samples with 5-sample median filtering.
4. **Database Resilience:** Async SQLAlchemy automatically falls back to SQLite (`cctv_speed_local.db`) if PostgreSQL is unavailable locally.
5. **WebSocket & MJPEG:** Zero-lag multipart JPEG streaming paired with WebSocket push for instantaneous violation notification.

### New Decisions (v2.0.0)
6. **Training Target:** Fine-tune YOLOv8n (pretrained COCO) on overhead CCTV frames from UA-DETRAC for superior elevated-camera detection.
7. **Local GPU Training:** RTX 4060 (8 GB VRAM), ~50 epochs, batch size 16, no cloud GPU needed.
