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

### Phase 6: Dataset Acquisition & Pipeline Verification (Completed)
- [x] Standardize dataset acquisition & fetching guide (`docs/07_DATASET_ACQUISITION_GUIDE.md`)
- [x] Automated CLI dataset downloader script (`scripts/fetch_datasets.py`)
- [x] UA-DETRAC XML-to-YOLO bounding-box coordinate normalizer (`scripts/convert_detrac_to_yolo.py`)
- [x] Fetch instant smoke-test traffic video clip (`data/smoke_test/sample_traffic.mp4`)
- [x] Verify full end-to-end pipeline on live video stream via FastAPI server (`test_smoke_pipeline.py`)

---

## Milestone 2: v2.0.0 — Model Training & Accuracy Validation

### Phase 7: UA-DETRAC Dataset Acquisition & YOLOv8 Fine-Tuning (Active)
- [x] Pull UA-DETRAC dataset via Kaggle CLI into `data/detrac_raw/` (9.23 GB original)
- [x] Convert all 60 UA-DETRAC XML sequences (82,085 frames) to normalized YOLO `.txt` labels
- [x] Partition 80/20 train/val split using zero-copy NTFS hard links (`data/detrac_yolo/`)
- [x] Generate `data.yaml` with classes `car`, `bus`, `others`
- [x] Setup CUDA PyTorch stack (`torch==2.11.0+cu128`, RTX 4060 verified)
- [x] Create GPU-accelerated training pipeline (`scripts/train_yolo.py`)
- [x] Create Zero-Stress Kaggle GPU Training Guide (`docs/08_KAGGLE_GPU_TRAINING_GUIDE.md`)
- [x] Create 1-Click Kaggle Training Notebook (`notebooks/argus_train_yolov8_kaggle.ipynb`)
- [x] Fine-tune YOLOv8n on UA-DETRAC overhead CCTV frames (Cloud Kaggle T4 30-ep mAP@50 = 0.765; Local RTX 4060 6-ep `best.pt` saved)
- [x] Evaluate trained model: mAP@50 (0.765 Kaggle / 0.565 local 6-ep), per-class precision/recall (`car` 0.941, `bus` 0.924)
- [x] Wire best checkpoint into `config/settings.yaml` as new default weights (`models/argus_detrac_yolov8n.pt`)
- [x] Regression test: re-run smoke-test pipeline with fine-tuned model (`test_smoke_pipeline.py` passed)
- [x] Document training results and model card in `docs/09_TRAINING_REPORT.md`

### Phase 7.5: Speed Physics Fix, 1:1 Manual Zone Setter & Clear Logs (Completed)
- [x] Deterministic video clock + frame deduplication + least-squares velocity regression (`core/stream_reader.py`, `core/homography_speed.py`)
- [x] Live Multi-Dataset Camera Switcher (`6-Lane Expressway`, `8-Lane Urban Intersection`, `Close-Up Overhead`)
- [x] Always-on 1:1 pixel-locked 4-corner Speed Zone drag setter (`static/index.html`)
- [x] Clear 2-tab Audit Panel (`Violations` with zoom lightbox + `Live Log` table) & CSV export (`/api/logs/export.csv`)

### Phase 8: Indian Traffic Multi-Class Expansion (Planned)
- [ ] Pull Indian CCTV Traffic dataset via Roboflow API
- [ ] Merge class taxonomy (auto-rickshaw, two-wheeler) into YOLO config
- [ ] Fine-tune on merged dataset with expanded class map
- [ ] Update `core/detector.py` class map for Indian vehicle types

### Phase 9: Speed Accuracy Validation with BrnoCompSpeed (Planned)
- [ ] Download BrnoCompSpeed Session 0 calibration data
- [ ] Build ground-truth comparison script against LiDAR speed readings
- [ ] Quantify homography engine accuracy: MAE, RMSE in km/h
- [ ] Publish accuracy report in `docs/09_SPEED_VALIDATION_REPORT.md`
