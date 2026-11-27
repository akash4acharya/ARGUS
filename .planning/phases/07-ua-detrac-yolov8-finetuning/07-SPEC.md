# Phase 7: UA-DETRAC Dataset Acquisition & YOLOv8 Fine-Tuning — Specification

**Created:** 2026-11-23  
**Status:** In Progress (Cloud Execution Active)  
**Ambiguity Score:** 0.05 (Gate passed: ≤ 0.20)  
**Requirements:** 5 locked  

---

## Goal
Fine-tune YOLOv8n on the overhead CCTV UA-DETRAC benchmark using Kaggle's cloud T4 GPU accelerator, producing an evaluated `best.pt` checkpoint with mAP@50 > 0.65, and integrate it into ARGUS without local hardware thermal stress.

---

## Background
The baseline ARGUS perception engine (`core/detector.py`) uses `yolov8n.pt` with COCO weights. COCO consists of consumer photos shot at eye level. ARGUS operates on elevated pole-mounted and overhead CCTV streams where vehicles exhibit top-down foreshortening, roof prominence, and severe inter-vehicle occlusions. Fine-tuning on UA-DETRAC aligns bounding-box priors with real-world traffic camera perspectives.

---

## Requirements

1. **REQ-7.1 [Dataset Ingestion & YOLO Normalization]**
   - **Current:** Raw UA-DETRAC XML files store pixel corner coordinates.
   - **Target:** All 60 training sequences (82,085 frames) converted to normalized $(x_c, y_c, w, h) \in [0.0, 1.0]$ in standard YOLO format with 80/20 train/val sequence split.
   - **Acceptance:** `data/detrac_yolo/data.yaml` validates with Ultralytics and references train and validation splits. *(Status: COMPLETED)*

2. **REQ-7.2 [Cloud-Offloaded GPU Training]**
   - **Current:** Local training risks thermal throttling and battery wear on laptop hardware.
   - **Target:** Training executed in the cloud using Kaggle's dual Tesla T4 GPUs (16 GB VRAM) for 30 epochs with FP16 acceleration.
   - **Acceptance:** Kaggle outputs `argus_yolov8_best_weights.zip` containing `best.pt`. *(Status: ACTIVE)*

3. **REQ-7.3 [Empirical Model Evaluation]**
   - **Current:** Pretrained COCO baseline mAP on overhead angles is unmeasured.
   - **Target:** Validation mAP@50 exceeds 0.65 on the 16,001 UA-DETRAC validation frames across `car`, `bus`, and `others`.
   - **Acceptance:** Ultralytics validation summary confirms mAP@50 ≥ 0.65 and produces `results.png` & `confusion_matrix.png`.

4. **REQ-7.4 [ARGUS Runtime Integration]**
   - **Current:** `config/settings.yaml` points to `yolov8n.pt`.
   - **Target:** `config/settings.yaml` points to `models/argus_detrac_yolov8n.pt` with classes `[0, 1, 2]`.
   - **Acceptance:** `core/detector.py` loads `models/argus_detrac_yolov8n.pt` without exceptions.

5. **REQ-7.5 [Pipeline Regression Verification]**
   - **Current:** Pipeline passes smoke test on baseline COCO weights.
   - **Target:** Pipeline passes `test_smoke_pipeline.py` with fine-tuned model at ≥25 FPS.
   - **Acceptance:** `test_smoke_pipeline.py` exits with code 0.

---

## Boundaries

**In scope:**
- UA-DETRAC 60-sequence conversion and dataset manifest.
- Cloud execution on Kaggle Dual Tesla T4 GPUs.
- Export of `best.pt` and integration into `config/settings.yaml`.
- End-to-end smoke verification on sample traffic video.

**Out of scope:**
- Training DeepSORT ReID appearance embedder (uses existing cosine/IOU fallback).
- Adding Indian traffic classes (auto-rickshaws, two-wheelers) — scheduled for Phase 8.
- LiDAR speed validation — scheduled for Phase 9 (BrnoCompSpeed).

---

## Acceptance Criteria
- [x] Dataset conversion scripts authored and verified (`convert_detrac_to_yolo.py`, `organize_detrac_split.py`).
- [x] Cloud training guide (`docs/08_KAGGLE_GPU_TRAINING_GUIDE.md`) and notebook (`notebooks/argus_train_yolov8_kaggle.ipynb`) delivered.
- [ ] Training completed on Kaggle GPU T4; `best.pt` downloaded.
- [ ] Model evaluated with mAP@50 > 0.65.
- [ ] `config/settings.yaml` updated and verified via `test_smoke_pipeline.py`.
