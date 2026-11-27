# CONTEXT: Phase 6 — Dataset Acquisition & Pipeline Verification

## Objective
Establish a clean, automated dataset fetching architecture and perform an immediate empirical verification of the complete ARGUS computer vision pipeline using a lightweight highway CCTV video stream.

## Core Technical Mandates (from docs/07_DATASET_ACQUISITION_GUIDE.md)
1. **Two-Phase Acquisition Rule:**
   - **Phase 1 (Immediate Smoke Test):** Zero-credential, 30MB highway CCTV sample video (`data/smoke_test/sample_traffic.mp4`) fetched automatically to verify Stages 1–4 (`stream_reader`, `detector`, `tracker`, `homography_speed`) in under 30 seconds.
   - **Phase 2 (Targeted Benchmarking/Fine-Tuning):** Authenticated CLI SDKs (`kaggle`, `roboflow`) pulling domain datasets (`detrac_yolo`, `indian_traffic_roboflow`, `veri_776`, `brno_compspeed`) on-demand.
2. **Directory & Storage Hygiene:**
   - All dataset directories live under `data/` and are strictly excluded via `.gitignore`.
   - Tooling scripts live under `scripts/`.
3. **Bounding-Box Coordinate Normalization:**
   - UA-DETRAC XML pixel format converted to normalized YOLO format $(x_{\text{center\_norm}}, y_{\text{center\_norm}}, w_{\text{norm}}, h_{\text{norm}}) \in [0.0, 1.0]$.
4. **End-to-End Pipeline Verification:**
   - Verified against `backend.main:app` with `CCTV_SOURCE="data/smoke_test/sample_traffic.mp4"`.

## Artifact Inventory
- `docs/07_DATASET_ACQUISITION_GUIDE.md` (Reference Architecture)
- `scripts/fetch_datasets.py` (CLI Downloader)
- `scripts/convert_detrac_to_yolo.py` (XML to YOLO Normalizer)
- `data/smoke_test/sample_traffic.mp4` (Test stream)

