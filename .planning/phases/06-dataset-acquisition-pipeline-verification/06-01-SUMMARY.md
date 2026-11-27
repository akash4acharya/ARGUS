---
phase: 06-dataset-acquisition-pipeline-verification
plan: "01"
subsystem: dataset-tooling
tags:
  - opencv
  - datasets
  - yolo
  - smoke-test
requires: []
provides:
  - Automated dataset downloader script `scripts/fetch_datasets.py`
  - UA-DETRAC XML-to-YOLO bounding-box coordinate normalizer `scripts/convert_detrac_to_yolo.py`
  - Instant 30MB smoke-test highway CCTV video `data/smoke_test/sample_traffic.mp4`
  - Dataset directory exclusion in `.gitignore`
affects:
  - "06-02"
actuals:
  tokens: 1800
  tasks: 2
  commits: 0
tech-stack:
  added:
    - urllib
    - xml.etree.ElementTree
  patterns:
    - "Zero-credential smoke-test tier for immediate local verification"
    - "Resolution-independent bounding-box coordinate normalization"
key-files:
  created:
    - scripts/fetch_datasets.py
    - scripts/convert_detrac_to_yolo.py
    - data/smoke_test/sample_traffic.mp4
  modified:
    - .gitignore
key-decisions:
  - "Used Intel IoT sample highway CCTV stream as universal zero-auth smoke test video"
  - "Standardized bounding box normalization formula: (x+w/2)/W, (y+h/2)/H, w/W, h/H clamped to [0.0, 1.0]"
patterns-established:
  - "Datasets isolated in data/ outside source tree and ignored via .gitignore"
requirements-completed:
  - FR-1
  - NFR-2
---

# Plan 06-01 Summary: Dataset Tooling & Smoke Test Acquisition

## Overview
Successfully set up the dataset acquisition tooling and downloaded the initial highway CCTV smoke-test video stream.

## Completed Tasks
1. **Tooling & Scripts:**
   - Created `scripts/fetch_datasets.py` with multi-tier download support (`smoke-test`, `detrac`, `roboflow-india`, `all`).
   - Created `scripts/convert_detrac_to_yolo.py` implementing mathematical coordinate normalization and UA-DETRAC XML sequence parsing.
   - Updated `.gitignore` to exclude `data/` and `graphify-out/`.
2. **Video Fetch & Verification:**
   - Fetched `data/smoke_test/sample_traffic.mp4` (2.68 MB).
   - Validated video integrity using OpenCV (`shape: (432, 768, 3)`).

