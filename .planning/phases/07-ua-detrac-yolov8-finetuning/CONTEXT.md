# CONTEXT: Phase 7 — UA-DETRAC Dataset Acquisition & YOLOv8 Fine-Tuning

## Objective
Pull the UA-DETRAC overhead CCTV vehicle detection dataset, convert its XML annotations to normalized YOLO format, and fine-tune YOLOv8n on this data to dramatically improve detection accuracy at elevated camera angles typical of traffic surveillance installations.

## Why This Matters
The pretrained `yolov8n.pt` (COCO weights) was trained on natural-perspective photographs — people holding cameras at eye level. CCTV cameras are mounted 5–10 meters overhead at steep downward angles. Vehicles look very different from this perspective: foreshortened, top-of-hood/roof dominant, with partial occlusion. Fine-tuning on UA-DETRAC (which is specifically collected from overhead CCTV) teaches the model what vehicles actually look like from ARGUS's operational viewpoint.

## Hardware Profile
- **GPU:** NVIDIA RTX 4060 (8 GB VRAM, Ada Lovelace, CUDA 12.x)
- **Expected Training Time:** ~20–40 minutes for 50 epochs on YOLOv8n
- **Batch Size:** 16 (fits comfortably in 8 GB VRAM for nano model)

## Dataset Profile: UA-DETRAC
- **Source:** Kaggle mirror (`dschettler8845/ua-detrac-dataset`)
- **Content:** ~140,000 frames from 100 sequences, overhead CCTV cameras at intersections and highways
- **Original Format:** XML per-sequence with pixel-coordinate bounding boxes
- **Target Format:** YOLO normalized center-x, center-y, width, height in [0.0, 1.0]
- **Class Mapping:** car/van → 0, bus → 1, others → 2
- **Resolution:** 960×540

## Dependencies
- `kaggle` CLI with authenticated `kaggle.json`
- `ultralytics>=8.2.0` (already in requirements.txt)
- `torch` with CUDA support (for GPU training)
- Existing `scripts/convert_detrac_to_yolo.py` for label conversion

