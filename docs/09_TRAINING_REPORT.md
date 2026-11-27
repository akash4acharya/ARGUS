# 09 — ARGUS YOLOv8n UA-DETRAC Training & Evaluation Report

## 1. Executive Summary
To adapt ARGUS from eye-level natural photography (COCO) to elevated traffic CCTV angles, `yolov8n.pt` was fine-tuned on the **UA-DETRAC** overhead vehicle benchmark using cloud GPU acceleration (NVIDIA Tesla T4, 15 GB VRAM).

- **Validation Dataset:** `16,001` images / `124,562` vehicle instances across `12` held-out camera sequences (`80/20` sequence-disjoint split)
- **Overall Validation mAP@50:** **`0.765`** *(exceeded target threshold of `0.65`)*
- **Car mAP@50:** **`0.941`** (`P = 0.875`, `R = 0.913`)
- **Bus mAP@50:** **`0.924`** (`P = 0.835`, `R = 0.895`)
- **Inference Latency (T4 FP16):** **`1.3 ms`** per frame (`0.1 ms` preprocess, `1.3 ms` postprocess)

---

## 2. Dataset & Training Configuration

| Parameter | Value |
|---|---|
| **Dataset** | UA-DETRAC (`bratjay/ua-detrac-orig`, 60 annotated sequences) |
| **Train Split** | 48 sequences — `67,790` images (`66,811` labeled, `979` backgrounds) |
| **Validation Split** | 12 sequences — `16,001` images (`15,274` labeled, `727` backgrounds) |
| **Classes (`nc=3`)** | `0: car` (car + van), `1: bus`, `2: others` |
| **Base Architecture** | YOLOv8n (`129` layers, `3,011,433` params / `73` fused layers, `8.6 GFLOPs`) |
| **Image Size & Batch** | `640 × 640`, batch size `32`, AMP FP16 |
| **Optimizer** | MuSGD (`lr0=0.01`, `momentum=0.9`, `weight_decay=0.0005`) |
| **Augmentations** | Mosaic (`1.0`), rotation (`±5°`), horizontal flip (`0.5`), vertical flip (`0.0`), HSV jitter |
| **Training Hardware** | Kaggle NVIDIA Tesla T4 (`14,912 MiB` VRAM), `30` epochs in `5.413 hours` |

---

## 3. Final Validation Breakdown (`best.pt`)

Evaluated on all `16,001` validation frames (`124,562` ground-truth boxes):

| Class | Images | Instances | Box Precision (`P`) | Recall (`R`) | `mAP@50` | `mAP@50-95` |
|---|---|---|---|---|---|---|
| **all** | 16,001 | 124,562 | **0.792** | **0.728** | **0.765** | **0.600** |
| **car** | 15,079 | 112,239 | **0.875** | **0.913** | **0.941** | **0.708** |
| **bus** | 8,683 | 11,498 | **0.835** | **0.895** | **0.924** | **0.785** |
| **others** | 825 | 825 | 0.664 | 0.377 | 0.431 | 0.308 |

*Note: `others` has only 825 validation instances (`< 0.7%` of labels) and represents heterogeneous/rare utility vehicles; primary traffic classes (`car` + `bus`, >99.3% of traffic) achieve `0.941` and `0.924` mAP@50.*

---

## 4. Training Convergence Summary (Selected Epochs from 30-Epoch Run)

| Epoch | `box_loss` | `cls_loss` | `dfl_loss` | Val Precision | Val Recall | Val `mAP@50` | Val `mAP@50-95` |
|---|---|---|---|---|---|---|---|
| **1** | 0.9455 | 0.6117 | 0.9403 | 0.802 | 0.714 | 0.753 | 0.587 |
| **5** | 0.7698 | 0.4218 | 0.8898 | 0.826 | 0.669 | 0.729 | 0.569 |
| **10** | 0.7022 | 0.3669 | 0.8700 | 0.819 | 0.673 | 0.741 | 0.585 |
| **15** | 0.6739 | 0.3448 | 0.8622 | 0.784 | 0.714 | 0.759 | 0.597 |
| **20** | 0.6498 | 0.3268 | 0.8560 | 0.777 | 0.723 | 0.761 | 0.597 |
| **23** | 0.6117 | 0.2951 | 0.8422 | 0.792 | 0.728 | **0.766** | **0.601** |
| **25** | 0.5987 | 0.2853 | 0.8380 | 0.795 | 0.734 | **0.766** | 0.600 |
| **30** | 0.5641 | 0.2598 | 0.8286 | 0.802 | 0.711 | 0.755 | 0.589 |

*(Source log archived locally at `C:\Users\Akash M Acharya\Downloads\argus 30 Epochs.ipynb`.)*

---

## 5. Key Engineering Takeaways
1. **Transfer Learning Efficiency:** Pretrained `yolov8n.pt` reaches `0.753` mAP@50 by Epoch 1 and converges within `0.007` of peak accuracy by Epoch 15 (`0.759` vs `0.766`). Subsequent rapid iteration cycles can safely use `10–15` epochs.
2. **Sequence-Disjoint Splitting Prevents Leakage:** Splitting strictly by camera sequence (`48` train / `12` val) guarantees the `0.941 car` / `0.924 bus` scores reflect generalization to **unseen camera angles**, not frame memorization.
3. **Phase 8 Next Step:** Expand `nc` to include Indian road vehicle classes (auto-rickshaw, two-wheeler) using Roboflow Universe datasets.
