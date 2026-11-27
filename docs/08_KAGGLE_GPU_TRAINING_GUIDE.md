# 08 — ZERO-STRESS KAGGLE GPU TRAINING GUIDE FOR ARGUS

## 1. Executive Summary & Why We Train in the Cloud

Running vision model training (YOLOv8, 80,000+ images, 25–50 epochs) on a gaming laptop's integrated or mobile GPU (NVIDIA RTX 4060 Laptop GPU) introduces several real-world hardware penalties:
- **Thermal Wear & Fan Noise:** Constant 80–90°C GPU temperatures over 1–2 hours accelerate thermal degradation of laptop VRMs and battery cells.
- **SSD Storage Inflation:** Caching 140,000 uncompressed frames and gradient checkpoints consumes 15–25 GB of local SSD space.
- **Multitasking Stutters:** Windows desktop latency degrades while the VRAM buffer is saturated.

**The Solution:** Offload the compute to **Kaggle GPU Notebooks**.
- **Free Compute:** **30 hours per week** of free NVIDIA Tesla T4 (16 GB VRAM) or P100 GPUs.
- **Instant Data Access:** The **UA-DETRAC dataset is already hosted on Kaggle servers** (`bratjay/ua-detrac-orig`). Mounts in under 2 seconds over data-center NVMe pipes without touching your laptop's network.
- **Lightweight Hand-off:** When training finishes, you only download the **6 MB `best.pt` weight file** back to your laptop. Your RTX 4060 is saved strictly for low-power, real-time 30 FPS inference.

---

## 2. Prerequisites & Kaggle Account Setup (5 Minutes)

### Step 2.1: Enable Free GPU Accelerator Access
1. Log into your Kaggle account at [kaggle.com](https://www.kaggle.com).
2. Go to **Settings** (`kaggle.com/settings`).
3. Scroll down to **Phone Verification**.
4. Complete SMS verification. *(Kaggle requires this one-time check to prevent crypto mining bots; it immediately unlocks 30 free GPU hours per week).*

---

## 3. Creating the Kaggle Training Notebook (Step-by-Step UI)

### Step 3.1: Start a New Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Name your notebook in the top left: `argus-yolov8-detrac-train`.

### Step 3.2: Attach the UA-DETRAC Dataset (Dataset, NOT Notebook!)
1. Look at the right-hand panel under **Input** -> Click **"+ Add Input"**.
2. Filter by **Datasets** and search for: `bratjay/ua-detrac-orig`. *(Do not attach a Notebook like `jprafi/ua-detrac`, which doesn't expose the raw XMLs/images).*
3. Click the **"+" (Add)** button next to the dataset.
4. Kaggle will mount it instantly at `/kaggle/input/ua-detrac-orig`.

### Step 3.3: Turn On the GPU Accelerator
1. In the right-hand sidebar under **Notebook options**:
   - Find **Accelerator**.
   - Change from `None` to **`GPU T4 x2`** (or **`GPU T4 x1`**).
   - Ensure **Internet** is toggled to **`Internet on`** (so pip can install Ultralytics).

### Step 3.4: Interactive vs. Background ("Save & Run All") Execution
- **Do NOT use interactive "Run All" for the full 30-epoch training** if you plan to close your laptop or switch tabs for 20+ minutes — Kaggle interactive sessions idle-timeout and stop.
- Instead, click **"Save Version"** (top right) -> choose **"Save & Run All (Commit)"** -> click **Save**.
- Kaggle will spin up a background cloud worker, run every cell end-to-end, and save the final `argus_yolov8_best_weights.zip` to the **Output** tab even if you close your browser or turn off your laptop!

---

## 4. The Complete Notebook Code (Cell-by-Cell)

Copy each cell below into your Kaggle notebook sequentially, or upload the ready-made `notebooks/argus_train_yolov8_kaggle.ipynb` included in this repository.

### Cell 1: Environment Setup
```python
# Install latest Ultralytics and dependencies
!pip install -q ultralytics pyyaml

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

---

### Cell 2: Convert UA-DETRAC XMLs to YOLO Format
Because `/kaggle/input` is read-only, we read the XML annotations directly from the mounted dataset, convert the bounding boxes to normalized coordinates $(x_c, y_c, w, h) \in [0.0, 1.0]$, and save the lightweight `.txt` files directly into `/kaggle/working/labels/`.

```python
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Map vehicle types to unified YOLO classes: 0: car, 1: bus, 2: others
DETRAC_CLASS_MAP = {
    "car": 0,
    "van": 0,
    "bus": 1,
    "others": 2,
}

IMG_WIDTH = 960.0
IMG_HEIGHT = 540.0

xml_dir = Path("/kaggle/input/ua-detrac-orig/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML")
labels_out_dir = Path("/kaggle/working/labels")
labels_out_dir.mkdir(parents=True, exist_ok=True)

xml_files = sorted(list(xml_dir.glob("*.xml")))
print(f"Found {len(xml_files)} sequence XML files. Converting...")

total_converted = 0
for i, xml_file in enumerate(xml_files, 1):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    seq_name = root.attrib.get("name", xml_file.stem)
    seq_lbl_dir = labels_out_dir / seq_name
    seq_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.attrib["num"])
        target_list = frame_elem.find("target_list")
        if target_list is None:
            continue
            
        lines = []
        for target in target_list.findall("target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None:
                continue
                
            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            
            v_type = attr.attrib.get("vehicle_type", "car").lower() if attr is not None else "car"
            cls_id = DETRAC_CLASS_MAP.get(v_type, 0)
            
            # Normalize to [0.0, 1.0]
            x_center = min(1.0, max(0.0, (left + width / 2.0) / IMG_WIDTH))
            y_center = min(1.0, max(0.0, (top + height / 2.0) / IMG_HEIGHT))
            w_norm = min(1.0, max(0.0, width / IMG_WIDTH))
            h_norm = min(1.0, max(0.0, height / IMG_HEIGHT))
            
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        out_lbl = seq_lbl_dir / f"img{frame_num:05d}.txt"
        out_lbl.write_text("\n".join(lines), encoding="utf-8")
        total_converted += 1

print(f"Finished! Converted {total_converted:,} frames across {len(xml_files)} sequences.")
```

---

### Cell 3: Zero-Copy Dataset Partitioning via Linux Symlinks
Kaggle's disk has a 20 GB limit on `/kaggle/working/`. By creating **symbolic links (`os.symlink`)** from `/kaggle/working/images/` pointing directly to `/kaggle/input/`, we use **0 MB of disk space** while giving Ultralytics the exact directory hierarchy it expects.

```python
import random
import shutil

images_root = Path("/kaggle/input/ua-detrac-orig/DETRAC-Images/DETRAC-Images")
train_img_dir = Path("/kaggle/working/images/train")
val_img_dir = Path("/kaggle/working/images/val")
train_lbl_dir = Path("/kaggle/working/labels/train")
val_lbl_dir = Path("/kaggle/working/labels/val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Discover matching sequences (80/20 split)
seq_names = sorted([d.name for d in labels_out_dir.iterdir() if d.is_dir() and d.name.startswith("MVI_")])
random.seed(42)
random.shuffle(seq_names)

split_idx = int(len(seq_names) * 0.8)
train_seqs = set(seq_names[:split_idx])
val_seqs = set(seq_names[split_idx:])

print(f"Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)}")

for seq_name in seq_names:
    is_train = seq_name in train_seqs
    dest_img_dir = train_img_dir if is_train else val_img_dir
    dest_lbl_dir = train_lbl_dir if is_train else val_lbl_dir
    
    src_img_seq = images_root / seq_name
    src_lbl_seq = labels_out_dir / seq_name
    
    for img_path in src_img_seq.glob("*.jpg"):
        # Symlink image (Zero disk space consumed)
        dest_img = dest_img_dir / f"{seq_name}_{img_path.name}"
        if not dest_img.exists():
            os.symlink(str(img_path), str(dest_img))
            
        # Move/link label
        lbl_path = src_lbl_seq / f"{img_path.stem}.txt"
        if lbl_path.exists():
            dest_lbl = dest_lbl_dir / f"{seq_name}_{img_path.stem}.txt"
            if not dest_lbl.exists():
                os.symlink(str(lbl_path), str(dest_lbl))

print(f"Train images linked: {len(list(train_img_dir.glob('*'))):,}")
print(f"Val images linked:   {len(list(val_img_dir.glob('*'))):,}")
```

---

### Cell 4: Generate `data.yaml`
```python
import yaml

data_yaml = {
    "path": "/kaggle/working",
    "train": "images/train",
    "val": "images/val",
    "nc": 3,
    "names": {
        0: "car",
        1: "bus",
        2: "others",
    },
}

with open("/kaggle/working/data.yaml", "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("data.yaml successfully generated:")
!cat /kaggle/working/data.yaml
```

---

### Cell 5: Fine-Tune YOLOv8n on Tesla T4 GPU
```python
from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train for 30 epochs with FP16 mixed precision
results = model.train(
    data="/kaggle/working/data.yaml",
    epochs=30,
    batch=32,            # Tesla T4 handles batch 32 with ease
    imgsz=640,
    device=0,
    project="/kaggle/working/runs",
    name="argus-detrac-v1",
    half=True,           # FP16 acceleration
    workers=4,
    patience=10,
    mosaic=1.0,
    degrees=5.0,
    flipud=0.0,
    fliplr=0.5,
    verbose=True,
)
```

---

### Cell 6: Package & Download `best.pt`
```python
# Create a zip of the final trained weights and performance curves
!zip -j /kaggle/working/argus_yolov8_best_weights.zip \
    /kaggle/working/runs/argus-detrac-v1/weights/best.pt \
    /kaggle/working/runs/argus-detrac-v1/results.png \
    /kaggle/working/runs/argus-detrac-v1/confusion_matrix.png

print("Weights ready for download at /kaggle/working/argus_yolov8_best_weights.zip")
```

To download the file:
1. In the right-hand panel of your Kaggle notebook, refresh the **Output** directory.
2. Click the three dots next to `argus_yolov8_best_weights.zip` -> **Download**.

---

## 5. Integrating the Downloaded Weights into ARGUS

Once you download the `argus_yolov8_best_weights.zip` file:
1. Extract `best.pt` into your ARGUS project directory:
   ```text
   ARGUS/
   └── models/
       └── argus_detrac_yolov8n.pt
   ```
2. Open `config/settings.yaml` and update the detector path:
   ```yaml
   detector:
     model_weights: "models/argus_detrac_yolov8n.pt"
     confidence_threshold: 0.35
     iou_threshold: 0.45
     classes: [0, 1, 2]     # 0: car, 1: bus, 2: others
     device: "0"            # Uses RTX 4060 for fast, cool inference
   ```
3. Run the live ARGUS pipeline:
   ```bash
   python -m backend.main
   ```

Now your system runs with high-precision overhead CCTV weights trained on 80,000+ real surveillance frames, while your local GPU stayed completely cool during the entire training cycle!
