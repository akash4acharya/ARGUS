# 07 — DATASET ACQUISITION & AUTOMATED FETCHING GUIDE (FOR AI AGENT)

## 1. Execution Rule for Coding Agent (Antigravity / Bionic)
Do **not** attempt to scrape web pages with BeautifulSoup or Selenium to download datasets, and do **not** download massive 50 GB archives during initial pipeline scaffolding. 
- **Phase 1 (Immediate Pipeline Verification):** Run `scripts/fetch_datasets.py --mode smoke-test` to pull a lightweight 30 MB calibrated traffic video clip so Stages 1–4 can be verified immediately.
- **Phase 2 (Benchmarking & Fine-Tuning):** Use authenticated CLI SDKs (`kaggle`, `roboflow`, `ultralytics`) to pull full datasets into `data/raw/` only when requested by the developer.

---

## 2. Target Directory Structure for Data
All downloaded datasets must be stored outside the `core/` and `backend/` source trees and excluded via `.gitignore`:

```text
cctv-speed-monitor/
├── data/
│   ├── smoke_test/                  # Fast 30MB traffic videos for instant local testing
│   ├── detrac_yolo/                 # UA-DETRAC converted to YOLOv8 format (Detection)
│   ├── indian_traffic_roboflow/     # Indian classes: Auto-rickshaw, two-wheeler, bus, truck
│   ├── veri_776/                    # Vehicle Re-ID embeddings for DeepSORT (Tracking)
│   └── brno_compspeed/              # 50 FPS videos + LiDAR ground-truth speeds (Physics)
└── scripts/
    ├── fetch_datasets.py            # Automated CLI downloader script
    └── convert_detrac_to_yolo.py    # XML-to-TXT bounding box normalizer
```

---

## 3. How Authentication Tokens Work (Plain-Language Setup)
Because dataset hosts block anonymous bots from draining their bandwidth, your script proves who you are by reading a secret text string saved on your computer instead of typing a password in a browser. This secret string is an **API Token**.

Before running the full benchmark downloads, set these two environment variables in your terminal (or `.env` file):
- **Kaggle Token (for UA-DETRAC & MIO-TCD):** Go to `kaggle.com/settings` -> "Create New Token". Save the downloaded `kaggle.json` inside `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<Username>\.kaggle\kaggle.json` (Windows).
- **Roboflow Key (for Indian CCTV Vehicles):** Copy your Private API Key from your Roboflow account settings and export it:
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

---

## 4. Dataset Catalog & Programmatic Fetch Commands

### Dataset A: Instant Smoke-Test Traffic Stream (No API Key Needed)
- **Purpose:** Verify `UniversalCCTVStream`, YOLOv8, DeepSORT, and `HomographySpeedEngine` in under 30 seconds.
- **How it works:** Uses sample asset mirrors to pull a clean 720p/1080p highway traffic stream directly into `data/smoke_test/sample_traffic.mp4`.

### Dataset B: UA-DETRAC (Overhead CCTV Vehicle Detection)
- **Purpose:** Fine-tune YOLOv8 on high-angle overhead CCTV cameras.
- **Source:** Kaggle Mirror (`dschettler8845/ua-detrac-dataset` — ~1.2 GB compressed subset or full 10 GB set).
- **CLI Command:**
```bash
kaggle datasets download -d dschettler8845/ua-detrac-dataset -p data/detrac_yolo --unzip
```

### Dataset C: Indian CCTV Traffic Vehicles (Roboflow Universe)
- **Purpose:** Standard Western datasets miss auto-rickshaws, Indian buses, and dense two-wheeler lanes. Pulling via Roboflow exports labels already pre-formatted for YOLOv8 (`data.yaml` + normalized `.txt` files).
- **Python SDK Fetch:** Uses the `roboflow` package to download directly into `data/indian_traffic_roboflow/`.

### Dataset D: BrnoCompSpeed (Real-World Speed Ground Truth)
- **Purpose:** Validate that `HomographySpeedEngine` outputs accurate km/h speeds against LiDAR laser ground truth.
- **Source:** Brno University of Technology Traffic Benchmark (`medusa.fit.vutbr.cz/traffic`).
- **Note:** The full archive is ~120 GB. Our fetcher script pulls only Session 0 (single camera calibration + ground truth JSON/mat metadata) or generates a synthetic calibrated test sequence if the university mirror is throttled.

---

## 5. Production Script 1: `scripts/fetch_datasets.py`
Have the coding agent create this file first so any dataset tier can be fetched via a single terminal command:

```python
import os
import sys
import urllib.request
import subprocess
import argparse
from pathlib import Path

BASE_DATA_DIR = Path("data")

# Direct raw MP4 test stream (highway CCTV overhead perspective) for immediate pipeline testing
SMOKE_TEST_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
)


def fetch_smoke_test() -> None:
    """Downloads a fast, lightweight CCTV-angle traffic MP4 with zero authentication."""
    target_dir = BASE_DATA_DIR / "smoke_test"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "sample_traffic.mp4"

    if out_file.exists() and out_file.stat().st_size > 100_000:
        print(f"[OK] Smoke-test video already exists at: {out_file}")
        return

    print(f"[DOWNLOADING] Pulling smoke-test CCTV stream -> {out_file} ...")
    urllib.request.urlretrieve(SMOKE_TEST_VIDEO_URL, out_file)
    print(f"[DONE] Saved {out_file.stat().st_size / (1024*1024):.2f} MB to {out_file}")


def fetch_ua_detrac_kaggle() -> None:
    """Downloads UA-DETRAC overhead CCTV dataset using Kaggle CLI."""
    target_dir = BASE_DATA_DIR / "detrac_yolo"
    target_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and not os.getenv("KAGGLE_KEY"):
        print(
            "[ERROR] Kaggle credentials missing. Place kaggle.json in ~/.kaggle/ "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
        sys.exit(1)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "dschettler8845/ua-detrac-dataset",
        "-p",
        str(target_dir),
        "--unzip",
    ]
    print(f"[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[DONE] UA-DETRAC extracted to {target_dir}")


def fetch_roboflow_indian_traffic(
    workspace: str = "indian-traffic-dataset",
    project_slug: str = "indian-vehicle-detection-cctv",
    version_num: int = 1,
) -> None:
    """Downloads YOLOv8-formatted Indian CCTV dataset via Roboflow API."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] Set ROBOFLOW_API_KEY environment variable before fetching Roboflow data.")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "roboflow"], check=True)
        from roboflow import Roboflow

    target_dir = BASE_DATA_DIR / "indian_traffic_roboflow"
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    project.version(version_num).download("yolov8", location=str(target_dir))
    print(f"[DONE] Roboflow YOLOv8 dataset ready at {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Dataset Fetcher for CCTV Speed Monitor")
    parser.add_argument(
        "--mode",
        choices=["smoke-test", "detrac", "roboflow-india", "all"],
        default="smoke-test",
        help="Which dataset tier to download.",
    )
    args = parser.parse_args()

    if args.mode in ("smoke-test", "all"):
        fetch_smoke_test()
    if args.mode in ("detrac", "all"):
        fetch_ua_detrac_kaggle()
    if args.mode in ("roboflow-india", "all"):
        fetch_roboflow_indian_traffic()
```

---

## 6. How Label Coordinate Normalization Works (`scripts/convert_detrac_to_yolo.py`)
Older academic benchmarks like UA-DETRAC record bounding boxes using raw pixel corners—such as `left=240`, `top=180`, `width=120`, `height=90` on a 960x540 image. However, YOLOv8 trains on images of varying resolutions, so it refuses raw pixel numbers. Instead, YOLOv8 requires every box to be expressed as the center point and size divided by the total image width and height so every number sits strictly between 0.0 and 1.0:

$$x_{\text{center\_norm}} = \frac{x_{\text{left}} + (w / 2)}{W_{\text{image}}}, \quad y_{\text{center\_norm}} = \frac{y_{\text{top}} + (h / 2)}{H_{\text{image}}}$$

$$w_{\text{norm}} = \frac{w}{W_{\text{image}}}, \quad h_{\text{norm}} = \frac{h}{H_{\text{image}}}$$

Converting raw pixel coordinates into resolution-independent 0.0–1.0 ratios is called **Bounding-Box Coordinate Normalization**.

### Production Script 2: `scripts/convert_detrac_to_yolo.py`

```python
import xml.etree.ElementTree as ET
from pathlib import Path

# Map UA-DETRAC XML vehicle types to unified 0-indexed YOLO training classes
DETRAC_CLASS_MAP = {
    "car": 0,
    "van": 0,
    "bus": 1,
    "others": 2,
}

IMG_WIDTH = 960.0
IMG_HEIGHT = 540.0


def convert_xml_sequence(xml_file: Path, output_labels_dir: Path) -> int:
    """
    Parses a single UA-DETRAC sequence XML file and writes per-frame
    normalized YOLOv8 .txt label files.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    seq_name = root.attrib.get("name", xml_file.stem)
    seq_out_dir = output_labels_dir / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    frames_converted = 0
    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.attrib["num"])
        target_list = frame_elem.find("target_list")
        if target_list is None:
            continue

        yolo_lines = []
        for target in target_list.findall("target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None or attr is None:
                continue

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            v_type = attr.attrib.get("vehicle_type", "car").lower()

            cls_id = DETRAC_CLASS_MAP.get(v_type, 0)

            # Normalize to [0.0, 1.0] center-x, center-y, width, height
            x_center = min(1.0, max(0.0, (left + width / 2.0) / IMG_WIDTH))
            y_center = min(1.0, max(0.0, (top + height / 2.0) / IMG_HEIGHT))
            w_norm = min(1.0, max(0.0, width / IMG_WIDTH))
            h_norm = min(1.0, max(0.0, height / IMG_HEIGHT))

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_file = seq_out_dir / f"img{frame_num:05d}.txt"
        label_file.write_text("\n".join(yolo_lines), encoding="utf-8")
        frames_converted += 1

    return frames_converted
```

---

## 7. Terminal Sequence for Antigravity to Run

Install dataset dependencies:
```bash
pip install kaggle roboflow gdown
```

Pull the instant smoke-test video and verify the end-to-end FastAPI + OpenCV pipeline before downloading gigabytes of training data:
```bash
python scripts/fetch_datasets.py --mode smoke-test
export CCTV_SOURCE="data/smoke_test/sample_traffic.mp4"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

