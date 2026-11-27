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

