"""
train_yolo.py — Fine-tune YOLOv8 on UA-DETRAC overhead CCTV dataset

Trains YOLOv8 (nano by default) using transfer learning from COCO pretrained
weights on overhead CCTV frames for dramatically improved surveillance-angle
vehicle detection.

Hardware target: NVIDIA RTX 4060 (8 GB VRAM)
Expected training time: ~20-40 minutes for 50 epochs on YOLOv8n

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 100 --batch 16 --model yolov8s.pt
    python scripts/train_yolo.py --resume  # resume interrupted training
"""

import argparse
import sys
import torch
from pathlib import Path


def detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        return "0"  # Ultralytics expects string "0" for first GPU
    else:
        print("[CPU] No CUDA GPU detected — training on CPU (will be slow)")
        return "cpu"


def validate_data_yaml(data_yaml: Path) -> bool:
    """Check that the dataset manifest exists and is valid."""
    if not data_yaml.exists():
        print(f"[ERROR] Dataset config not found: {data_yaml}")
        print("[HINT]  Run these steps first:")
        print("  1. python scripts/convert_detrac_to_yolo.py")
        print("  2. python scripts/organize_detrac_split.py")
        return False

    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    required = ["path", "train", "val", "nc", "names"]
    for key in required:
        if key not in cfg:
            print(f"[ERROR] Missing key '{key}' in {data_yaml}")
            return False

    train_dir = Path(cfg["path"]) / cfg["train"]
    val_dir = Path(cfg["path"]) / cfg["val"]
    train_count = len(list(train_dir.glob("*.*"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.*"))) if val_dir.exists() else 0

    print(f"[DATA] Config: {data_yaml}")
    print(f"  Classes: {cfg['nc']} — {cfg['names']}")
    print(f"  Train images: {train_count:,}")
    print(f"  Val images:   {val_count:,}")

    if train_count == 0:
        print("[ERROR] No training images found!")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on UA-DETRAC overhead CCTV dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model weights (yolov8n.pt, yolov8s.pt, yolov8m.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/detrac_yolo/data.yaml"),
        help="Path to data.yaml dataset manifest.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, fits 8GB VRAM for nano model).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saving runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="argus-detrac-v1",
        help="Run name within the project directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training dataset to use per epoch (e.g. 0.25 for fast runs).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on val set (skip training).",
    )
    args = parser.parse_args()

    # ── Device detection ─────────────────────────────────────────────
    device = detect_device()

    # ── Dataset validation ───────────────────────────────────────────
    if not validate_data_yaml(args.data):
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run:")
        print("  .venv/Scripts/pip install ultralytics")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ARGUS YOLOv8 Fine-Tuning")
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  ImgSize:  {args.imgsz}")
    print(f"  Device:   {device}")
    print(f"  Output:   {args.project}/{args.name}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    if args.eval_only:
        # ── Evaluation only ──────────────────────────────────────────
        print("[EVAL] Running validation on trained model...")
        best_weights = Path(args.project) / args.name / "weights" / "best.pt"
        if best_weights.exists():
            model = YOLO(str(best_weights))
        else:
            print(f"[WARN] No best.pt found at {best_weights}, using {args.model}")

        results = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
        )
        print(f"\n[RESULTS]")
        print(f"  mAP@50:      {results.box.map50:.4f}")
        print(f"  mAP@50-95:   {results.box.map:.4f}")
        return

    # ── Training ─────────────────────────────────────────────────────
    if args.resume:
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[RESUME] Loading checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"[WARN] No checkpoint found at {last_ckpt}, starting fresh")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
        # Augmentation (Ultralytics defaults are good for surveillance)
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,           # Slight rotation for camera angle variation
        translate=0.1,
        scale=0.5,
        flipud=0.0,           # No vertical flip (vehicles don't appear upside-down)
        fliplr=0.5,           # Horizontal flip is fine
        # Logging
        verbose=True,
        plots=True,
        save=True,
        save_period=10,        # Save checkpoint every 10 epochs
    )

    # ── Post-training summary ────────────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best weights: {best_weights}")

    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"  Model size:   {size_mb:.1f} MB")

    print(f"\n  To use in ARGUS, update config/settings.yaml:")
    print(f'    detector.model_weights: "{best_weights}"')
    print(f'    detector.device: "cuda:0"')
    print(f"\n  To evaluate:")
    print(f"    python scripts/train_yolo.py --eval-only")
    print(f"\n  Training curves and metrics saved to:")
    print(f"    {args.project}/{args.name}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

