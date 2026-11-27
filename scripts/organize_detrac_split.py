"""
organize_detrac_split.py — Fast train/val split for UA-DETRAC using NTFS hard links

Organizes UA-DETRAC images and converted YOLO labels into an 80/20 train/val split
by sequence and generates data.yaml for Ultralytics YOLOv8.

Uses NTFS hard links on Windows:
- Uses 0 additional disk space (critical for laptops with limited SSD storage)
- Creates 80,000+ file links in ~3 seconds instead of 10+ minutes of file copies
"""

import argparse
import os
import random
import shutil
import yaml
from pathlib import Path


def fast_link(src: Path, dst: Path) -> None:
    """Create a hard link, fallback to copy if hard links fail."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def find_sequence_dirs(image_base: Path) -> dict[str, Path]:
    """Find all sequence directories containing .jpg files."""
    seq_dirs = {}
    # Check directly inside DETRAC-Images nested structure first
    candidates = [
        image_base / "DETRAC-Images" / "DETRAC-Images",
        image_base / "DETRAC-Images",
        image_base,
    ]
    for root in candidates:
        if root.exists():
            for child in root.iterdir():
                if child.is_dir() and child.name.startswith("MVI_"):
                    seq_dirs[child.name] = child
    
    # Fallback to rglob if none found
    if not seq_dirs:
        for p in image_base.rglob("MVI_*"):
            if p.is_dir():
                seq_dirs[p.name] = p
                
    return seq_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Organize UA-DETRAC into Ultralytics train/val split with hard links and data.yaml"
    )
    parser.add_argument(
        "--image-base",
        type=Path,
        default=Path("data/detrac_raw"),
        help="Root directory containing UA-DETRAC image sequences.",
    )
    parser.add_argument(
        "--label-base",
        type=Path,
        default=Path("data/detrac_yolo/labels"),
        help="Directory containing converted YOLO label directories (one per sequence).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detrac_yolo"),
        help="Output root for organized dataset.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of sequences for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    args = parser.parse_args()

    image_base = args.image_base
    label_base = args.label_base
    output_dir = args.output_dir

    if not image_base.exists():
        print(f"[ERROR] Image base directory not found: {image_base}")
        raise SystemExit(1)
    if not label_base.exists():
        print(f"[ERROR] Label base directory not found: {label_base}")
        print("[HINT] Run conversion first: python scripts/convert_detrac_to_yolo.py")
        raise SystemExit(1)

    # 1. Discover image sequence directories
    img_seq_map = find_sequence_dirs(image_base)
    print(f"[SCAN] Found {len(img_seq_map)} image sequences")

    # 2. Discover label sequence directories
    lbl_seq_map = {d.name: d for d in label_base.iterdir() if d.is_dir() and d.name.startswith("MVI_")}
    print(f"[SCAN] Found {len(lbl_seq_map)} label sequences")

    # 3. Match sequences that have both images and labels (the 60 annotated training sequences)
    common_seq_names = sorted(list(set(img_seq_map.keys()) & set(lbl_seq_map.keys())))
    if not common_seq_names:
        print("[ERROR] No matching sequence names between images and labels.")
        raise SystemExit(1)

    print(f"[MATCH] {len(common_seq_names)} sequences have both images and annotations")

    # 4. Partition sequences 80/20 train/val
    random.seed(args.seed)
    shuffled_names = list(common_seq_names)
    random.shuffle(shuffled_names)

    split_idx = int(len(shuffled_names) * args.split_ratio)
    train_names = set(shuffled_names[:split_idx])
    val_names = set(shuffled_names[split_idx:])

    print(f"[SPLIT] Train: {len(train_names)} sequences | Val: {len(val_names)} sequences")

    # 5. Create output directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 6. Link files into train and val directories
    total_train_imgs = 0
    total_train_lbls = 0
    total_val_imgs = 0
    total_val_lbls = 0

    print("\n[LINKING] Linking sequence files via zero-copy NTFS hard links...")

    for i, seq_name in enumerate(common_seq_names, 1):
        is_train = seq_name in train_names
        dest_img_dir = train_img_dir if is_train else val_img_dir
        dest_lbl_dir = train_lbl_dir if is_train else val_lbl_dir

        img_dir = img_seq_map[seq_name]
        lbl_dir = lbl_seq_map[seq_name]

        seq_imgs = 0
        seq_lbls = 0

        # Link all JPGs in sequence
        for img_path in img_dir.glob("*.jpg"):
            dest_img = dest_img_dir / f"{seq_name}_{img_path.name}"
            fast_link(img_path, dest_img)
            seq_imgs += 1

            # Check matching label
            lbl_src = lbl_dir / f"{img_path.stem}.txt"
            if lbl_src.exists():
                dest_lbl = dest_lbl_dir / f"{seq_name}_{img_path.stem}.txt"
                fast_link(lbl_src, dest_lbl)
                seq_lbls += 1

        split_label = "TRAIN" if is_train else "VAL  "
        print(f"  [{i:2d}/{len(common_seq_names)}] [{split_label}] {seq_name}: {seq_imgs} imgs, {seq_lbls} labels linked")

        if is_train:
            total_train_imgs += seq_imgs
            total_train_lbls += seq_lbls
        else:
            total_val_imgs += seq_imgs
            total_val_lbls += seq_lbls

    # 7. Generate data.yaml (compatible with Ultralytics YOLOv8)
    data_yaml = {
        "path": str(output_dir.resolve().as_posix()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {
            0: "car",
            1: "bus",
            2: "others",
        },
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*65}")
    print(f"[DONE] Dataset successfully partitioned & linked:")
    print(f"  Train:  {total_train_imgs:,} images, {total_train_lbls:,} labels ({len(train_names)} sequences)")
    print(f"  Val:    {total_val_imgs:,} images, {total_val_lbls:,} labels ({len(val_names)} sequences)")
    print(f"  Config: {yaml_path}")
    print(f"{'='*65}")

    print("\n[data.yaml]")
    print(yaml_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
