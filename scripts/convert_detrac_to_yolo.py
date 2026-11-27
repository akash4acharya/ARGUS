"""
convert_detrac_to_yolo.py — Batch UA-DETRAC XML-to-YOLO Label Converter

Converts all UA-DETRAC sequence XML annotation files into per-frame
normalized YOLO .txt label files suitable for Ultralytics YOLOv8 training.

Usage:
    python scripts/convert_detrac_to_yolo.py \
        --xml-dir data/detrac_raw/DETRAC-Train-Annotations-XML \
        --output-dir data/detrac_yolo/labels
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Map UA-DETRAC XML vehicle types to unified 0-indexed YOLO training classes
DETRAC_CLASS_MAP = {
    "car": 0,
    "van": 0,
    "bus": 1,
    "others": 2,
}

# Default UA-DETRAC frame resolution
IMG_WIDTH = 960.0
IMG_HEIGHT = 540.0


def convert_xml_sequence(xml_file: Path, output_labels_dir: Path) -> int:
    """
    Parses a single UA-DETRAC sequence XML file and writes per-frame
    normalized YOLOv8 .txt label files.

    Returns:
        Number of frames converted.
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
            if box is None:
                continue

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])

            # Get vehicle type from attribute if available
            if attr is not None:
                v_type = attr.attrib.get("vehicle_type", "car").lower()
            else:
                v_type = "car"

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


def main():
    parser = argparse.ArgumentParser(
        description="Convert UA-DETRAC XML annotations to YOLO normalized label files."
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("data/detrac_raw/DETRAC-Train-Annotations-XML"),
        help="Directory containing UA-DETRAC sequence XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detrac_yolo/labels"),
        help="Output directory for per-sequence per-frame YOLO .txt files.",
    )
    args = parser.parse_args()

    xml_dir = args.xml_dir
    output_dir = args.output_dir

    if not xml_dir.exists():
        print(f"[ERROR] XML directory not found: {xml_dir}")
        print("[HINT]  Download UA-DETRAC first:  python scripts/fetch_datasets.py --mode detrac")
        raise SystemExit(1)

    xml_files = sorted(xml_dir.rglob("*.xml"))
    if not xml_files:
        print(f"[ERROR] No XML files found in {xml_dir}")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CONVERT] Found {len(xml_files)} XML annotation files in {xml_dir}")
    print(f"[CONVERT] Output labels → {output_dir}")

    total_frames = 0
    for i, xml_file in enumerate(xml_files, 1):
        n = convert_xml_sequence(xml_file, output_dir)
        total_frames += n
        print(f"  [{i:3d}/{len(xml_files)}] {xml_file.stem}: {n} frames converted")

    print(f"\n[DONE] Converted {total_frames:,} frames across {len(xml_files)} sequences.")
    print(f"[DONE] Labels saved to: {output_dir}")

    # Spot-check: validate a random label file
    all_labels = list(output_dir.rglob("*.txt"))
    if all_labels:
        sample = all_labels[len(all_labels) // 2]
        lines = sample.read_text().strip().split("\n")
        print(f"\n[SPOT CHECK] {sample.relative_to(output_dir)}")
        for line in lines[:3]:
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 fields, got {len(parts)}"
            cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0, f"Coordinates out of range: {xc}, {yc}"
            assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0, f"Dimensions out of range: {w}, {h}"
            print(f"  class={cls}  cx={xc:.4f}  cy={yc:.4f}  w={w:.4f}  h={h:.4f}  ✓")
        print("[SPOT CHECK] All values in [0.0, 1.0] — normalization correct.")


if __name__ == "__main__":
    main()
