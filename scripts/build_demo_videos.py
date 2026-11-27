"""
build_demo_videos.py — Build wide-angle multi-lane CCTV demo videos from local UA-DETRAC sequences.

Converts high-angle highway/arterial JPEG sequences already stored in data/detrac_raw/
into smooth 25 FPS MP4 streams under data/demo/ for live ARGUS demonstrations.
"""

import cv2
from pathlib import Path

SEQUENCES = [
    ("MVI_40191", Path("data/demo/highway_expressway.mp4"), 25.0, 900),   # 36s crisp 6-lane expressway loop
    ("MVI_40712", Path("data/demo/urban_arterial.mp4"), 25.0, 900),       # 36s wide 8-lane unseen test loop
]


def build_mp4(seq_name: str, out_path: Path, fps: float = 25.0, max_frames: int = 900) -> None:
    seq_dir = Path("data/detrac_raw/DETRAC-Images/DETRAC-Images") / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")

    frames = sorted(seq_dir.glob("*.jpg"))[:max_frames]
    if not frames:
        raise RuntimeError(f"No JPG frames found in {seq_dir}")

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for img_path in frames:
        img = cv2.imread(str(img_path))
        if img is not None:
            writer.write(img)
    writer.release()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Built {out_path} ({len(frames)} frames @ {fps} FPS, {w}x{h}, {size_mb:.1f} MB)")


if __name__ == "__main__":
    for seq_name, out_path, fps, max_frames in SEQUENCES:
        build_mp4(seq_name, out_path, fps, max_frames)
