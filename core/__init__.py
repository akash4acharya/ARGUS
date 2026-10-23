"""
ARGUS CCTV Speed & Telemetry Core Package.
Contains streaming, object detection, multi-object tracking,
planar homography math, and HUD overlay rendering.
"""

from core.stream_reader import UniversalCCTVStream, TimestampedFrame
from core.detector import YOLOv8VehicleDetector
from core.tracker import DeepSORTVehicleTracker, TrackedVehicle
from core.homography_speed import HomographySpeedEngine
from core.hud_renderer import ExceptionFirstHudRenderer

__all__ = [
    "UniversalCCTVStream",
    "TimestampedFrame",
    "YOLOv8VehicleDetector",
    "DeepSORTVehicleTracker",
    "TrackedVehicle",
    "HomographySpeedEngine",
    "ExceptionFirstHudRenderer",
]
