import cv2
import numpy as np
from typing import List, Tuple, Optional
from core.tracker import TrackedVehicle

# Semantic Color Tokens in OpenCV BGR format (from 01_DESIGN_SYSTEM.md)
COLOR_SURFACE_PRIMARY = (23, 15, 11)      # Web #0B0F17
COLOR_SURFACE_ELEVATED = (46, 30, 22)     # Web #161E2E
COLOR_NOMINAL = (129, 185, 16)            # Web #10B981 (Emerald)
COLOR_WARNING = (11, 158, 245)            # Web #F59E0B (Amber)
COLOR_VIOLATION = (68, 68, 239)           # Web #EF4444 (Crimson Red)
COLOR_CALIBRATION_ZONE = (248, 189, 56)   # Web #38BDF8 (Sky Cyan)
COLOR_TIRE_ANCHOR = (94, 63, 244)         # Web #F43F5E (Rose)
COLOR_TEXT_WHITE = (245, 245, 245)


class ExceptionFirstHudRenderer:
    """
    Renders telemetry overlays following the Exception-First Visual Hierarchy:
      - Normal traffic blends quietly (corner-bracket bounding boxes, muted green).
      - Vehicles approaching (85-99%) limit get high-contrast amber warnings.
      - Vehicles exceeding (>=100%) limit get flashing/prominent solid red violation boxes.
      - 5px ground-plane anchor dot at bottom-center tire contact point.
      - Semi-transparent floating telemetry pills above vehicles.
      - 20% alpha semi-transparent homography calibration polygon.
    """

    def __init__(self, speed_limit_kmh: float = 40.0):
        self.speed_limit_kmh = float(speed_limit_kmh)

    def set_speed_limit(self, limit: float) -> None:
        self.speed_limit_kmh = float(limit)

    def draw_calibration_polygon(
        self,
        frame: np.ndarray,
        src_points: List[List[float]],
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Renders the 4-point homography region with 20% alpha cyan fill and 2px border.
        """
        if len(src_points) != 4 or frame is None:
            return frame

        overlay = frame.copy()
        pts = np.array(src_points, dtype=np.int32).reshape((-1, 1, 2))

        # 20% alpha cyan fill
        cv2.fillPoly(overlay, [pts], COLOR_CALIBRATION_ZONE)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # 2px border
        cv2.polylines(frame, [pts], isClosed=True, color=COLOR_CALIBRATION_ZONE, thickness=2, lineType=cv2.LINE_AA)

        if show_labels:
            labels = ["P1 (TL)", "P2 (TR)", "P3 (BR)", "P4 (BL)"]
            for idx, pt in enumerate(src_points):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(frame, (px, py), 6, COLOR_CALIBRATION_ZONE, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    labels[idx],
                    (px + 8, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    COLOR_TEXT_WHITE,
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def _draw_corner_brackets(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """
        Draws only the 4 corners of a bounding box (length = 20% of width/height).
        """
        w = x2 - x1
        h = y2 - y1
        line_w = max(6, int(w * 0.20))
        line_h = max(6, int(h * 0.20))

        # Top-Left corner
        cv2.line(img, (x1, y1), (x1 + line_w, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + line_h), color, thickness, cv2.LINE_AA)

        # Top-Right corner
        cv2.line(img, (x2, y1), (x2 - line_w, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + line_h), color, thickness, cv2.LINE_AA)

        # Bottom-Left corner
        cv2.line(img, (x1, y2), (x1 + line_w, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - line_h), color, thickness, cv2.LINE_AA)

        # Bottom-Right corner
        cv2.line(img, (x2, y2), (x2 - line_w, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - line_h), color, thickness, cv2.LINE_AA)

    def draw_vehicle(
        self,
        frame: np.ndarray,
        vehicle: TrackedVehicle,
        speed_kmh: Optional[float],
    ) -> np.ndarray:
        """
        Renders bounding box, tire contact anchor, and floating telemetry pill for a vehicle.
        """
        x1, y1, x2, y2 = [int(v) for v in vehicle.bbox]
        u, v = int(vehicle.tire_point[0]), int(vehicle.tire_point[1])

        # Determine visual state based on speed vs speed limit
        if speed_kmh is None:
            state = "NOMINAL"
            color = COLOR_NOMINAL
            thickness = 2
        elif speed_kmh >= self.speed_limit_kmh:
            state = "VIOLATION"
            color = COLOR_VIOLATION
            thickness = 3
        elif speed_kmh >= (self.speed_limit_kmh * 0.85):
            state = "WARNING"
            color = COLOR_WARNING
            thickness = 2
        else:
            state = "NOMINAL"
            color = COLOR_NOMINAL
            thickness = 2

        # 1. Bounding box rendering
        if state == "NOMINAL":
            self._draw_corner_brackets(frame, x1, y1, x2, y2, color, thickness=thickness)
        else:
            # Full solid rectangle for Warning & Violation
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # 2. Tire Contact Anchor Dot (5px solid circle on road plane)
        cv2.circle(frame, (u, v), 5, COLOR_TIRE_ANCHOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (u, v), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Floating Telemetry Pill
        if speed_kmh is not None:
            telemetry_str = f"#{vehicle.track_id:03d} | {vehicle.class_name} | {speed_kmh:4.1f} km/h"
        else:
            telemetry_str = f"#{vehicle.track_id:03d} | {vehicle.class_name} | TRACKING"

        font_scale = 0.48
        font_thickness = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(
            telemetry_str, font_face, font_scale, font_thickness
        )

        pill_x1 = max(4, x1)
        pill_y2 = max(text_h + 10, y1 - 6)
        pill_y1 = pill_y2 - text_h - 8
        pill_x2 = pill_x1 + text_w + 14

        # Pill background with alpha = 0.75
        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), COLOR_SURFACE_PRIMARY, -1)
        cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Text rendering
        text_origin = (pill_x1 + 7, pill_y2 - 5)
        cv2.putText(
            frame,
            telemetry_str,
            text_origin,
            font_face,
            font_scale,
            color if state != "NOMINAL" else COLOR_TEXT_WHITE,
            font_thickness,
            cv2.LINE_AA,
        )

        return frame

    def render_overlay(
        self,
        frame: np.ndarray,
        src_points: List[List[float]],
        vehicles: List[TrackedVehicle],
        speed_map: dict[int, Optional[float]],
    ) -> np.ndarray:
        """
        Complete frame rendering pipeline.
        """
        # Draw calibration polygon first (underneath vehicles)
        self.draw_calibration_polygon(frame, src_points)

        # Draw each vehicle
        for v in vehicles:
            spd = speed_map.get(v.track_id)
            self.draw_vehicle(frame, v, spd)

        return frame
