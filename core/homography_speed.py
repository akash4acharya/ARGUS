import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Optional, Dict


class HomographySpeedEngine:
    """
    Translates 2D image tire-contact pixels (u, v) into real-world road-plane
    coordinates (X, Y) in meters and computes smoothed velocity in km/h.

    Implements:
      - Direct Linear Transform (DLT) via cv2.getPerspectiveTransform
      - Bottom-center tire contact projection
      - Point-in-polygon zone boundary gating
      - Rolling window trajectory buffer (min_samples=8, window_size=15)
      - Rolling median filter over recent speed estimations
      - Physical anomaly rejection (> 220 km/h)
    """

    def __init__(
        self,
        src_points: List[List[float]],
        real_width_m: float,
        real_length_m: float,
        window_size: int = 15,
        min_samples: int = 8,
    ):
        self.window_size = window_size
        self.min_samples = min_samples

        # Per-vehicle trajectory buffer: track_id -> deque of (timestamp, X_m, Y_m)
        self.trajectories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        # Per-vehicle smoothed speed buffer: track_id -> deque of recent km/h floats
        self.speed_buffer: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))

        self.update_calibration(src_points, real_width_m, real_length_m)

    def update_calibration(
        self, src_points: List[List[float]], real_width_m: float, real_length_m: float
    ) -> None:
        if len(src_points) != 4:
            raise ValueError("Calibration requires exactly 4 non-collinear source points.")

        self.real_width_m = float(real_width_m)
        self.real_length_m = float(real_length_m)
        self.src_polygon = np.array(src_points, dtype=np.float32)

        # Destination top-down metric plane:
        # P1: Top-Left (0, 0)
        # P2: Top-Right (W, 0)
        # P3: Bottom-Right (W, L)
        # P4: Bottom-Left (0, L)
        self.dst_metric = np.array(
            [
                [0.0, 0.0],
                [self.real_width_m, 0.0],
                [self.real_width_m, self.real_length_m],
                [0.0, self.real_length_m],
            ],
            dtype=np.float32,
        )

        self.H_matrix = cv2.getPerspectiveTransform(self.src_polygon, self.dst_metric)
        self.H_inv = cv2.getPerspectiveTransform(self.dst_metric, self.src_polygon)

        # Clear stale trajectories when geometry changes
        self.trajectories.clear()
        self.speed_buffer.clear()

    def is_inside_zone(self, tire_px: Tuple[float, float]) -> bool:
        """
        Ray-casting point-in-polygon test. Returns True if point is inside or on boundary.
        """
        return cv2.pointPolygonTest(self.src_polygon, (float(tire_px[0]), float(tire_px[1])), measureDist=False) >= 0

    def pixel_to_meters(self, tire_px: Tuple[float, float]) -> Tuple[float, float]:
        """
        Projects pixel coordinates (u, v) through the homography matrix onto the metric ground plane.
        """
        pt_array = np.array([[[float(tire_px[0]), float(tire_px[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_array, self.H_matrix)[0][0]
        return float(transformed[0]), float(transformed[1])

    def meters_to_pixel(self, meter_pt: Tuple[float, float]) -> Tuple[float, float]:
        """
        Inverse projection from metric ground plane back to image pixel coordinates.
        """
        pt_array = np.array([[[float(meter_pt[0]), float(meter_pt[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_array, self.H_inv)[0][0]
        return float(transformed[0]), float(transformed[1])

    def compute_speed_kmh(
        self, track_id: int, tire_px: Tuple[float, float], capture_time: float
    ) -> Optional[float]:
        """
        Calculates smoothed velocity for a vehicle track inside the calibration polygon.

        Returns:
            Smoothed speed in km/h if valid, otherwise None.
        """
        if not self.is_inside_zone(tire_px):
            return None

        x_m, y_m = self.pixel_to_meters(tire_px)
        traj = self.trajectories[track_id]
        traj.append((capture_time, x_m, y_m))

        if len(traj) < self.min_samples:
            return None

        t_start, x_start, y_start = traj[0]
        t_end, x_end, y_end = traj[-1]
        delta_t = t_end - t_start

        # Require a minimum duration to avoid division-by-zero or extreme short-baseline noise
        if delta_t <= 0.08:
            return None

        displacement_m = float(np.hypot(x_end - x_start, y_end - y_start))
        instant_kmh = (displacement_m / delta_t) * 3.6

        # Reject physically impossible glitch readings (> 220 km/h on campus/urban roads)
        if instant_kmh > 220.0 or instant_kmh < 0.0:
            return None

        self.speed_buffer[track_id].append(instant_kmh)
        return float(np.median(self.speed_buffer[track_id]))
