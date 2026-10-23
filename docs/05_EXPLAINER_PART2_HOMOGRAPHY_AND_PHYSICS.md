# 05 — A-TO-Z EXPLAINER (PART 2): Planar Homography Math, Speed Physics & HUD Overlay

## Purpose of This Document
This file covers **Stage 3 (Spatial Translation & Speed Physics)** and the **Exception-First OpenCV HUD Renderer** located in `core/homography_speed.py` and `core/hud_renderer.py`.

---

## 1. The Geometry of Flattening an Angled CCTV Feed

### Why Naive Pixel Speed Fails
A security camera mounted on a building or pole looks down at the road at an oblique angle. Because of perspective foreshortening, a 1-meter step near the top of the video frame might span only 4 pixels, whereas a 1-meter step near the bottom of the frame spans 45 pixels. Dividing pixel movement by time produces meaningless numbers.

### How the 4-Point Stretching Matrix Works
Imagine laying a rectangular rubber sheet over the angled trapezoid of road visible on your screen, and pulling its four corners until it forms a flat, top-down rectangle whose width is $W$ meters and whose length is $L$ meters.

1. **Solving for the Matrix:** When the operator clicks four points on the screen $\mathbf{P}_{src} = [(u_1, v_1), (u_2, v_2), (u_3, v_3), (u_4, v_4)]$ that correspond to a known physical rectangle on the asphalt $\mathbf{P}_{dst} = [(0, 0), (W, 0), (W, L), (0, L)]$, each corner pair gives two equations (one for $X$, one for $Y$). Setting up those eight linear equations and solving for the nine entries of the $3 \times 3$ transformation matrix $H$ is a mathematical procedure called the **Direct Linear Transform (DLT)**, executed inside `cv2.getPerspectiveTransform()`.
2. **Projecting the Tire-Contact Dot:** Why do we project the **bottom-center** of the bounding box $(u = \frac{x_1+x_2}{2}, v = y_2)$ instead of the box's center? Because the Planar Homography matrix $H$ is only mathematically valid for points that physically touch the flat 2D road plane ($Z = 0$). A truck's center sits 2 meters in the air ($Z = 2\text{ m}$); projecting an elevated point through a ground-plane matrix creates parallax error that inflates its calculated speed. The tires always sit on $Z = 0$.
3. **Checking Zone Boundaries:** Before computing speed, we must verify that the car's tire dot is actually inside our 4-point calibrated road zone. The software draws an imaginary ray from the tire dot `(u, v)` to the edge of the frame and counts how many times it crosses the polygon's boundary edges—an odd count means inside, an even count means outside. In computational geometry, this is a **Point-in-Polygon Test** (`cv2.pointPolygonTest`).

---

## 2. Eliminating Bounding-Box Jitter (Rolling Window + Median Filter)

Even when a car is parked or moving smoothly, YOLOv8's bounding box edges naturally expand or shrink by 1 to 2 pixels from frame to frame. 
- At 30 FPS ($\Delta t = 0.033\text{ s}$), a 2-pixel jitter representing $0.15\text{ meters}$ on the road creates an instant phantom speed spike of:
  $$\frac{0.15\text{ m}}{0.0333\text{ s}} \times 3.6 = 16.2\text{ km/h error!}$$
- **The Two-Stage Stabilizer:**
  1. **Temporal Window Span ($k \ge 8$ frames):** Instead of measuring frame-to-frame ($t - 1 \to t$), we store the last 15 projected meter coordinates `(timestamp, X_m, Y_m)` in a double-ended queue (`deque`). We measure total metric distance traveled between the oldest entry in the window and the newest entry ($\Delta t \approx 0.25\text{ to }0.50\text{ seconds}$), cutting pixel-jitter error by $10\times$.
  2. **Rolling Median Filter:** We store the last 5 calculated speeds and output their statistical median, completely rejecting single-frame outlier spikes.

---

## 3. Production Code: `core/homography_speed.py`

```python
import cv2
import numpy as np
from collections import defaultdict, deque


class HomographySpeedEngine:
    """
    Translates 2D image tire-contact pixels (u, v) into real-world road-plane
    coordinates (X, Y) in meters and computes smoothed velocity in km/h.
    """

    def __init__(
        self,
        src_points: list[list[float]],
        real_width_m: float,
        real_length_m: float,
        window_size: int = 15,
        min_samples: int = 8,
    ):
        self.window_size = window_size
        self.min_samples = min_samples

        # Per-vehicle trajectory buffer: track_id -> deque of (timestamp, X_m, Y_m)
        self.trajectories: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        # Per-vehicle smoothed speed buffer: track_id -> deque of recent km/h floats
        self.speed_buffer: dict[int, deque] = defaultdict(lambda: deque(maxlen=5))

        self.update_calibration(src_points, real_width_m, real_length_m)

    def update_calibration(
        self, src_points: list[list[float]], real_width_m: float, real_length_m: float
    ) -> None:
        if len(src_points) != 4:
            raise ValueError("Calibration requires exactly 4 non-collinear source points.")

        self.real_width_m = float(real_width_m)
        self.real_length_m = float(real_length_m)
        self.src_polygon = np.array(src_points, dtype=np.float32)

        # Destination top-down metric plane (0,0 -> W,0 -> W,L -> 0,L)
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
        # Clear stale trajectories when geometry changes
        self.trajectories.clear()
        self.speed_buffer.clear()

    def is_inside_zone(self, tire_px: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.src_polygon, (float(tire_px[0]), float(tire_px[1])), measureDist=False) >= 0

    def pixel_to_meters(self, tire_px: tuple[float, float]) -> tuple[float, float]:
        pt_array = np.array([[[tire_px[0], tire_px[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_array, self.H_matrix)[0][0]
        return float(transformed[0]), float(transformed[1])

    def compute_speed_kmh(
        self, track_id: int, tire_px: tuple[float, float], capture_time: float
    ) -> float | None:
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

        if delta_t <= 0.08:
            return None

        displacement_m = float(np.hypot(x_end - x_start, y_end - y_start))
        instant_kmh = (displacement_m / delta_t) * 3.6

        # Reject physically impossible glitch readings (> 220 km/h on campus/urban roads)
        if instant_kmh > 220.0:
            return None

        self.speed_buffer[track_id].append(instant_kmh)
        return float(np.median(self.speed_buffer[track_id]))
```
