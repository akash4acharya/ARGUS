# 02 — SKILL BUILD: Engineering Rules & Implementation Patterns

## Role & Execution Mandate for AI Coding Agent
When generating or modifying code for this project, you must strictly follow these five architectural rules. Violating any of these rules will corrupt the time-delta ($\Delta t$) math and ruin speed accuracy.

---

## Skill 1: Zero-Buffer RTSP/NVR Stream Ingestion
- **The Problem:** Pre-existing CCTV software (Hikvision, Dahua, Frigate, Shinobi, MediaMTX) outputs network streams via RTSP. Standard `cv2.VideoCapture.read()` queues frames in an internal waiting line (FIFO buffer). If AI inference slows down for 200ms, the stream falls behind real-time and drops packets silently.
- **Mandatory Pattern:**
  1. Run stream capture in an isolated daemon `threading.Thread`.
  2. Call `cap.grab()` continuously and record `time.monotonic()` immediately upon packet arrival.
  3. Store only the latest frame in a thread-locked variable (`threading.Lock()`). The inference loop must always pull the newest frame along with its hardware arrival timestamp.

---

## Skill 2: Bottom-Center Tire-Contact Projection
- **The Problem:** Using the center of a bounding box `(x_center, y_center)` for speed calculation fails because a tall truck's center sits meters above the road plane, creating parallax distortion.
- **Mandatory Pattern:**
  1. Extract bounding box coordinates as `(x1, y1, x2, y2)`.
  2. Compute the ground-plane contact point strictly as:
     $$u = \frac{x_1 + x_2}{2}, \quad v = y_2$$
  3. Pass `[u, v, 1.0]` into the $3 \times 3$ Homography matrix $H$ using `cv2.perspectiveTransform`.

---

## Skill 3: Rolling Window Trajectory Smoothing
- **The Problem:** Bounding boxes jitter by 1–3 pixels between consecutive frames due to detector noise. Calculating speed between Frame $t$ and Frame $t-1$ ($33\text{ ms}$) amplifies this 2-pixel jitter into massive fake speed spikes (e.g., jumping from $25\text{ km/h}$ to $90\text{ km/h}$).
- **Mandatory Pattern:**
  1. Maintain a `collections.deque(maxlen=15)` for each tracked `track_id`, storing tuples of `(timestamp, X_meters, Y_meters)`.
  2. Compute speed only after a vehicle has accumulated at least `N = 8` valid frames inside the calibration polygon.
  3. Calculate displacement over the window span ($\Delta t = t_{\text{latest}} - t_{\text{oldest}}$) and apply a median filter across the last 5 speed readings before triggering a violation.

---

## Skill 4: Non-Blocking Producer-Consumer Persistence
- **The Problem:** Writing a cropped image to disk or executing an `INSERT` query in PostgreSQL takes $5\text{–}40\text{ ms}$. Doing this inside the main video loop stalls frame ingestion.
- **Mandatory Pattern:**
  1. When `smoothed_speed > speed_limit` and `track_id not in logged_violations_set`:
  2. Copy the cropped vehicle numpy array in memory, package a `ViolationEvent` dictionary, and push it via `queue.put_nowait()` into an asynchronous worker queue.
  3. Add `track_id` to an in-memory cooldown cache so the same vehicle is not logged 30 times as it crosses the screen.

---

## Skill 5: Modular Verification & Testing Protocol
Every module must be testable in isolation without requiring a live physical camera:
- Allow a `--source` CLI flag that accepts either an RTSP URL (`rtsp://...`), an HTTP stream, or a local `.mp4` test file.
- Include a synthetic homography test that verifies `cv2.getPerspectiveTransform` maps the 4 source pixels to the exact metric width and length corners within `< 1e-4` floating-point tolerance.
