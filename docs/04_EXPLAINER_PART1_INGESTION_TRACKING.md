# 04 — A-TO-Z EXPLAINER (PART 1): Universal CCTV Ingestion, Perception & Multi-Object Tracking

## Purpose of This Document
This file covers the complete theory, failure modes, and production implementation for **Stage 1 (Stream Ingestion)** and **Stage 2 (Vehicle Perception & Tracking)** located in `core/stream_reader.py`, `core/detector.py`, and `core/tracker.py`.

---

## 1. Universal Piggybacking on Pre-Existing CCTV Software

### The Trap Most Developers Fall Into
When you connect to an existing campus NVR (Hikvision, Dahua, CP Plus, Frigate, or Shinobi) via an RTSP network link, the camera pushes 25 to 30 frames every second over the network. If you place a standard `cap.read()` inside the same loop that runs your neural network, your program cannot always finish its math in 33 milliseconds. 

Whenever a frame takes 50 milliseconds to process, the operating system holds the incoming network frames in a waiting line (a **FIFO Frame Buffer**). Within 60 seconds, your program is reading frames that are 10 seconds old. Worse, when that memory waiting line fills up, the network driver silently throws away chunks of frames to free up memory. Because speed is distance divided by time, dropped frames destroy your time measurement and produce completely fake speeds.

### How Our Architecture Solves It
We separate **grabbing the frame** from **analyzing the frame** using two different execution tracks:
1. **The Producer Track (Background Reader Thread):** A lightweight loop runs in the background. Its only job is to pull the latest packet off the network wire (`cap.grab()`), immediately stamp it with a high-precision hardware stopwatch reading (`time.monotonic()`), decode the image, and overwrite a single shared slot in memory.
2. **The Consumer Track (AI Vision Loop):** Whenever YOLOv8 and DeepSORT finish processing a frame, they reach into that shared memory slot and grab whichever frame is currently sitting there, along with its exact arrival stopwatch timestamp. Frames never queue up, memory never overflows, and the elapsed time between any two processed frames ($\Delta t$) is always measured by the hardware clock rather than guessed from the camera's nominal FPS.

---

## 2. How Detection and Tracking Work Together

### Step A: Finding Vehicles on the Screen (Object Detection)
A neural network scans the raw pixel grid of a single frame and predicts rectangular bounding boxes `[x1, y1, x2, y2]`, confidence scores (`0.0` to `1.0`), and class numbers (`2=car`, `3=motorcycle`, `5=bus`, `7=truck`). We use **YOLOv8** for this step. However, an object detector has zero memory of time—it treats Frame 1 and Frame 2 as completely unrelated photographs.

### Step B: Connecting the Dots Across Time (Multi-Object Tracking)
To measure speed, we must know that "Car #42" at pixel `(400, 300)` in Frame 1 is the exact same physical vehicle as the box at `(405, 312)` in Frame 2. **DeepSORT** solves this using three chained mechanisms:

1. **Predicting Where the Car Should Move Next:** Based on a vehicle's position and velocity in previous frames, a mathematical estimator guesses where its bounding box should appear in the current frame, along with an uncertainty zone around that guess. This predictor-corrector loop is a **Kalman Filter**.
2. **Measuring Motion & Visual Similarity:** 
   - First, the tracker measures how far each new detection box sits from the predicted trajectory, scaled by how uncertain that prediction is. This uncertainty-weighted spatial measurement is the **Mahalanobis Distance**.
   - Second, if a car passes behind a lamppost or a bus for 15 frames, the motion guess grows too uncertain. The tracker extracts a 128-number visual fingerprint of the car's paint color, roof shape, and aspect ratio, and compares it against the saved fingerprints of recently seen cars by measuring the angle between their feature vectors. This visual comparison is the **Cosine Distance** over **Appearance Embeddings**.
3. **Optimal Matchmaking Across the Entire Screen:** When 12 cars are moving on screen at once, the tracker builds a cost table combining motion distance and visual distance for every possible pair of old tracks and new detections. It then solves for the single global pairing that minimizes total mismatch error across all vehicles simultaneously. This mathematical assignment solver is the **Hungarian Algorithm**.

---

## 3. Production Code: `core/stream_reader.py`

```python
import os
import cv2
import time
import threading
from dataclasses import dataclass
import numpy as np


@dataclass
class TimestampedFrame:
    frame_id: int
    capture_time: float  # Monotonic hardware clock (seconds)
    image: np.ndarray


class UniversalCCTVStream:
    """
    Universal, zero-buffer stream reader compatible with pre-existing CCTV software
    (RTSP NVR restreams, ONVIF URLs, HTTP MJPEG streams, or local MP4 files).
    """

    def __init__(self, source_url: str | int, target_width: int = 1280, target_height: int = 720):
        self.source_url = source_url
        self.target_width = target_width
        self.target_height = target_height

        self._latest_frame: TimestampedFrame | None = None
        self._lock = threading.Lock()
        self._running = False
        self._frame_counter = 0
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> "UniversalCCTVStream":
        if isinstance(self.source_url, str) and self.source_url.lower().startswith("rtsp://"):
            # Force TCP transport to eliminate UDP packet loss (gray corrupted frames)
            # and disable internal FFmpeg buffering
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            self.cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source_url)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to connect to CCTV stream source: {self.source_url}")

        self._running = True
        self._thread = threading.Thread(target=self._ingest_worker, daemon=True)
        self._thread.start()
        return self

    def _ingest_worker(self) -> None:
        is_local_file = isinstance(self.source_url, str) and not self.source_url.lower().startswith(("rtsp://", "http://", "https://"))
        nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / nominal_fps if is_local_file else 0.0

        while self._running and self.cap is not None:
            loop_start = time.monotonic()
            grabbed = self.cap.grab()
            arrival_timestamp = time.monotonic()

            if not grabbed:
                if is_local_file:
                    # Loop local test video seamlessly for development
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Live RTSP stream dropped; wait briefly and reconnect
                time.sleep(1.0)
                self.cap.open(self.source_url)
                continue

            ret, raw_bgr = self.cap.retrieve()
            if ret and raw_bgr is not None:
                if (raw_bgr.shape[1], raw_bgr.shape[0]) != (self.target_width, self.target_height):
                    raw_bgr = cv2.resize(raw_bgr, (self.target_width, self.target_height))

                with self._lock:
                    self._frame_counter += 1
                    self._latest_frame = TimestampedFrame(
                        frame_id=self._frame_counter,
                        capture_time=arrival_timestamp,
                        image=raw_bgr,
                    )

            # If testing on a local MP4 file, pace reader to actual video FPS
            if is_local_file:
                elapsed = time.monotonic() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

    def get_latest(self) -> TimestampedFrame | None:
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
```
