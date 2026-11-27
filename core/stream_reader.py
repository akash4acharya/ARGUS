import os
import cv2
import time
import threading
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class TimestampedFrame:
    frame_id: int
    capture_time: float  # Monotonic hardware clock (seconds)
    image: np.ndarray


class UniversalCCTVStream:
    """
    Universal, zero-buffer stream reader compatible with pre-existing CCTV software
    (RTSP NVR restreams, ONVIF URLs, HTTP MJPEG streams, webcams, or local MP4 files).
    
    Adheres strictly to Skill 1:
      - Runs stream ingestion in an isolated daemon threading.Thread.
      - Calls cap.grab() continuously and records time.monotonic() immediately upon packet arrival.
      - Stores only the latest frame in a thread-locked variable.
    """

    def __init__(
        self,
        source_url: Union[str, int] = 0,
        target_width: int = 1280,
        target_height: int = 720,
    ):
        self.source_url = source_url
        self.target_width = target_width
        self.target_height = target_height

        self._latest_frame: Optional[TimestampedFrame] = None
        self._lock = threading.Lock()
        self._running = False
        self._frame_counter = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "UniversalCCTVStream":
        if isinstance(self.source_url, str) and self.source_url.lower().startswith("rtsp://"):
            # Force TCP transport to eliminate UDP packet loss (gray corrupted frames)
            # and disable internal FFmpeg buffering
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
            self.cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source_url)

        # Set buffer size to 1 to minimize internal OS / OpenCV queue lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            # If camera 0 fails, we don't throw immediate fatal crash; we allow reconnect or mock frame
            print(f"[WARN] Unable to open CCTV source: {self.source_url}. Will attempt reconnection.")

        self._running = True
        self._thread = threading.Thread(target=self._ingest_worker, daemon=True, name="CCTVStreamReader")
        self._thread.start()
        return self

    def _ingest_worker(self) -> None:
        is_local_file = (
            isinstance(self.source_url, str)
            and not self.source_url.lower().startswith(("rtsp://", "http://", "https://"))
            and os.path.exists(self.source_url)
        )
        
        nominal_fps = 30.0
        if self.cap is not None and self.cap.isOpened():
            detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if detected_fps and detected_fps > 0:
                nominal_fps = detected_fps

        frame_interval = 1.0 / nominal_fps if is_local_file else 0.0

        while self._running:
            loop_start = time.monotonic()

            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                try:
                    self.cap = cv2.VideoCapture(self.source_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception as ex:
                    print(f"[WARN] Reconnection error: {ex}")
                continue

            grabbed = self.cap.grab()
            arrival_timestamp = time.monotonic()

            if not grabbed:
                if is_local_file and self.cap.isOpened():
                    # Loop local test video seamlessly for development
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Live RTSP / camera stream dropped; wait briefly and reconnect
                time.sleep(0.5)
                try:
                    self.cap.open(self.source_url)
                except Exception:
                    pass
                continue

            ret, raw_bgr = self.cap.retrieve()
            if ret and raw_bgr is not None:
                if (raw_bgr.shape[1], raw_bgr.shape[0]) != (self.target_width, self.target_height):
                    raw_bgr = cv2.resize(raw_bgr, (self.target_width, self.target_height))

                with self._lock:
                    self._frame_counter += 1
                    # For pre-recorded MP4 files, use deterministic video clock so OS/CPU jitter
                    # never artificially compresses delta_t or inflates measured speeds.
                    frame_ts = (self._frame_counter / nominal_fps) if is_local_file else arrival_timestamp
                    self._latest_frame = TimestampedFrame(
                        frame_id=self._frame_counter,
                        capture_time=frame_ts,
                        image=raw_bgr,
                    )

            # If testing on a local MP4 file, pace reader to actual video FPS
            if is_local_file:
                elapsed = time.monotonic() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

    def get_latest(self) -> Optional[TimestampedFrame]:
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
