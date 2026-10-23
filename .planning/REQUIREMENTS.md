# REQUIREMENTS: ARGUS CCTV Speed & Telemetry Suite

## Functional Requirements
- **FR-1 [Stream Ingestion]:** Support RTSP, ONVIF, HTTP, webcams, and local MP4 video files via `UniversalCCTVStream` with zero FIFO buffer buildup and hardware monotonic clock stamping.
- **FR-2 [Vehicle Perception]:** Detect COCO vehicle classes (car, motorcycle, bus, truck) using YOLOv8 with configurable confidence and IOU thresholds.
- **FR-3 [Multi-Object Tracking]:** Persistently track vehicle instances across frames using DeepSORT Kalman filtering and appearance cosine distance.
- **FR-4 [Tire Anchor Projection]:** Compute bottom-center tire contact coordinates $(u, v) = ((x_1+x_2)/2, y_2)$ on the ground plane $Z = 0$.
- **FR-5 [Planar Homography Engine]:** Compute $3 \times 3$ IPM perspective transformation matrix from 4 clicked source pixels to real-world metric rectangle $(W, L)$.
- **FR-6 [Velocity Estimation]:** Calculate smoothed ground-plane velocity in km/h using rolling window ($\ge 8$ samples) and median filtering over recent speed outputs.
- **FR-7 [Exception-First HUD]:** Render corner brackets for nominal traffic, full amber box for warning ($\ge 85\%$), full crimson box for violation ($\ge 100\%$), 5px tire contact dot, and floating telemetry pill.
- **FR-8 [Asynchronous Evidence Logging]:** Enqueue speeding vehicle crops and metadata to background worker without stalling vision loop, saving crops to `evidence_crops/` and rows to DB.
- **FR-9 [Operator Web Dashboard]:** Present 3-Zone fixed layout: Top Telemetry Bar (64px), 70% Primary Video Canvas with draggable calibration handles, 30% Live Violation Audit Feed with WebSocket push.
- **FR-10 [Dynamic Calibration]:** Support updating 4 calibration anchor points and metric dimensions live via REST API without restarting stream.

## Non-Functional & Verification Requirements
- **NFR-1 [Mathematical Precision]:** Homography matrix DLT forward and inverse projection error must be $< 1e-4$ meters.
- **NFR-2 [Frame Rate Stability]:** Video feed and inference loop must maintain target 25-30 FPS on standard hardware.
- **NFR-3 [Jitter Rejection]:** Bounding-box detector noise ($\pm 1.5$ px) must not cause speed fluctuations $> 1.5$ km/h.
- **NFR-4 [Offline / Fallback Resilience]:** Automatically fall back to SQLite when PostgreSQL server is absent, and provide robust tracking fallback if DeepSORT embedder is unavailable.
