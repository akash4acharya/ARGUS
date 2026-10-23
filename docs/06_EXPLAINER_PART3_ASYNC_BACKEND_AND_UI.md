# 06 — A-TO-Z EXPLAINER (PART 3): Asynchronous Backend, FastAPI Server & Interactive Calibration UI

## Purpose of This Document
This file covers **Stage 4 (Asynchronous Persistence & Operator Dashboard)** located in `backend/database.py`, `backend/worker.py`, `backend/main.py`, and `static/index.html`, plus the exact execution prompts for your coding agent.

---

## 1. How the Backend Streams Video and Logs Violations Without Lag

### Mechanism A: Continuous Photo Replacement (MJPEG Streaming)
Browsers cannot natively play raw RTSP security camera feeds or OpenCV NumPy arrays. Encoding H.264 video on the fly adds 2 seconds of lag. Instead, FastAPI opens a single HTTP connection and continuously pushes compressed JPEG frames separated by a text boundary marker, instructing the browser's `<img>` tag to immediately replace the previous frame with the newest one at 30 FPS. This zero-latency technique is **Motion JPEG (MJPEG)** over an **HTTP Multipart Stream**.

### Mechanism B: Instant Two-Way Alert Tunnel (WebSockets)
Normally, a web page only receives new data when the user clicks refresh. To make a red violation card pop onto the right-hand audit sidebar the exact millisecond a vehicle exceeds the speed limit, the browser and FastAPI keep a permanent, two-way communication pipe open in the background. This persistent two-way channel is a **WebSocket**.

### Mechanism C: Pre-Warmed Database Connections (Connection Pooling)
Opening a fresh login handshake with PostgreSQL takes 15 to 40 milliseconds. Instead of logging in every time a car speeds, SQLAlchemy opens 5 persistent connections when the app boots up, keeps them warm in memory, and lends one to the background consumer worker whenever a violation needs to be written. This pre-warmed cache of database links is a **Database Connection Pool**.

---

## 2. Production Code: `backend/database.py` & `backend/worker.py`

```python
# backend/database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/cctv_speed_db",
)

engine = create_async_engine(DATABASE_URL, pool_size=5, max_overflow=10, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS speed_violations (
                    id BIGSERIAL PRIMARY KEY,
                    camera_id INT NOT NULL DEFAULT 1,
                    track_id INT NOT NULL,
                    vehicle_class VARCHAR(32) NOT NULL,
                    measured_speed_kmh FLOAT NOT NULL,
                    speed_limit_kmh FLOAT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        )
```

```python
# backend/worker.py
import os
import cv2
import time
import asyncio
from dataclasses import dataclass
import numpy as np
from sqlalchemy import text
from backend.database import AsyncSessionLocal


@dataclass
class ViolationEvent:
    camera_id: int
    track_id: int
    vehicle_class: str
    measured_speed_kmh: float
    speed_limit_kmh: float
    vehicle_crop_bgr: np.ndarray


class AsyncViolationWorker:
    """
    Decouples disk JPEG saving and PostgreSQL INSERTs from the real-time vision loop.
    """

    def __init__(self, evidence_dir: str = "evidence_crops"):
        self.queue: asyncio.Queue[ViolationEvent] = asyncio.Queue(maxsize=250)
        self.cooldown_tracks: set[int] = set()
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)

    def enqueue_from_vision_thread(
        self, loop: asyncio.AbstractEventLoop, event: ViolationEvent
    ) -> bool:
        if event.track_id in self.cooldown_tracks:
            return False
        self.cooldown_tracks.add(event.track_id)
        loop.call_soon_threadsafe(self._safe_put, event)
        return True

    def _safe_put(self, event: ViolationEvent) -> None:
        if not self.queue.full():
            self.queue.put_nowait(event)

    async def consume_forever(self, broadcast_callback) -> None:
        while True:
            ev = await self.queue.get()
            try:
                ts_ms = int(time.time() * 1000)
                filename = f"cam{ev.camera_id}_id{ev.track_id}_{ts_ms}.jpg"
                disk_path = os.path.join(self.evidence_dir, filename)

                # Save JPEG crop on a worker thread so the async event loop never blocks
                await asyncio.to_thread(cv2.imwrite, disk_path, ev.vehicle_crop_bgr)

                # Persist to PostgreSQL (gracefully logs to console if DB is offline during local dev)
                try:
                    async with AsyncSessionLocal() as session:
                        await session.execute(
                            text(
                                """
                                INSERT INTO speed_violations
                                (camera_id, track_id, vehicle_class, measured_speed_kmh, speed_limit_kmh, snapshot_path)
                                VALUES (:cam, :tid, :cls, :spd, :lim, :path)
                                """
                            ),
                            {
                                "cam": ev.camera_id,
                                "tid": ev.track_id,
                                "cls": ev.vehicle_class,
                                "spd": round(ev.measured_speed_kmh, 2),
                                "lim": ev.speed_limit_kmh,
                                "path": disk_path,
                            },
                        )
                        await session.commit()
                except Exception as db_err:
                    print(f"[WARN] PostgreSQL offline, saved snapshot locally: {db_err}")

                # Push live JSON alert to all connected operator WebSockets
                await broadcast_callback(
                    {
                        "track_id": ev.track_id,
                        "vehicle_class": ev.vehicle_class,
                        "speed_kmh": round(ev.measured_speed_kmh, 1),
                        "limit_kmh": ev.speed_limit_kmh,
                        "snapshot_url": f"/evidence/{filename}",
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                )
            finally:
                self.queue.task_done()
```
