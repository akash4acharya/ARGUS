import os
import cv2
import time
import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable, List, Dict, Any, Optional
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
    Decouples disk JPEG saving and PostgreSQL / SQLite INSERTs from the real-time vision loop.
    Enforces Skill 4: Non-Blocking Producer-Consumer Persistence.
    """

    def __init__(self, evidence_dir: str = "evidence_crops"):
        self.queue: asyncio.Queue[ViolationEvent] = asyncio.Queue(maxsize=250)
        self.cooldown_tracks: dict[int, float] = {}  # track_id -> timestamp
        self.cooldown_duration_sec = 15.0  # Cooldown before logging same vehicle ID again
        self.evidence_dir = evidence_dir
        self.recent_violations: List[Dict[str, Any]] = []
        os.makedirs(self.evidence_dir, exist_ok=True)

    def enqueue_from_vision_thread(
        self, loop: asyncio.AbstractEventLoop, event: ViolationEvent
    ) -> bool:
        """
        Thread-safe entry point invoked from the OpenCV / AI inference thread.
        """
        now = time.time()
        last_time = self.cooldown_tracks.get(event.track_id)
        if last_time and (now - last_time) < self.cooldown_duration_sec:
            return False

        self.cooldown_tracks[event.track_id] = now
        loop.call_soon_threadsafe(self._safe_put, event)
        return True

    def _safe_put(self, event: ViolationEvent) -> None:
        if not self.queue.full():
            self.queue.put_nowait(event)

    async def consume_forever(
        self, broadcast_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    ) -> None:
        """
        Asynchronous worker loop that drains the violation queue, writes crops to disk,
        inserts records into the database, and broadcasts WebSocket alerts.
        """
        while True:
            ev = await self.queue.get()
            try:
                ts_ms = int(time.time() * 1000)
                filename = f"cam{ev.camera_id}_id{ev.track_id}_{ts_ms}.jpg"
                disk_path = os.path.join(self.evidence_dir, filename)

                # Save JPEG crop on a worker thread so the async event loop never blocks
                if ev.vehicle_crop_bgr is not None and ev.vehicle_crop_bgr.size > 0:
                    await asyncio.to_thread(cv2.imwrite, disk_path, ev.vehicle_crop_bgr)
                else:
                    # Write small fallback image if crop is invalid
                    blank = np.zeros((120, 160, 3), dtype=np.uint8)
                    await asyncio.to_thread(cv2.imwrite, disk_path, blank)

                # Persist to Database (PostgreSQL or local SQLite fallback)
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
                    print(f"[WARN] Database persistence error, snapshot saved locally: {db_err}")

                alert_payload = {
                    "camera_id": ev.camera_id,
                    "track_id": ev.track_id,
                    "vehicle_class": ev.vehicle_class,
                    "speed_kmh": round(ev.measured_speed_kmh, 1),
                    "limit_kmh": ev.speed_limit_kmh,
                    "snapshot_url": f"/evidence/{filename}",
                    "timestamp": time.strftime("%H:%M:%S"),
                }

                # Maintain recent violations in memory (latest 50)
                self.recent_violations.insert(0, alert_payload)
                if len(self.recent_violations) > 50:
                    self.recent_violations.pop()

                # Push live JSON alert to all connected operator WebSockets
                if broadcast_callback:
                    try:
                        await broadcast_callback(alert_payload)
                    except Exception as ws_err:
                        print(f"[WARN] WebSocket broadcast error: {ws_err}")
            except Exception as outer_err:
                print(f"[ERROR] Error in violation consumer: {outer_err}")
            finally:
                self.queue.task_done()
