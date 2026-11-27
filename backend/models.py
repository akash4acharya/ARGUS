from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, DateTime, Text


class Base(DeclarativeBase):
    pass


class SpeedViolationModel(Base):
    __tablename__ = "speed_violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_class: Mapped[str] = mapped_column(String(32), nullable=False)
    measured_speed_kmh: Mapped[float] = mapped_column(Float, nullable=False)
    speed_limit_kmh: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot_path: Mapped[str] = mapped_column(Text, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


# Pydantic Schemas for API Requests & Responses

class CalibrationSchema(BaseModel):
    src_points: List[List[float]] = Field(
        ...,
        description="4 points (x, y) defining the calibration polygon: [TL, TR, BR, BL]",
    )
    real_width_m: float = Field(..., gt=0.0, description="Real road width in meters")
    real_length_m: float = Field(..., gt=0.0, description="Real road length in meters")


class SettingsSchema(BaseModel):
    speed_limit_kmh: Optional[float] = Field(None, gt=0.0)
    source: Optional[Union[str, int]] = Field(None)


class CameraSwitchSchema(BaseModel):
    preset_id: str = Field(..., description="Camera preset ID to activate")


class ViolationItemSchema(BaseModel):
    id: Optional[int] = None
    camera_id: int
    track_id: int
    vehicle_class: str
    measured_speed_kmh: float
    speed_limit_kmh: float
    snapshot_url: str
    captured_at: str


class TelemetryStatusSchema(BaseModel):
    fps: float
    delta_t_ms: float
    active_tracks: int
    speed_limit_kmh: float
    camera_source: str
    stream_status: str
