"""Validation boundary for all external data. Nothing from FastF1/OpenF1
reaches the database without passing through one of these — this is the
integrity control for the CIA triad."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class LapRecord(BaseModel):
    session_key: str
    driver: str = Field(min_length=1, max_length=3)
    lap_number: int = Field(ge=1)
    lap_time_seconds: float | None = Field(default=None, ge=0)
    compound: str | None = None
    stint: int | None = Field(default=None, ge=1)
    is_personal_best: bool = False

    @field_validator("driver")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()


class SessionMeta(BaseModel):
    year: int = Field(ge=1950, le=2100)
    event_name: str
    session_type: str  # FP1/FP2/FP3/Q/S/R
    loaded_at: datetime


class RaceControlMessage(BaseModel):
    session_key: str
    lap_number: int | None = None
    category: str | None = None
    message: str = Field(min_length=1)
    time: datetime | None = None


class ResultRecord(BaseModel):
    """One driver's classification for one session. For a race: grid
    position, finishing position, points, status (Finished/Retired/etc).
    For qualifying: position is the quali result; grid/points are null."""

    session_key: str
    session_type: str  # 'Q' or 'R'
    driver: str = Field(min_length=1, max_length=3)
    position: int | None = Field(default=None, ge=1)
    grid_position: int | None = Field(default=None, ge=0)
    points: float = Field(default=0.0, ge=0)
    status: str | None = None

    @field_validator("driver")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()
