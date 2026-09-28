"""DB layer. SQLite now, swappable to Postgres later by changing DB_PATH /
the engine URL — nothing else in the app should know which one is in use."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import settings

Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{settings.db_path}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Lap(Base):
    __tablename__ = "laps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_key = Column(String, index=True, nullable=False)
    driver = Column(String(3), index=True, nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time_seconds = Column(Float, nullable=True)
    compound = Column(String, nullable=True)
    stint = Column(Integer, nullable=True)
    is_personal_best = Column(Boolean, default=False)


class RaceControl(Base):
    __tablename__ = "race_control"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_key = Column(String, index=True, nullable=False)
    lap_number = Column(Integer, nullable=True)
    category = Column(String, nullable=True)
    message = Column(String, nullable=False)
    time = Column(DateTime, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
