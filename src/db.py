"""DB layer. SQLite now, swappable to Postgres later by changing DB_PATH /
the engine URL — nothing else in the app should know which one is in use."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import settings

logger = logging.getLogger(__name__)

Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{settings.db_path}")
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


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_key = Column(String, index=True, nullable=False)
    session_type = Column(String, nullable=False)
    driver = Column(String(3), index=True, nullable=False)
    position = Column(Integer, nullable=True)
    grid_position = Column(Integer, nullable=True)
    points = Column(Float, default=0.0)
    status = Column(String, nullable=True)
    event_date = Column(DateTime, nullable=True)


class RaceControl(Base):
    __tablename__ = "race_control"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_key = Column(String, index=True, nullable=False)
    lap_number = Column(Integer, nullable=True)
    category = Column(String, nullable=True)
    message = Column(String, nullable=False)
    time = Column(DateTime, nullable=True)


def _migrate_missing_columns() -> None:
    """SQLAlchemy's create_all() only creates tables that don't exist yet -
    it never alters an existing table, so adding a column to a model here
    silently does nothing to a database file that already has that table.
    This adds any column the model declares but the actual table is
    missing, via SQLite's ALTER TABLE ADD COLUMN.

    Deliberately simple (additive only - no renames, no drops, no type
    changes) rather than pulling in a full migration tool like Alembic:
    for a single-developer SQLite project, every schema change so far has
    been "add a new column," and this covers that case directly without
    the overhead of migration-file management."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table in Base.metadata.tables.values():
        if table.name not in existing_tables:
            continue  # brand-new table, create_all() already handled it
        existing_cols = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing_cols:
                continue
            col_type = col.type.compile(engine.dialect)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}'
                    )
                )
            logger.info("Migrated: added column %s.%s", table.name, col.name)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_missing_columns()
