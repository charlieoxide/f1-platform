"""Single entry point for writing external data into the DB.

Every write goes through here: validated first (schemas.py), transactional,
idempotent (re-running a session replaces its rows), with a logged failure
path if the live pull fails.

Usage:
    python -m src.ingest --session "2024 Monaco R"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd
from pydantic import ValidationError
from sqlalchemy import delete

from src.config import settings
from src.db import Lap, SessionLocal, init_db
from src.logging_conf import setup_logging
from src.schemas import LapRecord

logger = logging.getLogger(__name__)


def _clean(value):
    """Turn pandas NaN/NaT into None so pydantic sees a real 'missing'."""
    return None if pd.isna(value) else value


def parse_session_arg(arg: str) -> tuple[int, str, str]:
    """'2024 Monaco R' -> (2024, 'Monaco', 'R')"""
    year_str, *rest, session_type = arg.split()
    return int(year_str), " ".join(rest), session_type


def fetch_session(year: int, event: str, session_type: str) -> pd.DataFrame:
    Path(settings.fastf1_cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(settings.fastf1_cache_dir)
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load(telemetry=False, weather=False, messages=False)
        return session.laps
    except Exception:  # noqa: BLE001 - log full context, then re-raise
        logger.exception("FastF1 pull failed for %s %s %s", year, event, session_type)
        raise


def laps_to_records(
    session_key: str, laps_df: pd.DataFrame
) -> tuple[list[LapRecord], int]:
    """Validate every row. Bad rows are skipped and logged, never stored."""
    records: list[LapRecord] = []
    skipped = 0
    for _, row in laps_df.iterrows():
        try:
            lap_time = _clean(row.get("LapTime"))
            stint = _clean(row.get("Stint"))
            records.append(
                LapRecord(
                    session_key=session_key,
                    driver=str(row.get("Driver", "")),
                    lap_number=int(row["LapNumber"]),
                    lap_time_seconds=(
                        lap_time.total_seconds() if lap_time is not None else None
                    ),
                    compound=_clean(row.get("Compound")),
                    stint=int(stint) if stint is not None else None,
                    is_personal_best=bool(_clean(row.get("IsPersonalBest")) or False),
                )
            )
        except (ValidationError, ValueError, TypeError, KeyError) as e:
            skipped += 1
            logger.warning("Skipping invalid lap row: %s", e)
    return records, skipped


def ingest_laps(session_key: str, laps_df: pd.DataFrame) -> int:
    records, skipped = laps_to_records(session_key, laps_df)
    # One transaction: replace this session's laps atomically.
    with SessionLocal.begin() as db:
        db.execute(delete(Lap).where(Lap.session_key == session_key))
        db.add_all(Lap(**r.model_dump()) for r in records)
    logger.info(
        "Ingested %d laps (%d skipped) for %s", len(records), skipped, session_key
    )
    return len(records)


def main() -> None:
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="e.g. '2024 Monaco R'")
    args = parser.parse_args()

    year, event, session_type = parse_session_arg(args.session)
    session_key = f"{year}_{event.replace(' ', '')}_{session_type}"

    laps_df = fetch_session(year, event, session_type)
    count = ingest_laps(session_key, laps_df)
    print(f"Done: {count} laps stored for {session_key}")


if __name__ == "__main__":
    main()
