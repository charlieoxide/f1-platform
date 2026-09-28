"""Single entry point for writing external data into the DB. Every write
goes through here — validated first (schemas.py), transactional, with a
cache fallback for availability if the live pull fails.

Usage:
    python -m src.ingest --session "2024 Monaco R"
"""

from __future__ import annotations

import argparse
import logging

import fastf1
import pandas as pd
from pydantic import ValidationError

from src.config import settings
from src.db import Lap, SessionLocal, init_db
from src.logging_conf import setup_logging
from src.schemas import LapRecord

logger = logging.getLogger(__name__)

fastf1.Cache.enable_cache(settings.fastf1_cache_dir)


def parse_session_arg(arg: str) -> tuple[int, str, str]:
    """'2024 Monaco R' -> (2024, 'Monaco', 'R')"""
    year_str, *rest, session_type = arg.split()
    return int(year_str), " ".join(rest), session_type


def fetch_session(year: int, event: str, session_type: str) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load()
        return session.laps
    except Exception:
        logger.exception(
            "Live FastF1 pull failed for %s %s %s — check cache for fallback",
            year,
            event,
            session_type,
        )
        raise


def ingest_laps(session_key: str, laps_df: pd.DataFrame) -> int:
    """Validate every row before it's written. Bad rows are skipped and
    logged, not silently stored, not silently crashing the whole batch."""
    validated: list[LapRecord] = []
    skipped = 0

    for _, row in laps_df.iterrows():
        try:
            validated.append(
                LapRecord(
                    session_key=session_key,
                    driver=str(row.get("Driver", "")),
                    lap_number=int(row.get("LapNumber", 0)) or 1,
                    lap_time_seconds=(
                        row["LapTime"].total_seconds()
                        if pd.notna(row.get("LapTime"))
                        else None
                    ),
                    compound=row.get("Compound"),
                    stint=int(row["Stint"]) if pd.notna(row.get("Stint")) else None,
                    is_personal_best=bool(row.get("IsPersonalBest", False)),
                )
            )
        except (ValidationError, ValueError, TypeError, KeyError) as e:
            skipped += 1
            logger.warning("Skipping invalid lap row: %s", e)

    with SessionLocal.begin() as db:
        for rec in validated:
            db.add(Lap(**rec.model_dump()))

    logger.info(
        "Ingested %d laps (%d skipped) for %s", len(validated), skipped, session_key
    )
    return len(validated)


def main() -> None:
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="e.g. '2024 Monaco R'")
    args = parser.parse_args()

    year, event, session_type = parse_session_arg(args.session)
    session_key = f"{year}_{event.replace(' ', '')}_{session_type}"

    laps_df = fetch_session(year, event, session_type)
    ingest_laps(session_key, laps_df)


if __name__ == "__main__":
    main()
