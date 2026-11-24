"""Single entry point for writing external data into the DB.

Every write goes through here: validated first (schemas.py), transactional,
idempotent (re-running a session replaces its rows), with a logged failure
path if the live pull fails. Pulls both laps and session results (grid/
finishing position, points, status) - results are what the Module 4
predictor trains on.

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
from src.db import Lap, Result, SessionLocal, init_db
from src.logging_conf import setup_logging
from src.schemas import LapRecord, ResultRecord

logger = logging.getLogger(__name__)


def _clean(value):
    """Turn pandas NaN/NaT into None so pydantic sees a real 'missing'."""
    return None if pd.isna(value) else value


def parse_session_arg(arg: str) -> tuple[int, str, str]:
    """'2024 Monaco R' -> (2024, 'Monaco', 'R')"""
    year_str, *rest, session_type = arg.split()
    return int(year_str), " ".join(rest), session_type


class EventMismatchError(ValueError):
    """FastF1 fuzzy-matched the requested event name to something that
    doesn't actually correspond to it. Silently continuing would ingest
    the wrong race's data under the requested race's label - exactly the
    kind of thing the integrity layer exists to catch."""


def _check_event_match(requested_event: str, session) -> None:
    actual_name = str(session.event.get("EventName", ""))
    actual_location = str(session.event.get("Location", ""))
    requested = requested_event.lower()
    haystack = f"{actual_name} {actual_location}".lower()

    # Loose on purpose: "Saudi Arabia" should match "Saudi Arabian Grand
    # Prix" (one word rewritten), "Monaco" should match "Monaco Grand
    # Prix" (exact substring). What it must NOT do is accept a completely
    # different event FastF1's fuzzy matcher guessed at.
    requested_words = [w for w in requested.split() if len(w) > 3]
    if requested_words and not any(w in haystack for w in requested_words):
        raise EventMismatchError(
            f"Requested event '{requested_event}' does not match what "
            f"FastF1 actually loaded ('{actual_name}'). This usually means "
            "FastF1's fuzzy name matching picked the wrong race - use the "
            "exact official event name or a round number instead."
        )


def fetch_session(year: int, event: str, session_type: str):
    Path(settings.fastf1_cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(settings.fastf1_cache_dir)
    try:
        session = fastf1.get_session(year, event, session_type)
        session.load(telemetry=False, weather=False, messages=False)
        _check_event_match(event, session)
        return session
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


def results_to_records(
    session_key: str,
    session_type: str,
    results_df: pd.DataFrame,
    event_date=None,
) -> tuple[list[ResultRecord], int]:
    """Same validate-and-skip pattern as laps_to_records, for the results
    table (grid/finishing position, points, status, event_date)."""
    records: list[ResultRecord] = []
    skipped = 0
    for _, row in results_df.iterrows():
        try:
            position = _clean(row.get("Position"))
            grid = _clean(row.get("GridPosition"))
            points = _clean(row.get("Points"))
            records.append(
                ResultRecord(
                    session_key=session_key,
                    session_type=session_type,
                    driver=str(row.get("Abbreviation", "")),
                    position=int(position) if position is not None else None,
                    grid_position=int(grid) if grid is not None else None,
                    points=float(points) if points is not None else 0.0,
                    status=_clean(row.get("Status")),
                    event_date=event_date,
                )
            )
        except (ValidationError, ValueError, TypeError, KeyError) as e:
            skipped += 1
            logger.warning("Skipping invalid result row: %s", e)
    return records, skipped


def write_laps(session_key: str, records: list[LapRecord], skipped: int = 0) -> int:
    """One transaction: replace this session's laps atomically. Shared by
    every ingestion source, so they all write the same way."""
    with SessionLocal.begin() as db:
        db.execute(delete(Lap).where(Lap.session_key == session_key))
        db.add_all(Lap(**r.model_dump()) for r in records)
    logger.info(
        "Ingested %d laps (%d skipped) for %s", len(records), skipped, session_key
    )
    return len(records)


def write_results(
    session_key: str, records: list[ResultRecord], skipped: int = 0
) -> int:
    with SessionLocal.begin() as db:
        db.execute(delete(Result).where(Result.session_key == session_key))
        db.add_all(Result(**r.model_dump()) for r in records)
    logger.info(
        "Ingested %d results (%d skipped) for %s", len(records), skipped, session_key
    )
    return len(records)


def ingest_laps(session_key: str, laps_df: pd.DataFrame) -> int:
    records, skipped = laps_to_records(session_key, laps_df)
    return write_laps(session_key, records, skipped)


def ingest_results(
    session_key: str,
    session_type: str,
    results_df: pd.DataFrame,
    event_date=None,
) -> int:
    records, skipped = results_to_records(
        session_key, session_type, results_df, event_date=event_date
    )
    return write_results(session_key, records, skipped)


def main() -> None:
    setup_logging()
    init_db()

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="e.g. '2024 Monaco R'")
    parser.add_argument(
        "--source",
        choices=["openf1", "fastf1"],
        default="openf1",
        help="openf1 is free; fastf1 needs an F1TV subscription (default: openf1)",
    )
    args = parser.parse_args()

    if args.source == "openf1":
        from src.ingest_openf1 import ingest_session_openf1

        summary = ingest_session_openf1(args.session)
        print(
            f"Done: {summary['laps']} laps, {summary['results']} results stored "
            f"for {summary['session_key']} (grid positions: {summary['grid_source']})"
        )
        return

    year, event, session_type = parse_session_arg(args.session)
    session_key = f"{year}_{event.replace(' ', '')}_{session_type}"

    session = fetch_session(year, event, session_type)
    event_date = getattr(session, "date", None)
    lap_count = ingest_laps(session_key, session.laps)
    result_count = ingest_results(
        session_key, session_type, session.results, event_date=event_date
    )
    print(f"Done: {lap_count} laps, {result_count} results stored for {session_key}")


if __name__ == "__main__":
    main()
