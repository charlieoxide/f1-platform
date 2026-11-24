"""Ingest a session from OpenF1 into the local database - free, no account.

FastF1 now needs an F1TV subscription to read F1's timing server, so this is
the default ingestion path. It fills the same `laps` and `results` tables, so
the dashboard, predictor, summariser and chatbot work unchanged.

Mapping (checked against real OpenF1 responses for 2024 Monaco):
  laps          <- /laps (lap_duration may be null, e.g. a red-flag lap 1)
  compound/stint<- /stints; a lap belongs to the latest stint that started on or
                   before it. `lap_end` is NOT used: OpenF1 returns lap_end=0
                   for some stints, so it can't be trusted.
  results       <- /session_result (position, points, dnf/dns/dsq, laps done)
  status        <- derived: Finished, "+N Lap(s)" if a lap or more down,
                   DNF, DNS, DSQ (OpenF1 has no retirement reason)
  grid_position <- /starting_grid if it has data; otherwise the qualifying
                   classification, which ignores grid penalties. The source
                   used is reported so it is never silent.

Nothing is overwritten with an empty result: if OpenF1 has no results (yet) for
a session, results already stored for it are left alone.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from src.ingest import parse_session_arg, write_laps, write_results
from src.openf1 import OpenF1Error, _get, resolve_session
from src.schemas import LapRecord, ResultRecord

logger = logging.getLogger(__name__)

RACE_LIKE = ("R", "S")


def _acronyms(session_key: int) -> dict[int, str]:
    rows = _get("drivers", {"session_key": session_key})
    return {
        int(r["driver_number"]): str(r["name_acronym"]).upper()
        for r in rows
        if r.get("driver_number") is not None and r.get("name_acronym")
    }


def _group_stints(stints: list[dict]) -> dict[int, list[tuple[int, int, str | None]]]:
    """driver_number -> [(lap_start, stint_number, compound)] sorted by lap_start."""
    grouped: dict[int, list[tuple[int, int, str | None]]] = {}
    for s in stints:
        if s.get("driver_number") is None or s.get("lap_start") is None:
            continue
        compound = str(s["compound"]).upper() if s.get("compound") else None
        grouped.setdefault(int(s["driver_number"]), []).append(
            (int(s["lap_start"]), int(s.get("stint_number") or 1), compound)
        )
    for entries in grouped.values():
        entries.sort()
    return grouped


def _stint_for_lap(
    entries: list[tuple[int, int, str | None]], lap_number: int
) -> tuple[int | None, str | None]:
    current = None
    for lap_start, stint_number, compound in entries:
        if lap_start <= lap_number:
            current = (stint_number, compound)
        else:
            break
    return current if current else (None, None)


def build_lap_records(
    session_key: str,
    laps: list[dict],
    acronyms: dict[int, str],
    stints: list[dict],
) -> tuple[list[LapRecord], int]:
    """Validate every lap row; bad rows are skipped and logged, never stored."""
    by_driver = _group_stints(stints)
    best: dict[int, float] = {}
    records: list[LapRecord] = []
    skipped = 0
    ordered = sorted(
        laps, key=lambda r: (r.get("driver_number") or 0, r.get("lap_number") or 0)
    )
    for row in ordered:
        try:
            number = int(row["driver_number"])
            lap_no = int(row["lap_number"])
            raw = row.get("lap_duration")
            duration = float(raw) if raw is not None else None
            stint, compound = _stint_for_lap(by_driver.get(number, []), lap_no)
            is_best = duration is not None and duration < best.get(number, float("inf"))
            if is_best:
                best[number] = duration
            records.append(
                LapRecord(
                    session_key=session_key,
                    driver=acronyms[number],
                    lap_number=lap_no,
                    lap_time_seconds=duration,
                    compound=compound,
                    stint=stint,
                    is_personal_best=is_best,
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            skipped += 1
            logger.warning("Skipping invalid OpenF1 lap row: %s", e)
    return records, skipped


def _status(row: dict, laps_down: int) -> str:
    if row.get("dsq"):
        return "DSQ"
    if row.get("dns"):
        return "DNS"
    if row.get("dnf"):
        return "DNF"
    if laps_down > 0:
        return f"+{laps_down} Lap" + ("s" if laps_down > 1 else "")
    return "Finished"


def build_result_records(
    session_key: str,
    session_type: str,
    rows: list[dict],
    acronyms: dict[int, str],
    grid: dict[int, int],
    event_date: datetime | None,
) -> tuple[list[ResultRecord], int]:
    is_race = session_type.upper() in RACE_LIKE
    leader_laps = max(
        (int(r["number_of_laps"]) for r in rows if r.get("number_of_laps") is not None),
        default=0,
    )
    records: list[ResultRecord] = []
    skipped = 0
    for row in rows:
        try:
            number = int(row["driver_number"])
            position = int(row["position"]) if row.get("position") else None
            done = row.get("number_of_laps")
            laps_down = leader_laps - int(done) if is_race and done is not None else 0
            records.append(
                ResultRecord(
                    session_key=session_key,
                    session_type=session_type,
                    driver=acronyms[number],
                    position=position,
                    grid_position=(
                        grid.get(number) if session_type.upper() == "R" else None
                    ),
                    points=float(row.get("points") or 0.0),
                    status=_status(row, laps_down),
                    event_date=event_date,
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            skipped += 1
            logger.warning("Skipping invalid OpenF1 result row: %s", e)
    return records, skipped


def _positions(rows: list[dict]) -> dict[int, int]:
    out = {}
    for r in rows:
        if r.get("driver_number") is not None and r.get("position"):
            out[int(r["driver_number"])] = int(r["position"])
    return out


def resolve_grid(session: dict, session_type: str) -> tuple[dict[int, int], str]:
    """Starting positions for a race, and where they came from."""
    if session_type.upper() != "R":
        return {}, "not applicable"
    grid = _positions(_get("starting_grid", {"session_key": session["session_key"]}))
    if grid:
        return grid, "starting_grid"

    found = _get(
        "sessions",
        {"meeting_key": session["meeting_key"], "session_name": "Qualifying"},
    )
    if found:
        quali_key = found[0]["session_key"]
        grid = _positions(_get("starting_grid", {"session_key": quali_key}))
        if grid:
            return grid, "starting_grid (qualifying session)"
        grid = _positions(_get("session_result", {"session_key": quali_key}))
        if grid:
            return grid, "qualifying classification (grid penalties not reflected)"
    return {}, "unavailable"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(timezone.utc).replace(tzinfo=None)


def ingest_session_openf1(session_arg: str) -> dict:
    """session_arg like '2024 Monaco R'. Returns a summary of what was stored."""
    year, event, session_type = parse_session_arg(session_arg)
    session_key = f"{year}_{event.replace(' ', '')}_{session_type}"
    session = resolve_session(year, event, session_type)
    key = session["session_key"]

    acronyms = _acronyms(key)
    if not acronyms:
        raise OpenF1Error("OpenF1 returned no drivers for this session.")
    laps = _get("laps", {"session_key": key})
    stints = _get("stints", {"session_key": key})
    results = _get("session_result", {"session_key": key})
    if not laps and not results:
        raise OpenF1Error("OpenF1 has no lap or result data for this session yet.")

    grid, grid_source = resolve_grid(session, session_type)
    lap_records, lap_skipped = build_lap_records(session_key, laps, acronyms, stints)
    result_records, result_skipped = build_result_records(
        session_key,
        session_type,
        results,
        acronyms,
        grid,
        _parse_date(session.get("date_start")),
    )

    # Replace only with real data: never wipe stored rows with an empty answer.
    if lap_records:
        write_laps(session_key, lap_records, lap_skipped)
    else:
        logger.warning("No laps from OpenF1 for %s; existing laps kept.", session_key)
    if result_records:
        write_results(session_key, result_records, result_skipped)
    else:
        logger.warning(
            "No results from OpenF1 for %s; existing results kept.", session_key
        )

    return {
        "session_key": session_key,
        "laps": len(lap_records),
        "results": len(result_records),
        "skipped": lap_skipped + result_skipped,
        "grid_source": grid_source,
    }
