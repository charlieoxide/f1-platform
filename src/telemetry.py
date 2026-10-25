"""Loads lap telemetry from FastF1 for the driver coach (module 2).

Network-bound, so kept apart from coach.py (pure maths). Reuses ingest's
event-name guard, so a fuzzy-matched wrong race can't silently feed the
coach another session's telemetry.

Usage:
    python -m src.telemetry --session "2024 Monaco Q" --driver NOR --reference VER
    python -m src.telemetry --check "2024 Monaco R" "2023 Monaco Q"
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from src import openf1
from src.coach import compare_laps
from src.config import settings
from src.ingest import _check_event_match, parse_session_arg

logger = logging.getLogger(__name__)


class TelemetryUnavailableError(RuntimeError):
    """F1's servers did not provide car telemetry for this session."""


def load_session_with_telemetry(year: int, event: str, session_type: str):
    Path(settings.fastf1_cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(settings.fastf1_cache_dir)
    session = fastf1.get_session(year, event, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    _check_event_match(event, session)
    return session


def count_drivers_with_telemetry(session) -> int:
    try:
        return len(session.car_data)
    except DataNotLoadedError:
        return 0


def fastest_lap_telemetry(session, driver: str) -> pd.DataFrame:
    """Car data (Speed/Throttle/Brake) with a Distance column for the
    driver's fastest lap in this session."""
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        raise ValueError(f"No laps found for driver '{driver}' in this session.")
    lap = laps.pick_fastest()
    if lap is None:
        raise ValueError(f"Driver '{driver}' has no timed lap in this session.")
    try:
        car = lap.get_car_data()
    except (KeyError, DataNotLoadedError) as e:
        raise TelemetryUnavailableError(
            f"No car telemetry for '{driver}' in this session: F1's servers "
            "did not return it. Try another session, or run "
            "`python -m src.telemetry --check <sessions>` to see which have it."
        ) from e
    return car.add_distance()


def circuit_corners(session) -> list[tuple[str, float]] | None:
    """Approximate (name, distance) corner markers for 'Turn N' labels.
    Optional: if the corner map can't be fetched, the coach falls back to
    'Braking zone N' labels rather than failing."""
    try:
        corners = session.get_circuit_info().corners
        return [
            (f"{int(row.Number)}{row.Letter or ''}", float(row.Distance))
            for row in corners.itertuples()
        ]
    except Exception:  # noqa: BLE001 - labels are a nice-to-have
        logger.warning("Corner map unavailable; using generic zone labels.")
        return None


def coach_report(
    session_arg: str, driver: str, reference: str, source: str = "openf1"
) -> dict:
    """session_arg like '2024 Monaco Q'. Compares each driver's fastest
    lap in that session; the reference is usually the pole-sitter.

    source: 'openf1' (default - F1's own server now rejects FastF1 with
    HTTP 401) or 'fastf1' (kept for when it works again)."""
    driver, reference = driver.upper().strip(), reference.upper().strip()
    if driver == reference:
        raise ValueError("Pick two different drivers to compare.")
    if source == "openf1":
        return openf1.coach_report_openf1(session_arg, driver, reference)
    if source != "fastf1":
        raise ValueError(f"Unknown telemetry source '{source}'.")
    year, event, session_type = parse_session_arg(session_arg)
    session = load_session_with_telemetry(year, event, session_type)
    return compare_laps(
        fastest_lap_telemetry(session, driver),
        fastest_lap_telemetry(session, reference),
        corners=circuit_corners(session),
    )


def check_sessions(session_args: list[str]) -> list[tuple[str, str]]:
    """For each session, report how many drivers F1 returned car data for.
    Sessions are loaded one at a time and released, to stay inside this
    machine's RAM."""
    rows = []
    for arg in session_args:
        try:
            year, event, session_type = parse_session_arg(arg)
            session = load_session_with_telemetry(year, event, session_type)
            n = count_drivers_with_telemetry(session)
            status = f"OK - car data for {n} drivers" if n else "UNAVAILABLE"
            del session
        except Exception as e:  # noqa: BLE001 - report and keep checking
            status = f"ERROR - {type(e).__name__}: {e}"
        gc.collect()
        rows.append((arg, status))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="e.g. '2024 Monaco Q'")
    parser.add_argument("--driver", help="e.g. NOR")
    parser.add_argument("--reference", help="e.g. VER")
    parser.add_argument(
        "--check",
        nargs="+",
        metavar="SESSION",
        help="probe FastF1 telemetry availability",
    )
    parser.add_argument("--source", choices=["openf1", "fastf1"], default="openf1")
    args = parser.parse_args()

    if args.check:
        print("\nTelemetry availability:")
        for arg, status in check_sessions(args.check):
            print(f"  {arg:<20} {status}")
        return

    if not (args.session and args.driver and args.reference):
        parser.error("give --session, --driver and --reference (or use --check)")

    result = coach_report(args.session, args.driver, args.reference, args.source)
    print(
        f"\n{args.driver} vs {args.reference}: {result['total_delta_s']:+.3f} s "
        "over the lap (+ = behind the reference)\n"
    )
    times = result.get("lap_times_s")
    if times:
        d, r = args.driver.upper(), args.reference.upper()
        print(
            f"Official lap times: {d} {times[d]:.3f} s, {r} {times[r]:.3f} s "
            f"(gap {times[d] - times[r]:+.3f} s)\n"
        )
    for c in result["corners"]:
        print(" -", c.message)


if __name__ == "__main__":
    main()
