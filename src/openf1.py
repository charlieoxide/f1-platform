"""OpenF1 telemetry source for the driver coach (module 2).

Why this exists: F1's own timing server now answers FastF1's requests with
HTTP 401, so FastF1 cannot download car telemetry. OpenF1 (api.openf1.org)
is a free API serving the same channels. It has no distance channel, so
distance is built by integrating speed over time, and the lap's start and
end are interpolated from the timestamps so both laps cover a full lap.

Resolution is ~3.8 samples/second (~20 m between samples at full speed),
so braking points are only good to roughly 10-20 m. Fine for spotting real
differences; not for splitting hairs.

Safety rules shared with the rest of the project:
  * event/driver names must match clearly or the call fails with the
    candidates listed - never a silent best guess;
  * responses are cached on disk (historical data never changes), and a
    corrupt cache file is ignored, not trusted;
  * rate limits (HTTP 429) are retried with a short back-off.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.coach import compare_laps
from src.config import settings
from src.ingest import parse_session_arg

BASE_URL = "https://api.openf1.org/v1"
MAX_RETRIES = 3
PAD_S = 0.5  # fetch a little beyond the lap so its ends can be interpolated
MIN_SAMPLES = 20

SESSION_NAMES = {
    "Q": "Qualifying",
    "R": "Race",
    "S": "Sprint",
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
}


class OpenF1Error(RuntimeError):
    """OpenF1 was unreachable, refused the request, or had no usable data."""


# ---------------------------------------------------------------- HTTP layer


def _http_get(url: str) -> tuple[int, bytes]:
    """GET using urllib on purpose: it sends the URL exactly as written.
    httpx percent-encodes the '>' and '<' in OpenF1's date filters, and the
    raw form is the one verified to work."""
    request = urllib.request.Request(url, headers={"User-Agent": "f1-platform"})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:  # 4xx/5xx still carry a status + body
        return e.code, e.read()
    except OSError as e:  # DNS failure, refused, timeout, ...
        raise OpenF1Error(f"Could not reach OpenF1: {e}") from e


def _cache_dir() -> Path:
    return Path(settings.fastf1_cache_dir).parent / "openf1_cache"


def _build_url(endpoint: str, eq: dict, raw: list[str] | tuple = ()) -> str:
    """eq -> key=value pairs (values URL-quoted); raw -> OpenF1's operator
    filters written verbatim, e.g. 'date>=2024-05-25T14:26:53.000'."""
    parts = [f"{k}={quote(str(v), safe='')}" for k, v in eq.items()] + list(raw)
    return f"{BASE_URL}/{endpoint}?{'&'.join(parts)}"


def _cache_path(url: str) -> Path:
    return _cache_dir() / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")


def _cache_read(url: str):
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None  # corrupt cache entry: ignore it and refetch


def _cache_write(url: str, data) -> None:
    path = _cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)  # atomic: never leaves a half-written file


def _get(endpoint: str, eq: dict | None = None, raw: list[str] | tuple = ()):
    url = _build_url(endpoint, eq or {}, raw)
    cached = _cache_read(url)
    if cached is not None:
        return cached

    for attempt in range(MAX_RETRIES):
        status, body = _http_get(url)
        if status != 429:
            break
        time.sleep(1.0 * (attempt + 1))
    else:
        raise OpenF1Error("OpenF1 rate limit hit; wait a minute and retry.")

    if status == 404:
        return []  # OpenF1's "No results found."
    if status >= 400:
        raise OpenF1Error(f"OpenF1 returned HTTP {status}.")

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise OpenF1Error("OpenF1 returned a response that is not valid JSON.") from e
    if isinstance(data, dict):
        detail = str(data.get("detail", data))
        if "no results" in detail.lower():
            return []
        raise OpenF1Error(f"OpenF1 error: {detail}")
    if data:
        _cache_write(url, data)
    return data


# ------------------------------------------------------------ name resolving


def resolve_session(year: int, event: str, session_type: str) -> dict:
    name = SESSION_NAMES.get(session_type.upper())
    if name is None:
        raise ValueError(
            f"Unknown session type '{session_type}'. Use one of: "
            f"{', '.join(SESSION_NAMES)}."
        )
    rows = _get("sessions", {"year": year, "session_name": name})
    if not rows:
        raise OpenF1Error(f"OpenF1 has no {name} sessions for {year}.")

    wanted = event.strip().lower()

    def fields(r: dict) -> list[str]:
        keys = ("country_name", "location", "circuit_short_name")
        return [str(r.get(k) or "").lower() for k in keys]

    matches = [r for r in rows if wanted in fields(r)]
    if not matches and len(wanted) >= 4:
        matches = [r for r in rows if any(wanted in f for f in fields(r))]

    available = sorted({f"{r.get('country_name')} / {r.get('location')}" for r in rows})
    if not matches:
        raise OpenF1Error(
            f"No {year} {name} session matches '{event}'. Use a country or "
            f"location from: {'; '.join(available)}"
        )
    if len({r.get("meeting_key") for r in matches}) > 1:
        names = "; ".join(
            f"{r.get('country_name')} / {r.get('location')}" for r in matches
        )
        raise OpenF1Error(
            f"'{event}' matches several {year} events ({names}). "
            "Use the location name instead."
        )
    return matches[0]


def resolve_driver(session_key: int, driver: str) -> int:
    """Three-letter code (NOR) or a plain driver number (4) -> number."""
    driver = driver.strip()
    if driver.isdigit():
        return int(driver)
    rows = _get("drivers", {"session_key": session_key})
    for r in rows:
        if str(r.get("name_acronym") or "").upper() == driver.upper():
            return int(r["driver_number"])
    codes = sorted(str(r.get("name_acronym")) for r in rows if r.get("name_acronym"))
    raise OpenF1Error(
        f"Driver '{driver}' not found in this session. Available: {', '.join(codes)}"
    )


def fastest_lap(session_key: int, driver_number: int) -> dict:
    """Fastest timed, non-pit-out lap. (OpenF1 does not flag deleted laps,
    so a lap later deleted for track limits can still show as fastest.)"""
    laps = _get("laps", {"session_key": session_key, "driver_number": driver_number})
    timed = [
        lap
        for lap in laps
        if lap.get("lap_duration")
        and lap.get("date_start")
        and not lap.get("is_pit_out_lap")
    ]
    if not timed:
        raise OpenF1Error(f"No timed laps for driver {driver_number} in this session.")
    return min(timed, key=lambda lap: lap["lap_duration"])


# ----------------------------------------------------------------- telemetry


def _naive_utc(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .replace(tzinfo=None)
        .isoformat(timespec="milliseconds")
    )


def build_lap_frame(
    samples: list[dict], lap_start: datetime, lap_duration: float
) -> pd.DataFrame:
    """OpenF1 car_data samples -> the Distance/Speed/Throttle/Brake frame
    the coach expects. Brake is rescaled from 0/100 to 0/1; distance is the
    integral of speed; the lap's exact start and end are interpolated from
    the surrounding samples so the integral covers the whole lap."""
    rows = [s for s in samples if s.get("speed") is not None and s.get("date")]
    rows.sort(key=lambda s: s["date"])
    t = np.array(
        [(datetime.fromisoformat(s["date"]) - lap_start).total_seconds() for s in rows]
    )
    speed = np.array([float(s["speed"]) for s in rows])
    throttle = np.array([float(s.get("throttle") or 0) for s in rows])
    brake = np.array([float(s.get("brake") or 0) / 100.0 for s in rows])

    inside = t[(t > 0) & (t < lap_duration)]
    grid = np.concatenate([[0.0], inside, [lap_duration]])
    v = np.interp(grid, t, speed)
    dist = np.concatenate(
        [[0.0], np.cumsum(np.diff(grid) * ((v[:-1] + v[1:]) / 2) / 3.6)]
    )
    return pd.DataFrame(
        {
            "Distance": dist,
            "Speed": v,
            "Throttle": np.interp(grid, t, throttle),
            "Brake": np.interp(grid, t, brake),
        }
    )


def lap_telemetry(session_key: int, driver_number: int) -> tuple[pd.DataFrame, float]:
    lap = fastest_lap(session_key, driver_number)
    start = datetime.fromisoformat(lap["date_start"])
    duration = float(lap["lap_duration"])
    samples = _get(
        "car_data",
        {"session_key": session_key, "driver_number": driver_number},
        raw=[
            f"date>={_naive_utc(start - timedelta(seconds=PAD_S))}",
            f"date<{_naive_utc(start + timedelta(seconds=duration + PAD_S))}",
        ],
    )
    if len(samples) < MIN_SAMPLES:
        raise OpenF1Error(
            f"OpenF1 returned only {len(samples)} telemetry samples for "
            f"driver {driver_number}'s fastest lap."
        )
    return build_lap_frame(samples, start, duration), duration


def coach_report_openf1(session_arg: str, driver: str, reference: str) -> dict:
    """session_arg like '2024 Monaco Q'. Same result shape as the FastF1
    path, plus the official lap times for a sanity check against the
    coach's integrated delta."""
    driver, reference = driver.upper().strip(), reference.upper().strip()
    if driver == reference:
        raise ValueError("Pick two different drivers to compare.")
    year, event, session_type = parse_session_arg(session_arg)
    session_key = resolve_session(year, event, session_type)["session_key"]

    drv_tel, drv_time = lap_telemetry(session_key, resolve_driver(session_key, driver))
    ref_tel, ref_time = lap_telemetry(
        session_key, resolve_driver(session_key, reference)
    )

    result = compare_laps(drv_tel, ref_tel, corners=None)
    result["lap_times_s"] = {driver: drv_time, reference: ref_time}
    return result
