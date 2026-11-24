"""Module 2 — AI Driver Coach.

Compares a driver's lap against a reference lap and turns the difference
into specific feedback ("braking 40 m later than the reference, costing
0.08 s"). Deterministic signal processing, not a learned model: every
number can be traced back to the two telemetry traces.

Pure numpy/pandas on purpose - no FastF1/OpenF1 import here - so the maths
can be unit-tested against synthetic laps. Loading real telemetry lives in
telemetry.py / openf1.py.

Method:
  1. Validate and resample both laps onto the same distance grid, and
     rescale the driver's distance to the reference lap's length so both
     laps end at the same point (odometer drift would otherwise push late
     corners out of alignment).
  2. Time-at-distance comes from the measured clock when the telemetry has
     one (a 'Time' column, seconds), otherwise from integrating speed. The
     gap between the two curves is the running delta (+ = driver behind).
  3. Braking zones are found on both laps. Each reference zone is matched
     to the driver's zone that starts within BRAKE_MATCH_TOLERANCE_M of it;
     if there is none, the feedback says so rather than inventing a number.
  4. Differences smaller than the telemetry's own sample spacing are not
     reported - they are below what the data can resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Distance", "Speed", "Throttle", "Brake"]
MIN_SAMPLES = 50
MIN_SPEED_MS = 5 / 3.6  # floor so a near-stationary sample can't blow up dt
FULL_THROTTLE = 98.0
BRAKE_THRESHOLD = 0.5
BRAKE_MATCH_TOLERANCE_M = 75.0


class TelemetryError(ValueError):
    """Telemetry is missing channels or too sparse/garbled to compare."""


def normalize_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Validate one lap's telemetry and return a clean, sorted copy with
    Brake as float (FastF1 gives a bool). An optional 'Time' column (float
    seconds or Timedelta) is kept if it is usable as a clock; otherwise it
    is dropped and time is integrated from speed instead."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TelemetryError(f"Telemetry is missing columns: {missing}")

    cols = REQUIRED_COLUMNS + (["Time"] if "Time" in df.columns else [])
    d = df[cols].copy()
    d["Brake"] = d["Brake"].astype(float)
    if "Time" in d.columns:
        if pd.api.types.is_timedelta64_dtype(d["Time"]):
            d["Time"] = d["Time"].dt.total_seconds()
        d["Time"] = d["Time"].astype(float)
    d = d.dropna().sort_values("Distance").drop_duplicates(subset="Distance")

    if len(d) < MIN_SAMPLES:
        raise TelemetryError(
            f"Only {len(d)} usable telemetry samples (need {MIN_SAMPLES}+)."
        )
    if d["Distance"].iloc[-1] - d["Distance"].iloc[0] < 500:
        raise TelemetryError("Lap covers under 500 m - not a full lap.")
    if "Time" in d.columns and not d["Time"].is_monotonic_increasing:
        d = d.drop(columns="Time")  # unusable clock: integrate speed instead
    return d.reset_index(drop=True)


def _resample(d: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    out = {"Distance": grid}
    for col in ["Speed", "Throttle", "Brake"] + (["Time"] if "Time" in d else []):
        out[col] = np.interp(grid, d["Distance"], d[col])
    return pd.DataFrame(out)


def _cumulative_time(grid: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
    v = np.maximum(speed_kmh / 3.6, MIN_SPEED_MS)
    dt = np.diff(grid) / ((v[:-1] + v[1:]) / 2)
    return np.concatenate([[0.0], np.cumsum(dt)])


def find_braking_zones(
    distance: np.ndarray,
    brake: np.ndarray,
    min_len_m: float = 15.0,
    merge_gap_m: float = 30.0,
) -> list[tuple[float, float]]:
    """Contiguous stretches where the brake is on. Zones separated by a
    short gap (brake blips) are merged; very short zones are dropped."""
    on = brake >= BRAKE_THRESHOLD
    zones: list[list[float]] = []
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = distance[i]
        elif not flag and start is not None:
            zones.append([start, distance[i - 1]])
            start = None
    if start is not None:
        zones.append([start, distance[-1]])

    merged: list[list[float]] = []
    for z in zones:
        if merged and z[0] - merged[-1][1] <= merge_gap_m:
            merged[-1][1] = z[1]
        else:
            merged.append(z)
    return [(s, e) for s, e in merged if e - s >= min_len_m]


@dataclass
class CornerFeedback:
    label: str
    ref_brake_m: float
    driver_brake_m: float | None
    brake_status: str  # "matched" | "continuous" | "none"
    brake_delta_m: float | None  # + = driver brakes later than the reference
    apex_speed_delta_kmh: float  # + = driver is faster at the minimum
    throttle_delta_m: float | None  # + = driver is back to full throttle later
    time_delta_s: float  # + = driver lost time across this section
    message: str


def _match_zone_starts(
    ref_zones: list[tuple[float, float]],
    drv_zones: list[tuple[float, float]],
    tolerance: float,
) -> dict[int, float]:
    """Pair each driver braking zone with the reference zone whose start it is
    NEAREST to (within tolerance). A driver zone is never given to a farther
    reference zone, so two close zones can't both claim the same braking
    event; a zone nobody claims is reported as unmatched, not guessed at."""
    best: dict[int, float] = {}
    for d_start, _ in drv_zones:
        if not ref_zones:
            break
        i = min(range(len(ref_zones)), key=lambda k: abs(ref_zones[k][0] - d_start))
        gap = abs(ref_zones[i][0] - d_start)
        if gap <= tolerance and (i not in best or gap < abs(best[i] - ref_zones[i][0])):
            best[i] = d_start
    return best


def _label_for(
    index: int, ref_apex_m: float, corners: list[tuple[str, float]] | None
) -> str:
    if corners:
        name, dist = min(corners, key=lambda c: abs(c[1] - ref_apex_m))
        if abs(dist - ref_apex_m) <= 200:  # corner map is approximate
            return f"Turn {name}"
    return f"Braking zone {index + 1}"


def _first_at_or_after(
    distance: np.ndarray, mask: np.ndarray, from_m: float, to_m: float
) -> float | None:
    sel = np.where(mask & (distance >= from_m) & (distance <= to_m))[0]
    return float(distance[sel[0]]) if len(sel) else None


def _describe(
    label: str,
    brake_status: str,
    brake_delta: float | None,
    apex_delta: float,
    throttle_delta: float | None,
    time_delta: float,
    brake_min_m: float,
    throttle_min_m: float,
) -> str:
    parts: list[str] = []
    if brake_status == "none":
        parts.append(
            f"no braking onset within {BRAKE_MATCH_TOLERANCE_M:.0f} m of "
            "where the reference brakes"
        )
    elif brake_status == "continuous":
        parts.append("already braking from the previous zone")
    elif brake_delta is not None and abs(brake_delta) >= brake_min_m:
        word = "later" if brake_delta > 0 else "earlier"
        parts.append(f"braking {abs(brake_delta):.0f} m {word} than the reference")
    if abs(apex_delta) >= 2:
        word = "faster" if apex_delta > 0 else "slower"
        parts.append(f"{abs(apex_delta):.0f} km/h {word} at the apex")
    if throttle_delta is not None and abs(throttle_delta) >= throttle_min_m:
        word = "later" if throttle_delta > 0 else "earlier"
        parts.append(f"full throttle {abs(throttle_delta):.0f} m {word}")

    verb = "costing" if time_delta > 0 else "gaining"
    if not parts and abs(time_delta) >= 0.05:
        return (
            f"{label}: {verb} {abs(time_delta):.2f} s with no braking, apex or "
            "throttle difference above the data's resolution (timing alignment "
            "is the least certain part here)."
        )
    if not parts:
        return f"{label}: in line with the reference ({time_delta:+.2f} s)."
    return f"{label}: {', '.join(parts)} - {verb} {abs(time_delta):.2f} s."


def speed_mismatch_rms(
    driver_tel: pd.DataFrame, ref_tel: pd.DataFrame, step_m: float = 5.0
) -> float:
    """RMS speed difference (km/h) between two laps on a shared distance
    grid. Used to find the clock offset between two telemetry streams: the
    right offset is the one that makes the traces line up best."""
    drv = normalize_telemetry(driver_tel)
    ref = normalize_telemetry(ref_tel)
    ref_len = ref["Distance"].iloc[-1] - ref["Distance"].iloc[0]
    drv_len = drv["Distance"].iloc[-1] - drv["Distance"].iloc[0]
    grid = np.arange(0, ref_len, step_m)
    d_dist = (drv["Distance"] - drv["Distance"].iloc[0]) * ref_len / drv_len
    r_dist = ref["Distance"] - ref["Distance"].iloc[0]
    diff = np.interp(grid, d_dist, drv["Speed"]) - np.interp(grid, r_dist, ref["Speed"])
    return float(np.sqrt(np.mean(diff**2)))


def compare_laps(
    driver_tel: pd.DataFrame,
    ref_tel: pd.DataFrame,
    corners: list[tuple[str, float]] | None = None,
    step_m: float = 5.0,
) -> dict:
    """Compare a driver lap to a reference lap.

    corners: optional [(name, distance_m)] used only to label zones as
    'Turn N'. Returns a dict with:
      delta          DataFrame(Distance, delta_s, ref_time_s)
      aligned        DataFrame of both laps' channels on the shared grid
      corners        [CornerFeedback], sorted by time lost
      total_delta_s  lap-time gap (measured, if the telemetry has a clock)
      resolution_m   telemetry sample spacing; smaller differences aren't
                     reported
    """
    drv = normalize_telemetry(driver_tel)
    ref = normalize_telemetry(ref_tel)
    has_time = "Time" in drv.columns and "Time" in ref.columns

    spacing = max(
        float(np.median(np.diff(drv["Distance"]))),
        float(np.median(np.diff(ref["Distance"]))),
    )
    brake_min_m = max(3.0, spacing)
    throttle_min_m = max(5.0, spacing)

    ref_len = ref["Distance"].iloc[-1] - ref["Distance"].iloc[0]
    drv_len = drv["Distance"].iloc[-1] - drv["Distance"].iloc[0]
    drv = drv.assign(
        Distance=(drv["Distance"] - drv["Distance"].iloc[0]) * ref_len / drv_len
    )
    ref = ref.assign(Distance=ref["Distance"] - ref["Distance"].iloc[0])
    if has_time:
        drv = drv.assign(Time=drv["Time"] - drv["Time"].iloc[0])
        ref = ref.assign(Time=ref["Time"] - ref["Time"].iloc[0])

    grid = np.arange(0, ref_len, step_m)
    r = _resample(ref, grid)
    d = _resample(drv, grid)
    if has_time:
        r_time, d_time = r["Time"].to_numpy(), d["Time"].to_numpy()
    else:
        r_time = _cumulative_time(grid, r["Speed"].to_numpy())
        d_time = _cumulative_time(grid, d["Speed"].to_numpy())
    delta = d_time - r_time
    total_delta = (
        float(drv["Time"].iloc[-1] - ref["Time"].iloc[-1])
        if has_time
        else float(delta[-1])
    )

    r_speed, d_speed = r["Speed"].to_numpy(), d["Speed"].to_numpy()
    r_brake, d_brake = r["Brake"].to_numpy(), d["Brake"].to_numpy()
    r_zones = find_braking_zones(grid, r_brake)
    d_zones = find_braking_zones(grid, d_brake)

    matched = _match_zone_starts(r_zones, d_zones, BRAKE_MATCH_TOLERANCE_M)

    feedback: list[CornerFeedback] = []
    for i, (z_start, z_end) in enumerate(r_zones):
        next_start = r_zones[i + 1][0] if i + 1 < len(r_zones) else grid[-1]
        win_end = min(z_end + 150, next_start)
        in_win = (grid >= z_start) & (grid <= win_end)
        idx = np.where(in_win)[0]

        ref_apex_i = idx[np.argmin(r_speed[idx])]
        drv_apex_i = idx[np.argmin(d_speed[idx])]
        ref_apex_m = float(grid[ref_apex_i])
        drv_apex_m = float(grid[drv_apex_i])
        apex_delta = float(d_speed[drv_apex_i] - r_speed[ref_apex_i])

        driver_brake = matched.get(i)
        if driver_brake is not None:
            brake_status, brake_delta = "matched", float(driver_brake - z_start)
        else:
            driver_brake, brake_delta = None, None
            at_start = min(int(np.searchsorted(grid, z_start)), len(grid) - 1)
            brake_status = (
                "continuous" if d_brake[at_start] >= BRAKE_THRESHOLD else "none"
            )

        ref_ret = _first_at_or_after(
            grid, r["Throttle"].to_numpy() >= FULL_THROTTLE, ref_apex_m, win_end + 100
        )
        drv_ret = _first_at_or_after(
            grid, d["Throttle"].to_numpy() >= FULL_THROTTLE, drv_apex_m, win_end + 100
        )
        throttle_delta = (
            None if ref_ret is None or drv_ret is None else drv_ret - ref_ret
        )

        sec_start = max(z_start - 20, 0.0)
        sec_end = ref_ret if ref_ret is not None else ref_apex_m + 100
        time_delta = float(
            np.interp(sec_end, grid, delta) - np.interp(sec_start, grid, delta)
        )
        label = _label_for(i, ref_apex_m, corners)

        feedback.append(
            CornerFeedback(
                label=label,
                ref_brake_m=float(z_start),
                driver_brake_m=driver_brake,
                brake_status=brake_status,
                brake_delta_m=brake_delta,
                apex_speed_delta_kmh=apex_delta,
                throttle_delta_m=throttle_delta,
                time_delta_s=time_delta,
                message=_describe(
                    label,
                    brake_status,
                    brake_delta,
                    apex_delta,
                    throttle_delta,
                    time_delta,
                    brake_min_m,
                    throttle_min_m,
                ),
            )
        )

    feedback.sort(key=lambda c: c.time_delta_s, reverse=True)
    aligned = pd.DataFrame(
        {
            "Distance": grid,
            "ref_speed": r_speed,
            "drv_speed": d_speed,
            "ref_brake": r_brake,
            "drv_brake": d_brake,
            "ref_time_s": r_time,
            "drv_time_s": d_time,
            "delta_s": delta,
        }
    )
    return {
        "delta": aligned[["Distance", "delta_s", "ref_time_s"]].copy(),
        "aligned": aligned,
        "corners": feedback,
        "total_delta_s": total_delta,
        "resolution_m": spacing,
    }
