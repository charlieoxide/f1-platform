"""Module 2 — AI Driver Coach.

Compares a driver's lap against a reference lap and turns the difference
into specific feedback ("braking 15 m later than the reference into
Turn 3, costing 0.08 s"). Deterministic signal processing, not a
learned model: every number in the feedback can be traced back to the
two telemetry traces, which makes it testable and explainable.

Pure numpy/pandas on purpose - no FastF1 import here - so the maths can
be unit-tested against synthetic laps. Loading real telemetry lives in
telemetry.py.

Method:
  1. Validate and resample both laps onto the same distance grid.
  2. Rescale the driver's distance to the reference lap's length, so both
     laps end at the same point (odometer drift would otherwise push
     late corners out of alignment).
  3. Time-at-distance comes from integrating speed; the gap between the
     two curves is the running delta (positive = driver is behind).
  4. Braking zones are found on the reference lap; for each one compare
     brake point, minimum (apex) speed, and where full throttle returns,
     and measure the time gained/lost across that section.
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


class TelemetryError(ValueError):
    """Telemetry is missing channels or too sparse/garbled to compare."""


def normalize_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Validate one lap's telemetry and return a clean, sorted copy with
    Brake as float (FastF1 gives a bool). Bad input is rejected here, at
    the boundary, like every other external data source in this project."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TelemetryError(f"Telemetry is missing columns: {missing}")

    d = df[REQUIRED_COLUMNS].copy()
    d["Brake"] = d["Brake"].astype(float)
    d = d.dropna().sort_values("Distance")
    d = d.drop_duplicates(subset="Distance")

    if len(d) < MIN_SAMPLES:
        raise TelemetryError(
            f"Only {len(d)} usable telemetry samples (need {MIN_SAMPLES}+)."
        )
    if d["Distance"].iloc[-1] - d["Distance"].iloc[0] < 500:
        raise TelemetryError("Lap covers under 500 m - not a full lap.")
    return d.reset_index(drop=True)


def _resample(d: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Distance": grid,
            "Speed": np.interp(grid, d["Distance"], d["Speed"]),
            "Throttle": np.interp(grid, d["Distance"], d["Throttle"]),
            "Brake": np.interp(grid, d["Distance"], d["Brake"]),
        }
    )


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
    brake_delta_m: float | None  # + = driver brakes later than the reference
    apex_speed_delta_kmh: float  # + = driver is faster at the minimum
    throttle_delta_m: float | None  # + = driver is back to full throttle later
    time_delta_s: float  # + = driver lost time across this section
    message: str


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
    brake_delta: float | None,
    apex_delta: float,
    throttle_delta: float | None,
    time_delta: float,
    driver_brake: float | None,
) -> str:
    parts: list[str] = []
    if driver_brake is None:
        parts.append("no braking detected where the reference brakes")
    elif brake_delta is not None and abs(brake_delta) >= 3:
        word = "later" if brake_delta > 0 else "earlier"
        parts.append(f"braking {abs(brake_delta):.0f} m {word} than the reference")
    if abs(apex_delta) >= 2:
        word = "faster" if apex_delta > 0 else "slower"
        parts.append(f"{abs(apex_delta):.0f} km/h {word} at the apex")
    if throttle_delta is not None and abs(throttle_delta) >= 5:
        word = "later" if throttle_delta > 0 else "earlier"
        parts.append(f"full throttle {abs(throttle_delta):.0f} m {word}")

    if not parts:
        return f"{label}: in line with the reference ({time_delta:+.2f} s)."
    verb = "costing" if time_delta > 0 else "gaining"
    return f"{label}: {', '.join(parts)} - {verb} {abs(time_delta):.2f} s."


def compare_laps(
    driver_tel: pd.DataFrame,
    ref_tel: pd.DataFrame,
    corners: list[tuple[str, float]] | None = None,
    step_m: float = 5.0,
) -> dict:
    """Compare a driver lap to a reference lap.

    corners: optional [(name, distance_m)] used only to label zones as
    'Turn N'. Returns {"delta": DataFrame, "corners": [CornerFeedback],
    "total_delta_s": float} with corner feedback sorted by time lost.
    """
    drv = normalize_telemetry(driver_tel)
    ref = normalize_telemetry(ref_tel)

    ref_len = ref["Distance"].iloc[-1] - ref["Distance"].iloc[0]
    drv_len = drv["Distance"].iloc[-1] - drv["Distance"].iloc[0]
    drv = drv.assign(
        Distance=(drv["Distance"] - drv["Distance"].iloc[0]) * ref_len / drv_len
    )
    ref = ref.assign(Distance=ref["Distance"] - ref["Distance"].iloc[0])

    grid = np.arange(0, ref_len, step_m)
    r = _resample(ref, grid)
    d = _resample(drv, grid)
    r_time = _cumulative_time(grid, r["Speed"].to_numpy())
    d_time = _cumulative_time(grid, d["Speed"].to_numpy())
    delta = d_time - r_time

    r_brake = r["Brake"].to_numpy()
    d_brake = d["Brake"].to_numpy()
    zones = find_braking_zones(grid, r_brake)

    feedback: list[CornerFeedback] = []
    for i, (z_start, z_end) in enumerate(zones):
        next_start = zones[i + 1][0] if i + 1 < len(zones) else grid[-1]
        win_end = min(z_end + 150, next_start)

        in_win = (grid >= z_start) & (grid <= win_end)
        ref_apex_i = np.where(in_win)[0][np.argmin(r["Speed"].to_numpy()[in_win])]
        ref_apex_m = float(grid[ref_apex_i])
        ref_apex_speed = float(r["Speed"].to_numpy()[ref_apex_i])
        drv_apex_i = np.where(in_win)[0][np.argmin(d["Speed"].to_numpy()[in_win])]
        drv_apex_m = float(grid[drv_apex_i])
        drv_apex_speed = float(d["Speed"].to_numpy()[drv_apex_i])

        driver_brake = _first_at_or_after(
            grid, d_brake >= BRAKE_THRESHOLD, z_start - 150, z_end
        )
        brake_delta = None if driver_brake is None else driver_brake - z_start

        full = r["Throttle"].to_numpy() >= FULL_THROTTLE
        ref_ret = _first_at_or_after(grid, full, ref_apex_m, win_end + 100)
        full_d = d["Throttle"].to_numpy() >= FULL_THROTTLE
        drv_ret = _first_at_or_after(grid, full_d, drv_apex_m, win_end + 100)
        throttle_delta = (
            None if ref_ret is None or drv_ret is None else drv_ret - ref_ret
        )

        sec_start = max(z_start - 20, 0.0)
        sec_end = ref_ret if ref_ret is not None else ref_apex_m + 100
        time_delta = float(
            np.interp(sec_end, grid, delta) - np.interp(sec_start, grid, delta)
        )
        apex_delta = drv_apex_speed - ref_apex_speed
        label = _label_for(i, ref_apex_m, corners)

        feedback.append(
            CornerFeedback(
                label=label,
                ref_brake_m=float(z_start),
                driver_brake_m=driver_brake,
                brake_delta_m=brake_delta,
                apex_speed_delta_kmh=apex_delta,
                throttle_delta_m=throttle_delta,
                time_delta_s=time_delta,
                message=_describe(
                    label,
                    brake_delta,
                    apex_delta,
                    throttle_delta,
                    time_delta,
                    driver_brake,
                ),
            )
        )

    feedback.sort(key=lambda c: c.time_delta_s, reverse=True)
    return {
        "delta": pd.DataFrame({"Distance": grid, "delta_s": delta}),
        "corners": feedback,
        "total_delta_s": float(delta[-1]),
    }
