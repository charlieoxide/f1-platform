"""Pure data-shaping functions for the dashboard. Kept separate from
dashboard.py so this logic can be unit-tested without a running Streamlit
app or a browser."""

from __future__ import annotations

import pandas as pd

# No legitimate F1 lap - green flag, safety car, or VSC - takes this long.
# Red-flag stoppages get folded into FastF1's LapTime for the lap in which
# they occur (the session clock doesn't stop), which can inflate a single
# lap to tens of minutes. We keep that row in the raw data (it's real) but
# exclude it from charts, where it would dwarf every other value.
MAX_PLAUSIBLE_LAP_SECONDS = 200


def _drop_timing_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["lap_time_s"].isna() | (df["lap_time_s"] <= MAX_PLAUSIBLE_LAP_SECONDS)]


def pace_comparison(df: pd.DataFrame, driver_a: str, driver_b: str) -> pd.DataFrame:
    """Lap time per lap number for two drivers, side by side, for overlay
    charting. Only laps with a valid (non-null), plausible time are
    included."""
    df = _drop_timing_anomalies(df)
    subset = df[df["driver"].isin([driver_a, driver_b]) & df["lap_time_s"].notna()]
    pivot = subset.pivot_table(index="lap", columns="driver", values="lap_time_s")
    return pivot.sort_index()


def tyre_stints(df: pd.DataFrame, driver: str) -> pd.DataFrame:
    """One column per stint (labelled with its tyre compound), indexed by
    lap-within-stint, so degradation across a stint is a single line."""
    df = _drop_timing_anomalies(df)
    d = df[(df["driver"] == driver) & df["lap_time_s"].notna()].copy()
    if d.empty or d["stint"].isna().all():
        return pd.DataFrame()

    d["stint"] = d["stint"].fillna(0).astype(int)
    d = d.sort_values("lap")
    d["lap_in_stint"] = d.groupby("stint").cumcount() + 1

    compound_by_stint = d.groupby("stint")["compound"].first()
    pivot = d.pivot_table(index="lap_in_stint", columns="stint", values="lap_time_s")
    pivot.columns = [
        f"Stint {s} ({compound_by_stint.get(s, '?')})" for s in pivot.columns
    ]
    return pivot.sort_index()
