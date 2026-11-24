"""Accurate lap simulator for tests. Drives a synthetic lap with true
kinematics (5 ms steps), then samples it the way OpenF1 does (~3.8 Hz,
integer km/h), optionally with the telemetry clock running late by `lag`.

The earlier test simulator stepped the car at the sample rate, which put
its own ~7 m error into braking zones - the same size as the effect being
tested. Keep the true motion fine-grained and only the *output* coarse."""

from datetime import datetime, timedelta, timezone

import numpy as np

T0 = datetime(2024, 5, 25, 14, 0, 0, tzinfo=timezone.utc)


def drive(lap_df, fine=0.005):
    d = lap_df["Distance"].to_numpy()
    v = lap_df["Speed"].to_numpy()
    t_list, s_list, t, s = [0.0], [0.0], 0.0, 0.0
    while s < d[-1]:
        s += np.interp(s, d, v) / 3.6 * fine
        t += fine
        t_list.append(t)
        s_list.append(s)
    return np.array(t_list), np.array(s_list)


def sample(lap_df, lag=0.0, step=0.264, phase=0.0):
    """-> (samples, lap_duration, t_fine, s_fine). Samples cover 0.6 s either
    side of the lap, like fetch_lap_samples' padding."""
    t_fine, s_fine = drive(lap_df)
    d = lap_df["Distance"].to_numpy()
    rows = []
    for t in np.arange(-0.6 + phase, t_fine[-1] + 0.6, step):
        s = float(np.interp(t, t_fine, s_fine))
        rows.append(
            {
                "date": (T0 + timedelta(seconds=float(t) + lag)).isoformat(),
                "speed": round(float(np.interp(s, d, lap_df["Speed"]))),
                "throttle": round(float(np.interp(s, d, lap_df["Throttle"]))),
                "brake": (
                    100 if np.interp(s, d, lap_df["Brake"].astype(float)) >= 0.5 else 0
                ),
            }
        )
    return rows, float(t_fine[-1]), t_fine, s_fine


def lap_record(t_end, t_fine, s_fine, lines=(700, 1500)):
    """Official-style lap record; sector lines sit at fixed distances."""
    a, b = (float(np.interp(x, s_fine, t_fine)) for x in lines)
    return {
        "date_start": T0.isoformat(),
        "lap_duration": t_end,
        "duration_sector_1": a,
        "duration_sector_2": b - a,
        "duration_sector_3": t_end - b,
    }
