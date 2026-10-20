import numpy as np
import pandas as pd
import pytest

from src.coach import (
    TelemetryError,
    compare_laps,
    find_braking_zones,
    normalize_telemetry,
)

VMAX = 300.0


def make_lap(corners, length=2000.0, step=2.0, braking=True, scale=1.0):
    """Synthetic lap. corners = [(brake_start, apex_at, apex_speed,
    throttle_return)]: flat out, brake to the apex speed, then accelerate
    away. scale mimics odometer drift (distance reads long/short)."""
    d = np.arange(0, length + step, step)
    xs, ys = [0.0], [VMAX]
    for b, a, v, _ in corners:
        xs += [b, a, a + 200]
        ys += [VMAX, v, VMAX]
    xs.append(length)
    ys.append(VMAX)
    speed = np.interp(d, xs, ys)

    brake = np.zeros_like(d)
    throttle = np.full_like(d, 100.0)
    for b, a, _, ret in corners:
        if braking:
            brake[(d >= b) & (d < a)] = 1.0
        throttle[(d >= b) & (d < a)] = 0.0
        throttle[(d >= a) & (d < ret)] = 30.0
    return pd.DataFrame(
        {
            "Distance": d * scale,
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake.astype(bool),  # FastF1 gives bools
        }
    )


REF = [(600, 700, 100, 730)]


def test_identical_laps_are_in_line() -> None:
    result = compare_laps(make_lap(REF), make_lap(REF))
    assert abs(result["total_delta_s"]) < 0.01
    assert len(result["corners"]) == 1
    assert "in line with the reference" in result["corners"][0].message


def test_later_braking_is_reported_and_gains_time() -> None:
    drv = make_lap([(615, 700, 100, 730)])
    c = compare_laps(drv, make_lap(REF))["corners"][0]
    assert c.brake_delta_m == pytest.approx(15, abs=5)
    assert "later than the reference" in c.message
    assert c.time_delta_s < 0
    assert "gaining" in c.message


def test_slow_apex_and_late_throttle_are_reported_and_cost_time() -> None:
    drv = make_lap([(600, 700, 90, 750)])
    c = compare_laps(drv, make_lap(REF))["corners"][0]
    assert c.apex_speed_delta_kmh == pytest.approx(-10, abs=1)
    assert c.throttle_delta_m == pytest.approx(20, abs=5)
    assert c.time_delta_s > 0
    assert "10 km/h slower at the apex" in c.message
    assert "costing" in c.message


def test_corners_are_sorted_by_time_lost() -> None:
    ref = [(600, 700, 100, 730), (1400, 1500, 100, 1530)]
    drv = [(600, 700, 98, 735), (1400, 1500, 80, 1580)]  # corner 2 is far worse
    corners = compare_laps(make_lap(drv), make_lap(ref))["corners"]
    assert corners[0].ref_brake_m == pytest.approx(1400, abs=5)
    assert corners[0].time_delta_s > corners[1].time_delta_s


def test_odometer_drift_does_not_misalign_corners() -> None:
    """A 1% distance error must not look like a braking-point error;
    this is why the driver's distance is rescaled to the reference lap."""
    drv = make_lap(REF, scale=1.01)
    result = compare_laps(drv, make_lap(REF))
    assert abs(result["total_delta_s"]) < 0.05
    assert abs(result["corners"][0].brake_delta_m) <= 5


def test_no_braking_where_reference_brakes_is_flagged() -> None:
    drv = make_lap(REF, braking=False)
    c = compare_laps(drv, make_lap(REF))["corners"][0]
    assert c.driver_brake_m is None
    assert "no braking detected" in c.message


def test_corner_labels_use_turn_numbers_only_when_close() -> None:
    ref = make_lap(REF)
    near = compare_laps(make_lap(REF), ref, corners=[("3", 710.0)])["corners"][0]
    far = compare_laps(make_lap(REF), ref, corners=[("9", 1900.0)])["corners"][0]
    assert near.label == "Turn 3"
    assert far.label == "Braking zone 1"


def test_find_braking_zones_merges_blips_and_drops_tiny_zones() -> None:
    d = np.arange(0, 400, 5.0)
    brake = np.zeros_like(d)
    brake[(d >= 100) & (d <= 150)] = 1  # zone
    brake[(d >= 160) & (d <= 200)] = 1  # blip 10 m later -> merged
    brake[(d >= 300) & (d <= 305)] = 1  # 5 m tap -> dropped
    zones = find_braking_zones(d, brake)
    assert zones == [(100.0, 200.0)]


def test_normalize_telemetry_rejects_bad_input() -> None:
    good = make_lap(REF)
    with pytest.raises(TelemetryError, match="missing columns"):
        normalize_telemetry(good.drop(columns=["Brake"]))
    with pytest.raises(TelemetryError, match="usable telemetry samples"):
        normalize_telemetry(good.head(10))
    with pytest.raises(TelemetryError, match="under 500 m"):
        normalize_telemetry(good[good["Distance"] < 400])


def test_normalize_telemetry_converts_bool_brake_and_sorts() -> None:
    shuffled = make_lap(REF).sample(frac=1.0, random_state=1)
    clean = normalize_telemetry(shuffled)
    assert clean["Brake"].dtype == float
    assert clean["Distance"].is_monotonic_increasing
