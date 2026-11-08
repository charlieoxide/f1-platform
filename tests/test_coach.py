import numpy as np
import pandas as pd
import pytest

from src.coach import (
    TelemetryError,
    compare_laps,
    find_braking_zones,
    normalize_telemetry,
    speed_mismatch_rms,
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
    assert "no braking onset" in c.message


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


def _with_time(lap):
    v = np.maximum(lap["Speed"].to_numpy() / 3.6, 1.0)
    d = lap["Distance"].to_numpy()
    t = np.concatenate([[0.0], np.cumsum(np.diff(d) / ((v[:-1] + v[1:]) / 2))])
    return lap.assign(Time=t)


def test_measured_clock_is_used_for_the_total_gap() -> None:
    """The real bug: the total came from integrating speed and disagreed
    with the official lap times. With a Time column, the total must be the
    measured difference."""
    ref = _with_time(make_lap(REF))
    total = float(ref["Time"].iloc[-1])
    drv = ref.assign(Time=ref["Time"] * 1.02)
    result = compare_laps(drv, ref)
    assert result["total_delta_s"] == pytest.approx(0.02 * total, abs=1e-6)
    assert result["delta"]["delta_s"].iloc[-1] == pytest.approx(0.02 * total, abs=0.05)


def test_timedelta_clock_is_accepted() -> None:
    ref = _with_time(make_lap(REF))
    drv = ref.assign(Time=pd.to_timedelta(ref["Time"], unit="s"))
    assert abs(compare_laps(drv, ref)["total_delta_s"]) < 1e-6


def test_unusable_clock_falls_back_to_integrating_speed() -> None:
    ref = _with_time(make_lap(REF))
    broken = ref.assign(Time=ref["Time"].iloc[::-1].to_numpy())  # runs backwards
    assert "Time" not in normalize_telemetry(broken).columns
    assert abs(compare_laps(broken, ref)["total_delta_s"]) < 0.5


def test_continuous_braking_is_not_reported_as_150_m_early() -> None:
    """The real bug: the reference brakes in two zones 40 m apart; the
    driver brakes continuously through both. The old search window made
    the second zone read 'braking 150 m earlier'. It must now say the
    driver was already braking, with no distance claim."""
    ref = make_lap([(600, 700, 100, 730)])
    drv = ref.copy()
    ref["Brake"] = ((ref["Distance"] >= 600) & (ref["Distance"] < 650)) | (
        (ref["Distance"] >= 690) & (ref["Distance"] < 740)
    )
    drv["Brake"] = (drv["Distance"] >= 600) & (drv["Distance"] < 740)

    corners = sorted(compare_laps(drv, ref)["corners"], key=lambda c: c.ref_brake_m)
    assert len(corners) == 2
    second = corners[1]
    assert second.brake_status == "continuous"
    assert second.brake_delta_m is None
    assert "already braking" in second.message
    assert not any("150 m" in c.message for c in corners)


def test_braking_far_outside_tolerance_is_flagged_not_measured() -> None:
    drv = make_lap([(700, 780, 100, 800)])  # 100 m later: outside the 75 m match
    c = compare_laps(drv, make_lap(REF))["corners"][0]
    assert c.brake_status == "none" and c.brake_delta_m is None
    assert "no braking onset" in c.message


def test_differences_below_sample_spacing_are_not_reported() -> None:
    """With ~20 m between samples, a 10 m braking difference is noise: it
    must not appear in the message, while a 30 m one must."""
    from src.coach import _describe

    kwargs = dict(
        label="Zone",
        brake_status="matched",
        apex_delta=0.0,
        throttle_delta=None,
        time_delta=0.05,
        brake_min_m=20.0,
        throttle_min_m=20.0,
    )
    assert "braking" not in _describe(brake_delta=10.0, **kwargs)
    assert "braking 30 m later" in _describe(brake_delta=30.0, **kwargs)


def test_resolution_is_taken_from_the_telemetry_spacing() -> None:
    result = compare_laps(make_lap(REF, step=20.0), make_lap(REF, step=20.0))
    assert result["resolution_m"] == pytest.approx(20.0, abs=0.5)


def test_speed_mismatch_rms_is_zero_for_identical_laps_and_grows_with_a_shift() -> None:
    lap = make_lap(REF)
    shifted = lap.assign(
        Distance=lap["Distance"] + np.where(lap["Distance"] > 100, 15, 0)
    )
    assert speed_mismatch_rms(lap, lap) == pytest.approx(0.0, abs=1e-9)
    assert speed_mismatch_rms(shifted, lap) > 1.0
