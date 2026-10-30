import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError

from src.telemetry import (
    TelemetryUnavailableError,
    circuit_corners,
    coach_report,
    count_drivers_with_telemetry,
    describe_sector_check,
    fastest_lap_telemetry,
    trace_table,
)
from tests.test_coach import REF, make_lap


class FakeCar:
    def __init__(self, df):
        self.df = df

    def add_distance(self):
        return self.df


class FakeLap:
    def __init__(self, df):
        self.df = df

    def get_car_data(self):
        return FakeCar(self.df)


class FakeLaps:
    def __init__(self, by_driver):
        self.by_driver = by_driver

    def pick_drivers(self, driver):
        return FakeLaps({k: v for k, v in self.by_driver.items() if k == driver})

    @property
    def empty(self):
        return not self.by_driver

    def pick_fastest(self):
        return next(iter(self.by_driver.values()))


class FakeSession:
    def __init__(self, by_driver, corners=None):
        self.laps = FakeLaps(by_driver)
        self._corners = corners

    def get_circuit_info(self):
        if self._corners is None:
            raise RuntimeError("no network")
        return type("Info", (), {"corners": self._corners})()


def _session(corners=None):
    return FakeSession(
        {
            "NOR": FakeLap(make_lap([(615, 700, 100, 730)])),
            "VER": FakeLap(make_lap(REF)),
        },
        corners,
    )


def test_fastest_lap_telemetry_unknown_driver_raises() -> None:
    with pytest.raises(ValueError, match="No laps found for driver 'XXX'"):
        fastest_lap_telemetry(_session(), "XXX")


def test_circuit_corners_maps_number_and_letter() -> None:
    df = pd.DataFrame(
        {"Number": [2, 3], "Letter": ["A", ""], "Distance": [310.0, 710.0]}
    )
    assert circuit_corners(_session(df)) == [("2A", 310.0), ("3", 710.0)]


def test_circuit_corners_returns_none_when_unavailable() -> None:
    assert circuit_corners(_session(None)) is None


def test_coach_report_end_to_end_without_network(monkeypatch) -> None:
    corners = pd.DataFrame({"Number": [3], "Letter": [""], "Distance": [710.0]})
    monkeypatch.setattr(
        "src.telemetry.load_session_with_telemetry",
        lambda year, event, st: _session(corners),
    )
    result = coach_report("2024 Monaco Q", "nor", "ver", source="fastf1")
    first = result["corners"][0]
    assert first.label == "Turn 3"
    assert "later than the reference" in first.message


def test_coach_report_rejects_same_driver() -> None:
    with pytest.raises(ValueError, match="two different drivers"):
        coach_report("2024 Monaco Q", "NOR", "nor")


def test_missing_car_data_gives_clear_error_not_bare_keyerror() -> None:
    """The real failure: F1 returned no telemetry, FastF1 raised
    KeyError('4'), and the user saw only \"'4'\"."""

    class NoTelemetryLap:
        def get_car_data(self):
            raise KeyError("4")

    session = FakeSession({"NOR": NoTelemetryLap()})
    with pytest.raises(TelemetryUnavailableError, match="did not return it"):
        fastest_lap_telemetry(session, "NOR")


def test_count_drivers_with_telemetry() -> None:
    class S:
        car_data = {"1": object(), "4": object()}

    class Empty:
        car_data = {}

    class NotLoaded:
        @property
        def car_data(self):
            raise DataNotLoadedError("telemetry not loaded")

    assert count_drivers_with_telemetry(S()) == 2
    assert count_drivers_with_telemetry(Empty()) == 0
    assert count_drivers_with_telemetry(NotLoaded()) == 0


def test_coach_report_defaults_to_openf1(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        "src.openf1.coach_report_openf1",
        lambda s, d, r: seen.update(args=(s, d, r)) or {"corners": []},
    )
    coach_report("2024 Monaco Q", "nor", "ver")
    assert seen["args"] == ("2024 Monaco Q", "NOR", "VER")


def test_coach_report_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown telemetry source"):
        coach_report("2024 Monaco Q", "NOR", "VER", source="nope")


def test_describe_sector_check_reports_worst_error() -> None:
    checks = [
        {"line": "S1/S2", "official_s": -0.010, "coach_s": -0.030},
        {"line": "S2/S3", "official_s": -0.020, "coach_s": 0.080},
    ]
    text, worst = describe_sector_check(checks)
    assert "S1/S2: coach -0.030 s vs official -0.010 s" in text
    assert worst == pytest.approx(0.100)


def test_trace_table_shows_only_the_window_around_the_spot() -> None:
    aligned = pd.DataFrame(
        {"Distance": [float(x) for x in range(0, 3000, 5)], "delta_s": 0.0}
    )
    table = trace_table(aligned, at_m=1230, half_window_m=40, every_m=10)
    lines = table.splitlines()
    assert lines[0].split() == ["Distance", "delta_s"]
    distances = [float(row.split()[0]) for row in lines[1:]]
    assert min(distances) >= 1190 and max(distances) <= 1270
    assert 1230.0 in distances
