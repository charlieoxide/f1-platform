import pandas as pd
import pytest

from src.ingest import laps_to_records, parse_session_arg, results_to_records


def test_parse_session_arg() -> None:
    assert parse_session_arg("2024 Monaco R") == (2024, "Monaco", "R")
    assert parse_session_arg("2024 Abu Dhabi Q") == (2024, "Abu Dhabi", "Q")


def test_laps_to_records_handles_nan_and_bad_rows() -> None:
    df = pd.DataFrame(
        [
            {  # normal lap
                "Driver": "VER",
                "LapNumber": 1.0,
                "LapTime": pd.Timedelta(seconds=80.5),
                "Compound": "MEDIUM",
                "Stint": 1.0,
                "IsPersonalBest": True,
            },
            {  # deleted/missing lap time, NaN tyre + flag
                "Driver": "HAM",
                "LapNumber": 2.0,
                "LapTime": pd.NaT,
                "Compound": float("nan"),
                "Stint": float("nan"),
                "IsPersonalBest": float("nan"),
            },
            {  # garbage row: no lap number
                "Driver": "LEC",
                "LapNumber": float("nan"),
                "LapTime": pd.NaT,
                "Compound": None,
                "Stint": 1.0,
                "IsPersonalBest": False,
            },
        ]
    )
    records, skipped = laps_to_records("2024_Monaco_R", df)
    assert len(records) == 2 and skipped == 1
    assert records[0].lap_time_seconds == 80.5 and records[0].is_personal_best
    assert records[1].lap_time_seconds is None and records[1].compound is None
    assert records[1].is_personal_best is False


def test_results_to_records_handles_dnf_and_normal_finish() -> None:
    df = pd.DataFrame(
        [
            {  # normal finish
                "Abbreviation": "VER",
                "Position": 1.0,
                "GridPosition": 1.0,
                "Points": 25.0,
                "Status": "Finished",
            },
            {  # DNF: no finishing position, but grid position still exists
                "Abbreviation": "PER",
                "Position": float("nan"),
                "GridPosition": 5.0,
                "Points": 0.0,
                "Status": "Accident",
            },
            {  # garbage: no driver code at all
                "Abbreviation": "",
                "Position": 3.0,
                "GridPosition": 3.0,
                "Points": 15.0,
                "Status": "Finished",
            },
        ]
    )
    records, skipped = results_to_records("2024_Monaco_R", "R", df)
    assert len(records) == 2 and skipped == 1
    assert records[0].driver == "VER" and records[0].position == 1
    assert records[1].driver == "PER" and records[1].position is None
    assert records[1].grid_position == 5


class _FakeEvent(dict):
    """Mimics session.event's dict-like interface closely enough for
    _check_event_match without needing a real FastF1 session."""


class _FakeSession:
    def __init__(self, event_name: str, location: str) -> None:
        self.event = _FakeEvent(EventName=event_name, Location=location)


def test_check_event_match_accepts_reasonable_match() -> None:
    from src.ingest import _check_event_match

    _check_event_match("Monaco", _FakeSession("Monaco Grand Prix", "Monte Carlo"))
    _check_event_match(
        "Saudi Arabia", _FakeSession("Saudi Arabian Grand Prix", "Jeddah")
    )


def test_check_event_match_rejects_wrong_event_from_fuzzy_match_failure() -> None:
    """Reproduces exactly what happened on a real machine: requesting
    'Great Britain' and FastF1 silently handing back Austria instead."""
    from src.ingest import EventMismatchError, _check_event_match

    with pytest.raises(EventMismatchError):
        _check_event_match(
            "Great Britain", _FakeSession("Austrian Grand Prix", "Spielberg")
        )
