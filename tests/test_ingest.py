import pandas as pd

from src.ingest import laps_to_records, parse_session_arg


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
