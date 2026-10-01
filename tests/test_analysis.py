import pandas as pd

from src.analysis import pace_comparison, tyre_stints


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver": "VER",
                "lap": 1,
                "lap_time_s": 80.0,
                "compound": "MEDIUM",
                "stint": 1,
            },
            {
                "driver": "VER",
                "lap": 2,
                "lap_time_s": 79.5,
                "compound": "MEDIUM",
                "stint": 1,
            },
            {
                "driver": "VER",
                "lap": 3,
                "lap_time_s": 78.0,
                "compound": "HARD",
                "stint": 2,
            },
            {
                "driver": "HAM",
                "lap": 1,
                "lap_time_s": 81.0,
                "compound": "MEDIUM",
                "stint": 1,
            },
            {
                "driver": "HAM",
                "lap": 2,
                "lap_time_s": None,
                "compound": "MEDIUM",
                "stint": 1,
            },  # crash/DNF lap
            {
                "driver": "PER",
                "lap": 1,
                "lap_time_s": 82.0,
                "compound": "SOFT",
                "stint": 1,
            },
        ]
    )


def test_pace_comparison_only_includes_requested_drivers_and_valid_laps() -> None:
    pivot = pace_comparison(_sample_df(), "VER", "HAM")
    assert list(pivot.columns) == ["HAM", "VER"]  # alphabetical from pivot_table
    assert pivot.loc[1, "VER"] == 80.0
    assert pd.isna(
        pivot.loc[2, "HAM"]
    )  # null lap time stays out of the surviving row, but index exists
    assert "PER" not in pivot.columns


def test_tyre_stints_splits_by_stint_and_labels_compound() -> None:
    pivot = tyre_stints(_sample_df(), "VER")
    cols = list(pivot.columns)
    assert cols == ["Stint 1 (MEDIUM)", "Stint 2 (HARD)"]
    assert pivot["Stint 1 (MEDIUM)"].dropna().tolist() == [80.0, 79.5]
    assert pivot["Stint 2 (HARD)"].dropna().tolist() == [78.0]


def test_tyre_stints_empty_when_no_stint_data() -> None:
    df = _sample_df().assign(stint=None)
    assert tyre_stints(df, "VER").empty


def test_red_flag_lap_excluded_from_charts_but_not_mutated_elsewhere() -> None:
    df = _sample_df()
    df.loc[len(df)] = {
        "driver": "VER",
        "lap": 4,
        "lap_time_s": 2494.8,
        "compound": "HARD",
        "stint": 2,
    }
    pace = pace_comparison(df, "VER", "HAM")
    assert 4 not in pace.index  # the red-flag-inflated lap never reaches the chart

    stints = tyre_stints(df, "VER")
    assert (
        stints["Stint 2 (HARD)"].max() < 200
    )  # outlier excluded, real stint data kept
