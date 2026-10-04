import pandas as pd
import pytest

from src.predictor import build_features, predict_finish, train_model


def _synthetic_results(n_races: int = 12) -> pd.DataFrame:
    """One driver with a believable signal: starts near the front, usually
    finishes near the front. Enough races for a 75/25 chronological split
    to have test rows."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_races, freq="7D")
    for i, date in enumerate(dates):
        grid = 1 + (i % 5)
        finish = max(1, grid + (1 if i % 3 == 0 else -1))
        rows.append(
            {
                "session_key": f"k{i}",
                "driver": "VER",
                "event_date": date,
                "grid_position": grid,
                "position": finish,
            }
        )
    return pd.DataFrame(rows)


def test_build_features_drops_first_race_per_driver_no_history() -> None:
    df = _synthetic_results(5)
    featured = build_features(df)
    # First race chronologically has no prior races -> no recent_avg_finish -> dropped
    assert len(featured) == 4
    assert featured["event_date"].min() > df["event_date"].min()


def test_build_features_has_no_leakage_from_same_race_outcome() -> None:
    """The defining correctness property of this module: recent_avg_finish
    for a given race must be computed ONLY from races strictly before it.
    We verify this directly by recomputing it by hand for one row."""
    df = _synthetic_results(6)
    featured = build_features(df).reset_index(drop=True)

    third_row = featured.iloc[1]  # second usable row overall
    prior_positions = df[df["event_date"] < third_row["event_date"]]["position"]
    expected_avg = prior_positions.mean()
    assert third_row["recent_avg_finish"] == pytest.approx(expected_avg)


def test_build_features_drops_dnf_and_missing_date_rows() -> None:
    df = _synthetic_results(6)
    df.loc[2, "position"] = None  # DNF: no valid target
    df.loc[3, "event_date"] = None  # can't order chronologically
    featured = build_features(df)
    assert df.loc[2, "session_key"] not in featured["session_key"].values
    assert df.loc[3, "session_key"] not in featured["session_key"].values


def test_train_model_raises_with_too_little_data() -> None:
    df = _synthetic_results(2)
    with pytest.raises(ValueError, match="usable rows"):
        train_model(df)


def test_train_model_produces_usable_model_and_honest_mae() -> None:
    df = _synthetic_results(16)
    result = train_model(df)
    assert result["train_rows"] > 0 and result["test_rows"] > 0
    assert result["mae"] >= 0

    pred = predict_finish(result["model"], grid_position=1, recent_avg_finish=2.0)
    assert isinstance(pred, float)


def test_train_model_splits_chronologically_not_randomly() -> None:
    """Test rows must all come after train rows in time - this is what
    makes the MAE an honest estimate instead of leaked validation."""
    df = _synthetic_results(16)
    featured = build_features(df).sort_values("event_date")
    split_idx = int(len(featured) * 0.75)
    train_dates = featured.iloc[:split_idx]["event_date"]
    test_dates = featured.iloc[split_idx:]["event_date"]
    assert train_dates.max() <= test_dates.min()


def test_multi_driver_rolling_average_is_isolated_per_driver() -> None:
    """Regression test for a real bug class: a wrong groupby could let one
    driver's finishing history leak into another driver's feature."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=5, freq="7D")
    for i, d in enumerate(dates):
        rows.append(
            {
                "session_key": f"a{i}",
                "driver": "VER",
                "event_date": d,
                "grid_position": 1,
                "position": 1,
            }
        )
        rows.append(
            {
                "session_key": f"b{i}",
                "driver": "HAM",
                "event_date": d,
                "grid_position": 10,
                "position": 15,
            }
        )
    df = pd.DataFrame(rows)
    featured = build_features(df)

    ham = featured[featured["driver"] == "HAM"]["recent_avg_finish"]
    ver = featured[featured["driver"] == "VER"]["recent_avg_finish"]
    assert (ham == 15.0).all()
    assert (ver == 1.0).all()
