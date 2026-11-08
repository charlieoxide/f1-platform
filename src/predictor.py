"""Module 4 — Outcome Predictor.

Trains a gradient-boosting model (scikit-learn, CPU, no GPU needed) to
predict finishing position from grid position and recent form. This is
explicitly framed as a system to evaluate, not a forecast guaranteed to
be right - real F1 data includes DNFs, safety cars, and grid penalties
that make this a genuinely hard prediction problem.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.config import settings
from src.dbutil import readonly_uri

FEATURE_COLUMNS = ["grid_position", "recent_avg_finish"]
TARGET_COLUMN = "position"


def load_race_results(db_path: str | None = None) -> pd.DataFrame:
    """Race (not qualifying) results only, with event_date for ordering.
    Read-only connection — this module never writes to the DB."""
    conn = sqlite3.connect(readonly_uri(db_path or settings.db_path), uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT session_key, driver, event_date, grid_position, position "
            "FROM results WHERE session_type = 'R'",
            conn,
            parse_dates=["event_date"],
        )
    finally:
        conn.close()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recent_avg_finish: each driver's mean finishing position over
    their previous races only (shift(1) before the rolling mean), so no
    race's features include its own outcome - that would be leakage, not
    a real predictive feature.

    Drops rows with no event_date (can't order chronologically) and rows
    with no finishing position (DNF/DSQ - no valid target to train on).
    Each driver's first race has no prior history, so it's dropped too.
    """
    d = df.dropna(subset=["event_date", "grid_position", "position"]).copy()
    d = d.sort_values(["driver", "event_date"])
    d["recent_avg_finish"] = d.groupby("driver")["position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return d.dropna(subset=["recent_avg_finish"])


def train_model(df: pd.DataFrame) -> dict:
    """Chronological holdout (not random split) - the last ~25% of races
    by date become the test set, since shuffling would let the model
    'see the future' relative to a race it's tested on. Returns the
    fitted model plus honest out-of-sample MAE."""
    featured = build_features(df)
    if len(featured) < 10:
        raise ValueError(
            f"Only {len(featured)} usable rows after feature engineering - "
            "need more ingested races (see scripts/ingest_season.sh)."
        )

    featured = featured.sort_values("event_date")
    split_idx = int(len(featured) * 0.75)
    train, test = featured.iloc[:split_idx], featured.iloc[split_idx:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    preds = model.predict(test[FEATURE_COLUMNS])
    mae = mean_absolute_error(test[TARGET_COLUMN], preds)

    return {
        "model": model,
        "mae": mae,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def predict_finish(
    model: GradientBoostingRegressor, grid_position: int, recent_avg_finish: float
) -> float:
    df = pd.DataFrame(
        [{"grid_position": grid_position, "recent_avg_finish": recent_avg_finish}]
    )
    return float(model.predict(df)[0])


def main() -> None:
    """CLI entry point: python -m src.predictor"""
    df = load_race_results()
    result = train_model(df)
    print(
        f"Trained on {result['train_rows']} rows, tested on "
        f"{result['test_rows']} rows (chronological holdout)."
    )
    print(f"Mean Absolute Error: {result['mae']:.2f} finishing positions")
    print(
        "This means predictions are off by about "
        f"{result['mae']:.1f} positions on average, on races the model "
        "never trained on."
    )


if __name__ == "__main__":
    main()
