import sqlite3

from sqlalchemy import create_engine, inspect

import src.db as db_mod


def test_init_db_adds_missing_column_to_preexisting_table_without_data_loss(
    tmp_path,
) -> None:
    """Reproduces exactly what happened on a real machine: a 'results'
    table created before event_date existed in the model. init_db() must
    add the column AND must not touch the rows already in the table."""
    db_path = tmp_path / "old_schema.db"

    # Simulate the old schema: results table WITHOUT event_date, with data.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE results (id INTEGER PRIMARY KEY, session_key TEXT, "
        "session_type TEXT, driver TEXT, position INTEGER, "
        "grid_position INTEGER, points FLOAT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO results (session_key, session_type, driver, position, "
        "grid_position, points, status) VALUES ('k','R','VER',1,1,25.0,'Finished')"
    )
    conn.commit()
    conn.close()

    # Point the module at this pre-existing file, as if it were a real run.
    original_engine = db_mod.engine
    db_mod.engine = create_engine(f"sqlite:///{db_path}")
    db_mod.SessionLocal.configure(bind=db_mod.engine)
    try:
        db_mod.init_db()

        inspector = inspect(db_mod.engine)
        cols = {c["name"] for c in inspector.get_columns("results")}
        assert "event_date" in cols

        with db_mod.SessionLocal() as s:
            row = s.query(db_mod.Result).one()
            assert row.driver == "VER"  # pre-existing data survived
            assert row.position == 1
            assert row.event_date is None  # new column, no backfill expected
    finally:
        db_mod.engine = original_engine
        db_mod.SessionLocal.configure(bind=original_engine)


def test_init_db_is_idempotent_on_already_migrated_table(tmp_path) -> None:
    """Running init_db() twice (every app start) must not error just
    because the column already exists from last time."""
    db_path = tmp_path / "fresh.db"
    original_engine = db_mod.engine
    db_mod.engine = create_engine(f"sqlite:///{db_path}")
    db_mod.SessionLocal.configure(bind=db_mod.engine)
    try:
        db_mod.init_db()
        db_mod.init_db()  # must not raise
    finally:
        db_mod.engine = original_engine
        db_mod.SessionLocal.configure(bind=original_engine)
