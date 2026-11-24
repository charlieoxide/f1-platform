import sys
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

import src.db as db_mod
from src.ingest_openf1 import (
    build_lap_records,
    build_result_records,
    ingest_session_openf1,
    resolve_grid,
)
from src.openf1 import OpenF1Error

RACE_KEY, QUALI_KEY, MEETING = 9523, 9519, 1236
RACE = {
    "session_key": RACE_KEY,
    "meeting_key": MEETING,
    "date_start": "2024-05-26T13:00:00+00:00",
}

DRIVERS = [
    {"driver_number": n, "name_acronym": a}
    for n, a in [
        (1, "VER"),
        (4, "NOR"),
        (16, "LEC"),
        (11, "PER"),
        (22, "TSU"),
        (77, "BOT"),
    ]
]
LAPS = [
    {"driver_number": 1, "lap_number": 1, "lap_duration": None},  # red-flag lap
    {"driver_number": 1, "lap_number": 2, "lap_duration": 80.5},
    {"driver_number": 1, "lap_number": 3, "lap_duration": 79.9},
    {"driver_number": 1, "lap_number": 4, "lap_duration": 81.0},
    {"driver_number": 4, "lap_number": 1, "lap_duration": 85.0},
    {"driver_number": 99, "lap_number": 1, "lap_duration": 90.0},  # not in drivers
]
STINTS = [
    # lap_end 0 is what OpenF1 really returns for some stints: it must be ignored
    {
        "driver_number": 1,
        "stint_number": 1,
        "lap_start": 1,
        "lap_end": 0,
        "compound": "MEDIUM",
    },
    {
        "driver_number": 1,
        "stint_number": 2,
        "lap_start": 3,
        "lap_end": 4,
        "compound": "HARD",
    },
    {
        "driver_number": 4,
        "stint_number": 1,
        "lap_start": 1,
        "lap_end": 0,
        "compound": "soft",
    },
]
RESULTS = [
    {"driver_number": 1, "position": 1, "number_of_laps": 78, "points": 25.0},
    {"driver_number": 4, "position": 2, "number_of_laps": 78, "points": 18.0},
    {"driver_number": 16, "position": 3, "number_of_laps": 77, "points": 15.0},
    {
        "driver_number": 11,
        "position": 4,
        "number_of_laps": 70,
        "points": 0.0,
        "dnf": True,
    },
    {
        "driver_number": 22,
        "position": 5,
        "number_of_laps": 0,
        "points": 0.0,
        "dns": True,
    },
    {
        "driver_number": 77,
        "position": 6,
        "number_of_laps": 78,
        "points": 0.0,
        "dsq": True,
    },
]
GRID = [
    {"driver_number": 1, "position": 2},
    {"driver_number": 4, "position": 1},
    {"driver_number": 16, "position": 3},
]
ACRONYMS = {r["driver_number"]: r["name_acronym"] for r in DRIVERS}


def router(table, quali_sessions=None):
    """Fake OpenF1: table maps (endpoint, session_key) -> rows."""

    def fake_get(endpoint, eq=None, raw=()):
        eq = eq or {}
        if endpoint == "sessions":
            return quali_sessions if quali_sessions is not None else []
        return table.get((endpoint, eq.get("session_key")), [])

    return fake_get


def full_table():
    return {
        ("drivers", RACE_KEY): DRIVERS,
        ("laps", RACE_KEY): LAPS,
        ("stints", RACE_KEY): STINTS,
        ("session_result", RACE_KEY): RESULTS,
        ("starting_grid", RACE_KEY): GRID,
    }


# ----------------------------------------------------------------- mapping


def test_laps_get_stint_compound_and_personal_best_and_bad_rows_are_skipped() -> None:
    records, skipped = build_lap_records("k", LAPS, ACRONYMS, STINTS)
    by = {(r.driver, r.lap_number): r for r in records}

    assert skipped == 1  # driver 99 is not in the session's driver list
    assert by[("VER", 1)].lap_time_seconds is None  # null duration is kept as null
    assert by[("VER", 1)].stint == 1 and by[("VER", 1)].compound == "MEDIUM"
    assert by[("VER", 2)].stint == 1 and by[("VER", 2)].compound == "MEDIUM"
    # laps 3-4 belong to stint 2 even though stint 1 says lap_end=0
    assert by[("VER", 3)].stint == 2 and by[("VER", 3)].compound == "HARD"
    assert by[("VER", 4)].stint == 2
    assert by[("NOR", 1)].compound == "SOFT"  # normalised to upper case
    # personal best = a new fastest lap for that driver so far
    assert [by[("VER", n)].is_personal_best for n in (1, 2, 3, 4)] == [
        False,
        True,
        True,
        False,
    ]


def test_result_statuses_points_and_grid() -> None:
    records, skipped = build_result_records(
        "k", "R", RESULTS, ACRONYMS, {1: 2, 4: 1, 16: 3}, datetime(2024, 5, 26, 13)
    )
    by = {r.driver: r for r in records}
    assert skipped == 0
    assert by["VER"].status == "Finished" and by["VER"].points == 25.0
    assert by["VER"].grid_position == 2 and by["NOR"].grid_position == 1
    assert by["LEC"].status == "+1 Lap"  # lapped finisher is NOT a retirement
    assert by["PER"].status == "DNF"
    assert by["TSU"].status == "DNS"
    assert by["BOT"].status == "DSQ"
    assert by["PER"].grid_position is None  # no grid known for them
    assert by["VER"].event_date == datetime(2024, 5, 26, 13)


def test_two_laps_down_is_plural_and_qualifying_has_no_grid_or_lapped_status() -> None:
    rows = [
        {"driver_number": 1, "position": 1, "number_of_laps": 78},
        {"driver_number": 4, "position": 2, "number_of_laps": 76},
    ]
    race, _ = build_result_records("k", "R", rows, ACRONYMS, {}, None)
    assert {r.driver: r.status for r in race} == {"VER": "Finished", "NOR": "+2 Laps"}
    quali, _ = build_result_records("k", "Q", rows, ACRONYMS, {1: 5}, None)
    assert all(r.status == "Finished" and r.grid_position is None for r in quali)


def test_unknown_driver_or_missing_position_in_results() -> None:
    rows = [
        {"driver_number": 1, "position": None, "number_of_laps": 78},
        {"driver_number": 999, "position": 3, "number_of_laps": 78},
    ]
    records, skipped = build_result_records("k", "R", rows, ACRONYMS, {}, None)
    assert skipped == 1 and records[0].position is None


# ------------------------------------------------------------- grid sources


def test_grid_prefers_starting_grid_then_falls_back_and_says_which(monkeypatch) -> None:
    quali = [{"session_key": QUALI_KEY}]
    qual_result = [
        {"driver_number": 4, "position": 1},
        {"driver_number": 1, "position": 3},
    ]

    monkeypatch.setattr("src.ingest_openf1._get", router(full_table(), quali))
    assert resolve_grid(RACE, "R") == ({1: 2, 4: 1, 16: 3}, "starting_grid")

    table = {("starting_grid", QUALI_KEY): GRID}
    monkeypatch.setattr("src.ingest_openf1._get", router(table, quali))
    assert resolve_grid(RACE, "R")[1] == "starting_grid (qualifying session)"

    table = {("session_result", QUALI_KEY): qual_result}
    monkeypatch.setattr("src.ingest_openf1._get", router(table, quali))
    grid, source = resolve_grid(RACE, "R")
    assert grid == {4: 1, 1: 3}
    assert "penalties not reflected" in source

    monkeypatch.setattr("src.ingest_openf1._get", router({}, []))
    assert resolve_grid(RACE, "R") == ({}, "unavailable")
    assert resolve_grid(RACE, "Q") == ({}, "not applicable")


# ------------------------------------------------------- database round trip


@pytest.fixture
def temp_db(tmp_path):
    original = db_mod.engine
    db_mod.engine = create_engine(URL.create("sqlite", database=str(tmp_path / "t.db")))
    db_mod.SessionLocal.configure(bind=db_mod.engine)
    db_mod.init_db()
    yield
    db_mod.engine = original
    db_mod.SessionLocal.configure(bind=original)


def _counts():
    with db_mod.SessionLocal() as s:
        return s.query(db_mod.Lap).count(), s.query(db_mod.Result).count()


def test_ingest_stores_rows_and_rerunning_does_not_duplicate(
    monkeypatch, temp_db
) -> None:
    monkeypatch.setattr("src.ingest_openf1.resolve_session", lambda y, e, s: RACE)
    monkeypatch.setattr("src.ingest_openf1._get", router(full_table()))

    summary = ingest_session_openf1("2024 Monaco R")
    assert summary == {
        "session_key": "2024_Monaco_R",
        "laps": 5,  # six rows, one for an unknown driver
        "results": 6,
        "skipped": 1,
        "grid_source": "starting_grid",
    }
    assert _counts() == (5, 6)

    ingest_session_openf1("2024 Monaco R")  # idempotent
    assert _counts() == (5, 6)

    with db_mod.SessionLocal() as s:
        ver = s.query(db_mod.Result).filter_by(driver="VER").one()
        assert (ver.position, ver.grid_position, ver.points, ver.status) == (
            1,
            2,
            25.0,
            "Finished",
        )
        assert ver.event_date == datetime(2024, 5, 26, 13, 0)


def test_empty_answer_from_openf1_never_wipes_stored_results(
    monkeypatch, temp_db
) -> None:
    monkeypatch.setattr("src.ingest_openf1.resolve_session", lambda y, e, s: RACE)
    monkeypatch.setattr("src.ingest_openf1._get", router(full_table()))
    ingest_session_openf1("2024 Monaco R")

    table = full_table()
    table[("session_result", RACE_KEY)] = []  # OpenF1 has no results right now
    monkeypatch.setattr("src.ingest_openf1._get", router(table))
    summary = ingest_session_openf1("2024 Monaco R")

    assert summary["results"] == 0
    assert _counts() == (5, 6)  # results from the first run are still there


def test_session_with_no_data_at_all_is_an_error_not_a_silent_success(
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.ingest_openf1.resolve_session", lambda y, e, s: RACE)
    table = {("drivers", RACE_KEY): DRIVERS}
    monkeypatch.setattr("src.ingest_openf1._get", router(table))
    with pytest.raises(OpenF1Error, match="no lap or result data"):
        ingest_session_openf1("2024 Monaco R")

    monkeypatch.setattr("src.ingest_openf1._get", router({}))
    with pytest.raises(OpenF1Error, match="no drivers"):
        ingest_session_openf1("2024 Monaco R")


# --------------------------------------------------------------------- CLI


def test_cli_defaults_to_openf1(monkeypatch, capsys) -> None:
    import src.ingest as ingest

    monkeypatch.setattr(ingest, "setup_logging", lambda: None)
    monkeypatch.setattr(ingest, "init_db", lambda: None)
    monkeypatch.setattr(
        "src.ingest_openf1.ingest_session_openf1",
        lambda arg: {
            "session_key": "2024_Monaco_R",
            "laps": 3,
            "results": 2,
            "skipped": 0,
            "grid_source": "starting_grid",
        },
    )
    monkeypatch.setattr(sys, "argv", ["ingest", "--session", "2024 Monaco R"])
    ingest.main()
    out = capsys.readouterr().out
    assert "Done: 3 laps, 2 results stored for 2024_Monaco_R" in out
    assert "grid positions: starting_grid" in out
