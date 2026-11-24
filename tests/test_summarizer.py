import sqlite3

import httpx
import pytest

from src.summarizer import (
    MAX_ATTEMPTS,
    generate_summary,
    get_race_facts,
    is_usable_summary,
    summarize_race,
)

GOOD_SUMMARY = (
    "HAM won the race from second on the grid, finishing ahead of VER and "
    "NOR, while PER retired after an accident."
)


def _make_temp_db(tmp_path) -> str:
    db_path = str(tmp_path / "t.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE results (session_key TEXT, session_type TEXT, driver TEXT, "
        "position INTEGER, grid_position INTEGER, points REAL, status TEXT, "
        "event_date TEXT)"
    )
    conn.execute(
        "CREATE TABLE laps (session_key TEXT, driver TEXT, lap_number INTEGER, "
        "lap_time_seconds REAL, compound TEXT, stint INTEGER, "
        "is_personal_best INTEGER)"
    )
    conn.executemany(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
        [
            ("k", "R", "HAM", 1, 2, 25.0, "Finished", None),
            ("k", "R", "VER", 2, 1, 18.0, "Finished", None),
            ("k", "R", "NOR", 3, 3, 15.0, "Finished", None),
            ("k", "R", "PER", None, 5, 0.0, "Accident", None),
        ],
    )
    conn.executemany(
        "INSERT INTO laps VALUES (?,?,?,?,?,?,?)",
        [
            ("k", "HAM", 1, 2494.8, "MEDIUM", 1, 0),  # red-flag-inflated lap
            ("k", "HAM", 2, 88.5, "HARD", 2, 0),
            ("k", "VER", 2, 87.9, "HARD", 1, 1),  # genuine fastest lap
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_get_race_facts_orders_podium_and_counts_dnfs(tmp_path) -> None:
    facts = get_race_facts("k", db_path=_make_temp_db(tmp_path))
    assert [r["driver"] for r in facts["podium"]] == ["HAM", "VER", "NOR"]
    assert facts["dnf_count"] == 1
    assert len(facts["full_classification"]) == 4


def test_get_race_facts_ignores_red_flag_inflated_lap(tmp_path) -> None:
    """The 2494s lap (a red-flag stoppage folded into one lap) must not
    leak into the fastest-lap fact; same guard as the dashboard charts."""
    facts = get_race_facts("k", db_path=_make_temp_db(tmp_path))
    assert facts["fastest_lap"]["driver"] == "VER"
    assert facts["fastest_lap"]["lap_time"] == pytest.approx(87.9)


def test_get_race_facts_handles_missing_session(tmp_path) -> None:
    facts = get_race_facts("nope", db_path=_make_temp_db(tmp_path))
    assert facts["full_classification"] == []
    assert facts["fastest_lap"] is None


FACTS = {"podium": [{"driver": "HAM"}, {"driver": "VER"}, {"driver": "NOR"}]}


@pytest.mark.parametrize(
    "summary,expected",
    [
        (GOOD_SUMMARY, True),
        ("User Safety: safe", False),  # the real failure: a classifier's reply
        ("", False),
        (None, False),
        (
            "The race was held on Sunday and it was a very exciting event overall.",
            False,
        ),
    ],
)
def test_is_usable_summary(summary, expected) -> None:
    assert is_usable_summary(summary, FACTS) is expected


def test_summarize_race_no_results_returns_error(tmp_path) -> None:
    result = summarize_race("nope", db_path=_make_temp_db(tmp_path), api_key="sk-x")
    assert result["summary"] is None
    assert "No race results" in result["error"]


def test_summarize_race_without_api_key_returns_error(tmp_path) -> None:
    result = summarize_race("k", db_path=_make_temp_db(tmp_path), api_key="")
    assert result["summary"] is None
    assert "OPENROUTER_API_KEY" in result["error"]


def test_summarize_race_returns_valid_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.summarizer.generate_summary", lambda facts, api_key=None: GOOD_SUMMARY
    )
    result = summarize_race("k", db_path=_make_temp_db(tmp_path), api_key="sk-x")
    assert result["summary"] == GOOD_SUMMARY
    assert result["error"] is None
    assert result["facts"]["podium"][0]["driver"] == "HAM"


def test_summarize_race_retries_after_junk_then_succeeds(tmp_path, monkeypatch) -> None:
    """Reproduces the real bug: first the router returns 'User Safety:
    safe', then a proper model answers. The junk must never be shown."""
    replies = iter(["User Safety: safe", GOOD_SUMMARY])
    monkeypatch.setattr(
        "src.summarizer.generate_summary", lambda facts, api_key=None: next(replies)
    )
    result = summarize_race("k", db_path=_make_temp_db(tmp_path), api_key="sk-x")
    assert result["summary"] == GOOD_SUMMARY
    assert result["error"] is None


def test_summarize_race_gives_up_after_max_attempts_of_junk(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def junk(facts, api_key=None):
        calls.append(1)
        return "User Safety: safe"

    monkeypatch.setattr("src.summarizer.generate_summary", junk)
    result = summarize_race("k", db_path=_make_temp_db(tmp_path), api_key="sk-x")
    assert result["summary"] is None
    assert "unusable output" in result["error"]
    assert len(calls) == MAX_ATTEMPTS


def test_summarize_race_handles_401_without_retrying(tmp_path, monkeypatch) -> None:
    calls = []

    def fake(facts, api_key=None):
        calls.append(1)
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat")
        raise httpx.HTTPStatusError(
            "401", request=request, response=httpx.Response(401, request=request)
        )

    monkeypatch.setattr("src.summarizer.generate_summary", fake)
    result = summarize_race("k", db_path=_make_temp_db(tmp_path), api_key="sk-x")
    assert result["summary"] is None
    assert "401" in result["error"]
    assert len(calls) == 1  # an auth failure won't fix itself, so no retry


def test_summarize_race_handles_network_failure(tmp_path, monkeypatch) -> None:
    def fake(facts, api_key=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("src.summarizer.generate_summary", fake)
    result = summarize_race("k", db_path=_make_temp_db(tmp_path), api_key="sk-x")
    assert result["summary"] is None
    assert result["error"] is not None


def test_generate_summary_accepts_api_key_kwarg() -> None:
    import inspect

    assert "api_key" in inspect.signature(generate_summary).parameters


def test_lapped_cars_are_not_counted_as_retirements(tmp_path) -> None:
    db_path = str(tmp_path / "t.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE results (session_key TEXT, session_type TEXT, driver TEXT, "
        "position INTEGER, grid_position INTEGER, points REAL, status TEXT, "
        "event_date TEXT)"
    )
    conn.execute(
        "CREATE TABLE laps (session_key TEXT, driver TEXT, lap_number INTEGER, "
        "lap_time_seconds REAL, compound TEXT, stint INTEGER, is_personal_best INTEGER)"
    )
    conn.executemany(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
        [
            ("k", "R", "HAM", 1, 1, 25.0, "Finished", None),
            ("k", "R", "VER", 2, 2, 18.0, "+1 Lap", None),
            ("k", "R", "NOR", 3, 3, 15.0, "+2 Laps", None),
            ("k", "R", "ALB", 4, 4, 12.0, "Lapped", None),
            ("k", "R", "PER", 19, 5, 0.0, "Accident", None),
            ("k", "R", "LEC", 20, 6, 0.0, "DNF", None),
        ],
    )
    conn.commit()
    conn.close()
    assert get_race_facts("k", db_path=db_path)["dnf_count"] == 2  # PER and LEC only
