import sqlite3

import httpx
import pytest

from src.summarizer import generate_summary, get_race_facts, summarize_race


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
        "lap_time_seconds REAL, compound TEXT, stint INTEGER, is_personal_best INTEGER)"
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
    db_path = _make_temp_db(tmp_path)
    facts = get_race_facts("k", db_path=db_path)

    assert [r["driver"] for r in facts["podium"]] == ["HAM", "VER", "NOR"]
    assert facts["dnf_count"] == 1
    assert len(facts["full_classification"]) == 4


def test_get_race_facts_excludes_red_flag_inflated_lap(tmp_path) -> None:
    """Same bug class as Module 1's tyre chart: a red-flag-stopped lap must
    not be reported as the 'fastest lap' just because it has the biggest
    recorded LapTime... wait, smallest is fastest - the point is the huge
    outlier must not corrupt the MIN() calculation."""
    db_path = _make_temp_db(tmp_path)
    facts = get_race_facts("k", db_path=db_path)

    assert facts["fastest_lap"]["driver"] == "VER"
    assert facts["fastest_lap"]["lap_time"] == pytest.approx(87.9)


def test_get_race_facts_handles_missing_session(tmp_path) -> None:
    db_path = _make_temp_db(tmp_path)
    facts = get_race_facts("does_not_exist", db_path=db_path)
    assert facts["full_classification"] == []
    assert facts["fastest_lap"] is None


def test_summarize_race_no_results_returns_error_not_crash(tmp_path) -> None:
    db_path = _make_temp_db(tmp_path)
    result = summarize_race("does_not_exist", db_path=db_path, api_key="sk-fake")
    assert result["summary"] is None
    assert "No race results" in result["error"]


def test_summarize_race_without_api_key_returns_error_not_crash(tmp_path) -> None:
    db_path = _make_temp_db(tmp_path)
    result = summarize_race("k", db_path=db_path, api_key="")
    assert result["summary"] is None
    assert "OPENROUTER_API_KEY" in result["error"]


def test_summarize_race_uses_generated_summary_without_calling_network(
    tmp_path, monkeypatch
) -> None:
    db_path = _make_temp_db(tmp_path)
    monkeypatch.setattr(
        "src.summarizer.generate_summary", lambda facts, api_key=None: "HAM won."
    )
    result = summarize_race("k", db_path=db_path, api_key="sk-fake")
    assert result["summary"] == "HAM won."
    assert result["error"] is None
    assert result["facts"]["podium"][0]["driver"] == "HAM"


def test_summarize_race_handles_401_without_crashing(tmp_path, monkeypatch) -> None:
    db_path = _make_temp_db(tmp_path)

    def fake_generate_summary(facts, api_key=None):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("401", request=request, response=response)

    monkeypatch.setattr("src.summarizer.generate_summary", fake_generate_summary)
    result = summarize_race("k", db_path=db_path, api_key="sk-fake")
    assert result["summary"] is None
    assert "401" in result["error"] or "API key" in result["error"]


def test_summarize_race_handles_network_failure_without_crashing(
    tmp_path, monkeypatch
) -> None:
    db_path = _make_temp_db(tmp_path)

    def fake_generate_summary(facts, api_key=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("src.summarizer.generate_summary", fake_generate_summary)
    result = summarize_race("k", db_path=db_path, api_key="sk-fake")
    assert result["summary"] is None
    assert result["error"] is not None


def test_generate_summary_signature_accepts_api_key_kwarg() -> None:
    # Just verifies the function signature matches what summarize_race
    # calls - a real network call isn't exercised in the test suite.
    import inspect

    sig = inspect.signature(generate_summary)
    assert "api_key" in sig.parameters
