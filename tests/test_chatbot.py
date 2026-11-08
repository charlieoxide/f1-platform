import sqlite3

import pytest

from src.chatbot import (
    UnsafeQueryError,
    answer_question,
    is_safe_select,
    run_query_safe,
)
from src.dbutil import readonly_uri


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT driver FROM laps", True),
        ("select * from laps where driver = 'VER'", True),
        ("SELECT driver FROM laps;", True),  # single trailing semicolon ok
        ("DROP TABLE laps", False),
        ("SELECT * FROM laps; DROP TABLE laps;", False),  # stacked statements
        ("INSERT INTO laps VALUES (1)", False),
        ("UPDATE laps SET driver='X'", False),
        ("", False),
        ("   ", False),
        ("ATTACH DATABASE '/etc/passwd' AS x", False),
    ],
)
def test_is_safe_select(sql: str, expected: bool) -> None:
    assert is_safe_select(sql) is expected


def _make_temp_db(tmp_path) -> str:
    db_path = tmp_path / "t.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE laps (session_key TEXT, driver TEXT, lap_number INTEGER, "
        "lap_time_seconds REAL, compound TEXT, stint INTEGER, is_personal_best INTEGER)"
    )
    conn.execute("INSERT INTO laps VALUES ('k','VER',1,80.0,'MEDIUM',1,0)")
    conn.commit()
    conn.close()
    return str(db_path)


def test_run_query_safe_executes_select(tmp_path) -> None:
    db_path = _make_temp_db(tmp_path)
    rows = run_query_safe("SELECT driver, lap_time_seconds FROM laps", db_path=db_path)
    assert rows == [{"driver": "VER", "lap_time_seconds": 80.0}]


def test_run_query_safe_rejects_write_before_touching_db(tmp_path) -> None:
    db_path = _make_temp_db(tmp_path)
    with pytest.raises(UnsafeQueryError):
        run_query_safe("DELETE FROM laps", db_path=db_path)


def test_readonly_connection_blocks_writes_even_bypassing_validation(tmp_path) -> None:
    """Second safety layer, tested directly: open the same read-only URI
    the way run_query_safe does, and prove a write fails at the DB level
    even without going through is_safe_select at all."""
    db_path = _make_temp_db(tmp_path)
    conn = sqlite3.connect(readonly_uri(db_path), uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM laps")
    finally:
        conn.close()


def test_run_query_safe_enforces_row_limit(tmp_path) -> None:
    db_path = tmp_path / "t.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE laps (driver TEXT)")
    conn.executemany("INSERT INTO laps VALUES (?)", [(f"D{i}",) for i in range(300)])
    conn.commit()
    conn.close()

    rows = run_query_safe("SELECT driver FROM laps", db_path=str(db_path))
    assert len(rows) == 200  # default cap applied automatically


def test_answer_question_uses_generated_sql_without_calling_network(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.chatbot.generate_sql", lambda q, api_key=None: "SELECT 1 AS x"
    )
    monkeypatch.setattr("src.chatbot.run_query_safe", lambda sql: [{"x": 1}])

    result = answer_question("how many laps?", api_key="sk-fake")
    assert result["sql"] == "SELECT 1 AS x"
    assert result["rows"] == [{"x": 1}]
    assert result["error"] is None


def test_answer_question_surfaces_unsafe_query_as_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.chatbot.generate_sql", lambda q, api_key=None: "DROP TABLE laps"
    )

    result = answer_question("delete everything", api_key="sk-fake")
    assert result["rows"] == []
    assert "Rejected" in result["error"]


def test_answer_question_without_api_key_returns_error_not_crash() -> None:
    result = answer_question("anything", api_key="")
    assert result["error"] is not None
    assert "OPENROUTER_API_KEY" in result["error"]
    assert result["rows"] == []


def test_answer_question_handles_401_without_crashing(monkeypatch) -> None:
    import httpx

    def fake_generate_sql(question, api_key=None):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError(
            "401 Unauthorized", request=request, response=response
        )

    monkeypatch.setattr("src.chatbot.generate_sql", fake_generate_sql)
    result = answer_question("anything", api_key="sk-fake")
    assert result["error"] is not None
    assert "401" in result["error"] or "API key" in result["error"]
    assert result["rows"] == []


def test_answer_question_handles_network_failure_without_crashing(monkeypatch) -> None:
    import httpx

    def fake_generate_sql(question, api_key=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("src.chatbot.generate_sql", fake_generate_sql)
    result = answer_question("anything", api_key="sk-fake")
    assert result["error"] is not None
    assert result["rows"] == []
