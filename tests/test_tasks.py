import sqlite3
from datetime import datetime, timedelta

import pytest

import tasks


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_call(cmd, cwd=None):
        recorded.append(cmd)
        return 0

    monkeypatch.setattr(tasks.subprocess, "call", fake_call)
    return recorded


def test_test_task_runs_lint_format_then_pytest(calls) -> None:
    assert tasks.main(["test"]) == 0
    tools = [c[2] for c in calls]  # python -m <tool>
    assert tools == ["ruff", "black", "pytest"]
    assert all(c[0] == tasks.sys.executable for c in calls)  # same interpreter/venv


def test_test_task_stops_at_the_first_failure(monkeypatch) -> None:
    seen = []

    def fake_call(cmd, cwd=None):
        seen.append(cmd[2])
        return 1 if cmd[2] == "ruff" else 0

    monkeypatch.setattr(tasks.subprocess, "call", fake_call)
    assert tasks.main(["test"]) == 1
    assert seen == ["ruff"]  # no point formatting/testing after a lint failure


def test_arguments_are_forwarded_untouched(calls) -> None:
    argv = [
        "--session",
        "2024 Monaco Q",
        "--driver",
        "NOR",
        "--reference",
        "VER",
        "--trace-at",
        "1230",
        "2080",
    ]
    assert tasks.main(["coach", *argv]) == 0
    assert calls[0][1:3] == ["-m", "src.telemetry"]
    assert calls[0][3:] == argv


def test_dashboard_and_ingest_commands(calls) -> None:
    tasks.main(["dashboard"])
    tasks.main(["ingest", "--session", "2024 Monaco R"])
    assert calls[0][1:] == ["-m", "streamlit", "run", "src/dashboard.py"]
    assert calls[1][1:] == ["-m", "src.ingest", "--session", "2024 Monaco R"]


def test_unknown_arguments_are_rejected_for_tasks_that_take_none() -> None:
    with pytest.raises(SystemExit):
        tasks.main(["backup", "--bogus"])


def test_ingest_season_continues_past_failures_and_reports_them(
    monkeypatch, capsys
) -> None:
    attempted = []

    def fake_call(cmd, cwd=None):
        label = cmd[-1]
        attempted.append(label)
        return 1 if label == "2024 Japan Q" else 0

    monkeypatch.setattr(tasks.subprocess, "call", fake_call)
    assert tasks.main(["ingest-season", "2024"]) == 1
    assert len(attempted) == len(tasks.SEASON_EVENTS) * 2  # kept going
    assert attempted[0] == "2024 Bahrain Q" and attempted[1] == "2024 Bahrain R"
    assert "2024 Japan Q" in capsys.readouterr().out


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (42)")
    conn.commit()
    conn.close()


def test_backup_is_a_consistent_copy_and_prunes_old_snapshots(tmp_path) -> None:
    src = tmp_path / "f1.db"
    _make_db(src)
    dest = tmp_path / "backups"
    start = datetime(2026, 1, 1, 12, 0, 0)
    made = [
        tasks.backup_db(src, dest, keep=3, now=start + timedelta(seconds=i))
        for i in range(5)
    ]

    kept = sorted(p.name for p in dest.glob("f1_*.db"))
    assert kept == [p.name for p in made[-3:]]  # only the newest three
    conn = sqlite3.connect(made[-1])
    assert conn.execute("SELECT x FROM t").fetchall() == [(42,)]
    conn.close()


def test_backup_works_while_another_connection_is_open(tmp_path) -> None:
    src = tmp_path / "f1.db"
    _make_db(src)
    holder = sqlite3.connect(src)  # the app has the database open
    try:
        assert tasks.backup_db(src, tmp_path / "b") is not None
    finally:
        holder.close()


def test_backup_without_a_database_returns_none(tmp_path) -> None:
    assert tasks.backup_db(tmp_path / "missing.db", tmp_path / "b") is None


def test_sql_task_is_read_only_and_validated(tmp_path, capsys) -> None:
    db = tmp_path / "f1.db"
    _make_db(db)
    assert tasks.cmd_sql("SELECT x FROM t", db_path=str(db)) == 0
    assert "42" in capsys.readouterr().out
    assert tasks.cmd_sql("DELETE FROM t", db_path=str(db)) == 2
    assert tasks.cmd_sql("SELECT nope FROM t", db_path=str(db)) == 2
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone() == (1,)  # untouched
    conn.close()


@pytest.mark.parametrize("version", [(3, 11), (3, 12), (3, 14)])
def test_supported_python_versions_pass(version) -> None:
    assert tasks.python_problem(version) is None


def test_python_that_is_too_old_is_an_error() -> None:
    level, message = tasks.python_problem((3, 10))
    assert level == "error" and "3.11" in message


def test_python_that_is_too_new_gets_a_clear_warning_pointing_at_3_12() -> None:
    level, message = tasks.python_problem((3, 15))
    assert level == "warning"
    assert "3.15" in message and "3.12" in message and "pandas" in message


def test_main_refuses_to_run_on_an_unsupported_old_python(
    monkeypatch, calls, capsys
) -> None:
    monkeypatch.setattr(tasks, "_python_version", lambda: (3, 9))
    assert tasks.main(["test"]) == 2
    assert calls == []  # nothing was run
    assert "too old" in capsys.readouterr().err


def test_main_warns_but_still_runs_on_a_newer_python(
    monkeypatch, calls, capsys
) -> None:
    monkeypatch.setattr(tasks, "_python_version", lambda: (3, 15))
    assert tasks.main(["lint"]) == 0
    assert calls  # it still ran
    assert "WARNING" in capsys.readouterr().err


def test_probe_task_forwards_to_the_openf1_module(calls) -> None:
    tasks.main(["probe", "--probe", "2024 Monaco R"])
    assert calls[0][1:] == ["-m", "src.openf1", "--probe", "2024 Monaco R"]
