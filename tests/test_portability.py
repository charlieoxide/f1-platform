"""Things that behave differently across Windows / macOS / Linux."""

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from src.dbutil import _uri_for, readonly_uri

ROOT = Path(__file__).resolve().parent.parent


def test_windows_path_becomes_a_valid_sqlite_uri() -> None:
    """The bug: 'file:' + C:\\Users\\... is not a URI SQLite can open."""
    p = PureWindowsPath(r"C:\Users\charlie\My Projects\f1-platform\data\f1.db")
    assert _uri_for(p) == (
        "file:///C:/Users/charlie/My%20Projects/f1-platform/data/f1.db?mode=ro"
    )


def test_posix_path_with_spaces_and_percent_is_encoded() -> None:
    p = PurePosixPath("/home/charlie/F1 the Project/100%/f1.db")
    assert _uri_for(p) == "file:///home/charlie/F1%20the%20Project/100%25/f1.db?mode=ro"


def test_readonly_uri_really_opens_read_only_in_a_folder_with_spaces(tmp_path) -> None:
    folder = tmp_path / "my data folder"
    folder.mkdir()
    db = folder / "f1.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()

    ro = sqlite3.connect(readonly_uri(db), uri=True)
    try:
        assert ro.execute("SELECT x FROM t").fetchall() == [(1,)]
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM t")
    finally:
        ro.close()


def test_engine_url_handles_absolute_paths_with_spaces(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL

    db = tmp_path / "with space" / "x.db"
    db.parent.mkdir()
    engine = create_engine(URL.create("sqlite", database=str(db)))
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
    assert db.exists()


def _run(code: str, cwd: Path, env_extra: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_dashboard_renders_from_another_folder_without_pythonpath(tmp_path) -> None:
    """Runs the real dashboard script (Streamlit's headless test harness) from
    an unrelated working directory with PYTHONPATH unset. A bare server start
    would not catch import errors; executing the script does."""
    db = tmp_path / "t.db"
    seed = f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from src.db import Lap, Result, SessionLocal, init_db
        init_db()
        with SessionLocal.begin() as s:
            s.add_all([
                Lap(session_key="k", driver="VER", lap_number=1,
                    lap_time_seconds=80.0, compound="MEDIUM", stint=1),
                Result(session_key="k", session_type="R", driver="VER",
                       position=1, grid_position=1, points=25.0, status="Finished"),
            ])
    """
    assert _run(seed, tmp_path, {"DB_PATH": str(db)}).returncode == 0

    script = str(ROOT / "src" / "dashboard.py")
    render = f"""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file({script!r}, default_timeout=120)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        names = [h.value for h in at.subheader]
        assert "Driver coach" in names and "Race summary" in names, names
    """
    result = _run(render, tmp_path, {"DB_PATH": str(db)})
    assert result.returncode == 0, result.stderr[-2000:]
