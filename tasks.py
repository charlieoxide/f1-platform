#!/usr/bin/env python3
"""Cross-platform task runner: the same commands on Windows, macOS and Linux.

    python tasks.py test                  lint + format check + tests
    python tasks.py dashboard             open the Streamlit app
    python tasks.py coach --session "2024 Monaco Q" --driver NOR --reference VER
    python tasks.py ingest --session "2024 Monaco R"
    python tasks.py train                 train the outcome predictor
    python tasks.py sql "SELECT driver, COUNT(*) FROM laps GROUP BY driver"
    python tasks.py backup                snapshot the database
    python tasks.py probe --probe "2024 Monaco R"   what OpenF1 returns for a session

Why not make/bash? `make`, the sqlite3 CLI and shell scripts are not on a
stock Windows install. Everything here is plain Python (stdlib + the
project's own dependencies), and every command runs through the same
interpreter that runs this file, so an activated virtualenv is respected.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINT_PATHS = ["src", "tests", "tasks.py"]

# First batch of 2024 events used to build the predictor's training data.
SEASON_EVENTS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
    "Monaco", "Canada", "Spain", "Austria", "Silverstone",
]  # fmt: skip


MIN_PYTHON = (3, 11)
MAX_SUPPORTED_PYTHON = (3, 14)


def _python_version() -> tuple[int, int]:
    return sys.version_info[0], sys.version_info[1]


def python_problem(version: tuple[int, int]) -> tuple[str, str] | None:
    """("error" | "warning", message) if this Python is a bad fit, else None.
    Caught here so the user gets a plain explanation instead of a cryptic
    pip failure ("no matching distribution")."""
    shown = f"{version[0]}.{version[1]}"
    if version < MIN_PYTHON:
        return "error", f"Python {shown} is too old; this project needs 3.11 or newer."
    if version > MAX_SUPPORTED_PYTHON:
        return (
            "warning",
            f"Python {shown} is newer than this project supports (3.11-3.14). "
            "FastF1 needs pandas 2, which has no builds for it yet, so installing "
            "the requirements will fail. Use Python 3.12 (py -3.12 on Windows).",
        )
    return None


def _call(*args: str) -> int:
    cmd = [sys.executable, *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def _module(*args: str) -> int:
    return _call("-m", *args)


def cmd_lint(_extra: list[str]) -> int:
    return _module("ruff", "check", *LINT_PATHS) or _module(
        "black", "--check", *LINT_PATHS
    )


def cmd_test(extra: list[str]) -> int:
    return cmd_lint([]) or _module("pytest", "-q", *extra)


def cmd_install(_extra: list[str]) -> int:
    return _module("pip", "install", "-r", "requirements.txt")


def cmd_dashboard(extra: list[str]) -> int:
    return _module("streamlit", "run", "src/dashboard.py", *extra)


def cmd_api(extra: list[str]) -> int:
    return _module("uvicorn", "src.api:app", "--reload", *extra)


def cmd_train(extra: list[str]) -> int:
    return _module("src.predictor", *extra)


def cmd_ingest(extra: list[str]) -> int:
    return _module("src.ingest", *extra)


def cmd_coach(extra: list[str]) -> int:
    return _module("src.telemetry", *extra)


def cmd_probe(extra: list[str]) -> int:
    return _module("src.openf1", *extra)


def cmd_ingest_season(year: str) -> int:
    failed = []
    for event in SEASON_EVENTS:
        for session_type in ("Q", "R"):
            label = f"{year} {event} {session_type}"
            print(f"== {label} ==", flush=True)
            if _module("src.ingest", "--session", label) != 0:
                failed.append(label)
    if failed:
        print("\nFailed sessions:", *failed, sep="\n  ")
        return 1
    return 0


def backup_db(
    src: Path, dest_dir: Path, keep: int = 14, now: datetime | None = None
) -> Path | None:
    """Consistent snapshot using SQLite's own backup API, so it is safe while
    the app has the database open (a plain file copy can fail on Windows,
    where open files are locked, or catch a half-written page elsewhere).
    Keeps only the newest `keep` snapshots."""
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    target = dest_dir / f"f1_{now:%Y%m%d_%H%M%S}.db"
    source = sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    for old in sorted(dest_dir.glob("f1_*.db"), reverse=True)[keep:]:
        old.unlink()
    return target


def cmd_backup(dest: str | None) -> int:
    from src.config import settings

    dest_dir = Path(dest) if dest else Path.home() / "f1-platform-backups"
    target = backup_db(Path(settings.db_path), dest_dir)
    if target is None:
        print(f"No database at {settings.db_path}; nothing to back up.")
        return 1
    print(f"Backed up to {target}")
    return 0


def cmd_sql(query: str, db_path: str | None = None) -> int:
    """Read-only SELECT against the project database (the sqlite3 CLI isn't
    installed on most Windows machines). Uses the same validation and
    read-only connection as the chatbot."""
    import pandas as pd

    from src.chatbot import UnsafeQueryError, run_query_safe

    try:
        rows = run_query_safe(query, db_path=db_path)
    except UnsafeQueryError as e:
        print(e)
        return 2
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return 2
    print(pd.DataFrame(rows).to_string(index=False) if rows else "(no rows)")
    return 0


PASS_THROUGH = {
    "install": cmd_install,
    "lint": cmd_lint,
    "test": cmd_test,
    "dashboard": cmd_dashboard,
    "api": cmd_api,
    "train": cmd_train,
    "ingest": cmd_ingest,
    "coach": cmd_coach,
    "probe": cmd_probe,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-platform project tasks.")
    sub = parser.add_subparsers(dest="task", required=True)
    for name in PASS_THROUGH:
        sub.add_parser(name, help=f"run '{name}' (extra arguments are passed on)")
    p = sub.add_parser("ingest-season", help="ingest Q+R for a season's first events")
    p.add_argument("year")
    p = sub.add_parser("backup", help="snapshot the database")
    p.add_argument(
        "--dest", help="folder for snapshots (default ~/f1-platform-backups)"
    )
    p = sub.add_parser("sql", help="run a read-only SELECT")
    p.add_argument("query")

    problem = python_problem(_python_version())
    if problem:
        level, message = problem
        print(f"{level.upper()}: {message}", file=sys.stderr)
        if level == "error":
            return 2

    args, extra = parser.parse_known_args(argv)
    if args.task in PASS_THROUGH:
        return PASS_THROUGH[args.task](extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.task == "ingest-season":
        return cmd_ingest_season(args.year)
    if args.task == "backup":
        return cmd_backup(args.dest)
    return cmd_sql(args.query)


if __name__ == "__main__":
    sys.exit(main())
