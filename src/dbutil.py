"""Small cross-platform helper for opening SQLite read-only."""

from __future__ import annotations

from pathlib import Path, PurePath


def _uri_for(path: PurePath) -> str:
    # Path.as_uri() gives file:///C:/Users/... on Windows and percent-encodes
    # spaces. Building "file:" + str(path) by hand yields file:C:\Users\...,
    # which is not a valid SQLite URI on Windows (and breaks on Linux paths
    # containing spaces or '%' too).
    return path.as_uri() + "?mode=ro"


def readonly_uri(path: str | Path) -> str:
    """URI for sqlite3.connect(uri, uri=True) that opens `path` read-only on
    every OS. Read-only is a safety layer: modules that only read (chatbot,
    summariser, predictor) cannot write even if their own checks have a bug."""
    return _uri_for(Path(path).resolve())
