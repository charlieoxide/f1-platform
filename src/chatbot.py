"""Module 5 — Knowledge Chatbot (text-to-SQL).

Design rule: the LLM only ever produces a SQL query against the real
schema. It never answers from its own training data — the response is
always the actual query result, so the chatbot can't invent facts.

Two independent safety layers, because LLM-generated SQL is untrusted
input (same as any other external data in this project):
  1. is_safe_select() rejects anything that isn't a single SELECT.
  2. The DB connection itself is opened read-only (sqlite3 URI mode=ro),
     so even a validation bug can't result in a write.
"""

from __future__ import annotations

import re
import sqlite3

import httpx

from src.config import settings
from src.dbutil import readonly_uri

SCHEMA_DESCRIPTION = """
Table: laps
  session_key   TEXT   -- e.g. '2024_Monaco_R'
  driver        TEXT   -- 3-letter code, e.g. 'VER'
  lap_number    INTEGER
  lap_time_seconds FLOAT  -- NULL if the lap had no valid time
  compound      TEXT   -- tyre compound, e.g. 'MEDIUM', 'HARD', 'SOFT'
  stint         INTEGER
  is_personal_best BOOLEAN
"""

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|CREATE|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)
_MAX_ROWS = 200


class UnsafeQueryError(ValueError):
    pass


def is_safe_select(sql: str) -> bool:
    """True only for a single, read-only SELECT statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    if ";" in stripped:  # no stacked statements
        return False
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return False
    if _FORBIDDEN.search(stripped):
        return False
    return True


def _enforce_row_limit(sql: str) -> str:
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {_MAX_ROWS}"


def run_query_safe(sql: str, db_path: str | None = None) -> list[dict]:
    if not is_safe_select(sql):
        raise UnsafeQueryError(f"Rejected non-SELECT or unsafe query: {sql!r}")

    sql = _enforce_row_limit(sql)
    uri = readonly_uri(db_path or settings.db_path)

    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def generate_sql(question: str, api_key: str | None = None) -> str:
    """Call the LLM to turn a natural-language question into SQL. Isolated
    behind this function so tests can monkeypatch it without hitting the
    network. api_key is injected (defaulting to the real config) rather
    than read from settings directly, so tests don't need to mutate the
    frozen global config object."""
    key = api_key if api_key is not None else settings.openrouter_api_key
    prompt = (
        "You translate questions into a single SQLite SELECT query.\n"
        f"Schema:\n{SCHEMA_DESCRIPTION}\n"
        "Rules: output ONLY the SQL query, no explanation, no markdown "
        "fences. Never use INSERT/UPDATE/DELETE/DROP. Always SELECT only.\n\n"
        f"Question: {question}\nSQL:"
    )
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            # OpenRouter's own auto-router for free models. Points at
            # whichever free model is currently available, so this doesn't
            # break again when an individual :free model gets retired
            # (which is exactly what happened to the slug this replaced).
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content.strip().strip("`").removeprefix("sql").strip()


def answer_question(question: str, api_key: str | None = None) -> dict:
    """Full pipeline: question -> SQL -> validated, read-only execution ->
    grounded result. No second LLM call for phrasing - the rows ARE the
    answer, which keeps this fully grounded and free of extra API cost.

    Availability: a failure anywhere in this pipeline (bad/missing API
    key, network issue, unsafe SQL) is returned as a structured error,
    never an uncaught exception that would take down the whole app."""
    key = api_key if api_key is not None else settings.openrouter_api_key
    if not key:
        return {
            "question": question,
            "sql": None,
            "rows": [],
            "error": "OPENROUTER_API_KEY is not set. Add it to .env and restart.",
        }

    try:
        sql = generate_sql(question, api_key=key)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (401, 403):
            msg = (
                f"OpenRouter rejected the API key ({status}). "
                "Check OPENROUTER_API_KEY in .env."
            )
        else:
            msg = f"OpenRouter request failed ({status})."
        return {"question": question, "sql": None, "rows": [], "error": msg}
    except httpx.RequestError:
        return {
            "question": question,
            "sql": None,
            "rows": [],
            "error": "Could not reach OpenRouter — check your internet connection.",
        }

    try:
        rows = run_query_safe(sql)
        return {"question": question, "sql": sql, "rows": rows, "error": None}
    except UnsafeQueryError as e:
        return {"question": question, "sql": sql, "rows": [], "error": str(e)}
