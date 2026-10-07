"""Module 3 — Race Summariser.

Same grounding rule as the chatbot (module 5): the LLM only ever writes
prose from facts we retrieved from the database. It is never asked to
recall or guess a race result from its own training data - the facts
dict is the entire source of truth, and the prompt says so explicitly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx

from src.config import settings

MAX_PLAUSIBLE_LAP_SECONDS = 200  # same red-flag-timing guard as analysis.py


def get_race_facts(session_key: str, db_path: str | None = None) -> dict:
    """Pull the structured facts a race summary is grounded in. Read-only
    connection, same pattern as the chatbot module."""
    resolved = Path(db_path or settings.db_path).resolve()
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row

        classification = [
            dict(r)
            for r in conn.execute(
                "SELECT driver, position, grid_position, points, status "
                "FROM results WHERE session_key = ? AND session_type = 'R' "
                "ORDER BY position IS NULL, position",
                (session_key,),
            ).fetchall()
        ]

        fastest = conn.execute(
            "SELECT driver, MIN(lap_time_seconds) AS lap_time "
            "FROM laps WHERE session_key = ? AND lap_time_seconds IS NOT NULL "
            "AND lap_time_seconds <= ?",
            (session_key, MAX_PLAUSIBLE_LAP_SECONDS),
        ).fetchone()

        dnf_count = sum(
            1
            for r in classification
            if r["position"] is None or (r["status"] and r["status"] != "Finished")
        )

        return {
            "session_key": session_key,
            "podium": classification[:3],
            "full_classification": classification,
            "fastest_lap": dict(fastest) if fastest and fastest["driver"] else None,
            "dnf_count": dnf_count,
        }
    finally:
        conn.close()


def generate_summary(facts: dict, api_key: str | None = None) -> str:
    """Call the LLM to turn the facts dict into readable prose. Isolated
    behind this function so tests can monkeypatch it without hitting the
    network - same pattern as chatbot.generate_sql."""
    key = api_key if api_key is not None else settings.openrouter_api_key
    prompt = (
        "Write a short, factual race summary (3-5 sentences) using ONLY "
        "the facts below. Do not invent anything not present here - if a "
        "fact isn't in this data, don't mention it.\n\n"
        f"Facts: {facts}\n\nSummary:"
    )
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def summarize_race(
    session_key: str, db_path: str | None = None, api_key: str | None = None
) -> dict:
    """Full pipeline: DB facts -> grounded LLM prose. Same graceful-failure
    shape as chatbot.answer_question - a bad/missing key or a network
    failure is a structured error, never an uncaught exception."""
    facts = get_race_facts(session_key, db_path=db_path)
    if not facts["full_classification"]:
        return {
            "session_key": session_key,
            "facts": facts,
            "summary": None,
            "error": f"No race results found for '{session_key}'.",
        }

    key = api_key if api_key is not None else settings.openrouter_api_key
    if not key:
        return {
            "session_key": session_key,
            "facts": facts,
            "summary": None,
            "error": "OPENROUTER_API_KEY is not set. Add it to .env and restart.",
        }

    try:
        summary = generate_summary(facts, api_key=key)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        msg = (
            f"OpenRouter rejected the API key ({status}). Check "
            "OPENROUTER_API_KEY in .env."
            if status in (401, 403)
            else f"OpenRouter request failed ({status})."
        )
        return {
            "session_key": session_key,
            "facts": facts,
            "summary": None,
            "error": msg,
        }
    except httpx.RequestError:
        return {
            "session_key": session_key,
            "facts": facts,
            "summary": None,
            "error": "Could not reach OpenRouter — check your internet connection.",
        }

    return {
        "session_key": session_key,
        "facts": facts,
        "summary": summary,
        "error": None,
    }
