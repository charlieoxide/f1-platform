"""Module 3 — Race Summariser.

Same grounding rule as the chatbot (module 5): the LLM only ever writes
prose from facts we retrieved from the database. It is never asked to
recall or guess a race result from its own training data - the facts
dict is the entire source of truth, and the prompt says so explicitly.

LLM output is untrusted input, like any other external data here: the
free auto-router can hand the request to an unsuitable model (a safety
classifier once answered with just "User Safety: safe"), so every
response is checked before it is shown, and retried if it fails.
"""

from __future__ import annotations

import json
import re
import sqlite3

import httpx

from src.config import settings
from src.dbutil import readonly_uri

MAX_PLAUSIBLE_LAP_SECONDS = 200  # same red-flag-timing guard as analysis.py
MAX_ATTEMPTS = 3
MIN_SUMMARY_CHARS = 60


def _finished(status: str) -> bool:
    """A car a lap or more down is classified ('+1 Lap', 'Lapped'): it
    finished, it did not retire."""
    s = status.strip()
    return s == "Finished" or s.startswith("+") or s.lower() == "lapped"


def get_race_facts(session_key: str, db_path: str | None = None) -> dict:
    """Pull the structured facts a race summary is grounded in. Read-only
    connection, same pattern as the chatbot module."""
    conn = sqlite3.connect(readonly_uri(db_path or settings.db_path), uri=True)
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
            if r["position"] is None or (r["status"] and not _finished(r["status"]))
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


def is_usable_summary(summary: str | None, facts: dict) -> bool:
    """A real summary is more than a few words and mentions at least one
    podium driver by the three-letter code we gave it. This rejects
    off-task output such as a classifier's 'User Safety: safe'."""
    if not summary or len(summary.strip()) < MIN_SUMMARY_CHARS:
        return False
    codes = [r["driver"] for r in facts["podium"] if r.get("driver")]
    return any(
        re.search(rf"\b{re.escape(code)}\b", summary, re.IGNORECASE) for code in codes
    )


def generate_summary(facts: dict, api_key: str | None = None) -> str:
    """One LLM call: facts dict -> prose. Isolated behind this function so
    tests can monkeypatch it without hitting the network."""
    key = api_key if api_key is not None else settings.openrouter_api_key
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write short Formula 1 race reports. You use "
                        "ONLY the facts you are given and never add "
                        "anything from memory."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Write a factual race summary of 3-5 sentences from "
                        "these facts. Refer to drivers by the three-letter "
                        "codes exactly as given (e.g. VER). If a fact is not "
                        "in the data, do not mention it.\n\n"
                        f"Facts (JSON): {json.dumps(facts, default=str)}"
                    ),
                },
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content") or ""
    return content.strip()


def summarize_race(
    session_key: str, db_path: str | None = None, api_key: str | None = None
) -> dict:
    """Full pipeline: DB facts -> grounded LLM prose, validated. A bad/
    missing key, a network failure, or unusable model output is returned
    as a structured error, never an uncaught exception."""
    facts = get_race_facts(session_key, db_path=db_path)

    def fail(error: str) -> dict:
        return {
            "session_key": session_key,
            "facts": facts,
            "summary": None,
            "error": error,
        }

    if not facts["full_classification"]:
        return fail(f"No race results found for '{session_key}'.")

    key = api_key if api_key is not None else settings.openrouter_api_key
    if not key:
        return fail("OPENROUTER_API_KEY is not set. Add it to .env and restart.")

    for _ in range(MAX_ATTEMPTS):
        try:
            candidate = generate_summary(facts, api_key=key)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (401, 403):
                return fail(
                    f"OpenRouter rejected the API key ({status}). "
                    "Check OPENROUTER_API_KEY in .env."
                )
            return fail(f"OpenRouter request failed ({status}).")
        except httpx.RequestError:
            return fail("Could not reach OpenRouter — check your connection.")

        if is_usable_summary(candidate, facts):
            return {
                "session_key": session_key,
                "facts": facts,
                "summary": candidate,
                "error": None,
            }

    return fail(
        f"The model returned unusable output {MAX_ATTEMPTS} times in a row. "
        "Try again, or pin a specific model with OPENROUTER_MODEL in .env."
    )
