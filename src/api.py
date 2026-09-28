"""FastAPI service. Module 5 (chatbot) lives behind here, plus a /health
endpoint for availability checks. Any endpoint beyond /health requires the
shared API key (confidentiality) and is rate-limited (availability/abuse
prevention)."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import FastAPI, Header, HTTPException

from src.config import settings
from src.logging_conf import setup_logging

setup_logging()

app = FastAPI(title="F1 Unified Platform API")

# Simple in-memory token-bucket rate limiter — fine for a single-instance
# portfolio deployment; swap for Redis-backed if this ever scales out.
_REQUESTS: dict[str, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 60
_MAX_REQUESTS = 30


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    recent = [t for t in _REQUESTS[client_id] if now - t < _WINDOW_SECONDS]
    if len(recent) >= _MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    recent.append(now)
    _REQUESTS[client_id] = recent


def _require_api_key(x_api_key: str | None) -> None:
    if not settings.f1_api_key:
        return  # no key configured (local dev) — skip check
    if x_api_key != settings.f1_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(query: dict, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    _check_rate_limit(x_api_key or "anonymous")

    question = query.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    # Placeholder — wire to text-to-SQL / RAG module here.
    return {"answer": f"(stub) received: {question}"}
