# Security Overview — CIA Triad Applied

This is a lightweight threat model, kept current as the project grows.

## Confidentiality

- Secrets (OpenRouter key, any API tokens) live only in `.env`, which is
  git-ignored. `.env.example` documents required variables with placeholder
  values only.
- If/when the FastAPI chatbot endpoint or Telegram bot is exposed beyond
  localhost, it requires an API key header (`X-API-Key`) checked against a
  value loaded from `.env`, plus a basic in-memory rate limiter
  (`src/rate_limit.py`) to prevent open abuse of the LLM-backed endpoints.
- Logging (`src/logging_conf.py`) redacts any string matching common
  token/key patterns before writing to disk.

## Integrity

- All external data (FastF1, OpenF1) is validated against pydantic schemas
  (`src/schemas.py`) before it is written to the database — malformed or
  unexpected API responses are rejected at the boundary, not silently stored.
- Database writes go through a single ingestion module (`src/ingest.py`)
  using transactions; no ad-hoc writes elsewhere in the codebase.
- Cached raw data files get a SHA-256 checksum recorded at write time
  (`src/integrity.py`) so corruption/tampering can be detected on read.

## Availability

- `/health` endpoint on the FastAPI service for basic liveness checks.
- Ingestion falls back to the local cache/last-known-good data if a live
  FastF1/OpenF1 call fails, rather than raising to the user.
- `docker-compose.yml` sets `restart: unless-stopped` on long-running
  services.
- Daily local backup of `data/f1.db` (see `scripts/backup.sh`) to a second
  local path, so a corrupted DB file isn't a total loss.

## Out of scope (for now, documented so it's a conscious choice)

- No authentication beyond a single shared API key — fine for a portfolio
  demo with a small number of users; would need real auth (OAuth or similar)
  before wider public use.
- No encryption at rest for the SQLite file — data is public F1 timing data,
  not sensitive; revisit if user accounts / personal data are ever added.
