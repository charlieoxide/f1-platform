# F1 Unified Platform

A single platform combining five F1 data/ML capabilities on top of one shared
data layer (FastF1 + OpenF1):

1. **Race Dashboard** — pace deltas, tyre degradation, track evolution
2. **AI Driver Coach** — telemetry comparison against a reference lap
3. **Race Summariser** — LLM-generated post-session report
4. **Outcome Predictor** — gradient boosting finishing-position model
5. **Knowledge Chatbot** — text-to-SQL / RAG over the session database

Built and run by a single developer on modest local hardware (4-thread CPU,
~5.7GB RAM, ~11GB free disk, no dedicated GPU). That constraint shapes several
decisions documented below — see `docs/DECISIONS.md`.

## Architecture

```
FastF1 / OpenF1  ──▶  ingestion (validated via pydantic)  ──▶  SQLite (f1.db)
                                                                     │
                        ┌────────────────────────────────────────────┼──────────────────────┐
                        ▼                        ▼                   ▼                       ▼
                  Streamlit dashboard      FastAPI service     Telegram bot            (future) React UI
                  (modules 1-4 UI)         (module 5 chatbot,  (chatbot access
                                            health checks)      point)
```

## Hardware-aware decisions (see docs/DECISIONS.md for full rationale)

- Deep learning (LSTM autoencoder) is **trained on Google Colab's free GPU tier**,
  not locally. Only inference happens on-device.
- Classical ML (gradient boosting predictor) trains locally via scikit-learn —
  lightweight, already installed, no GPU needed.
- Docker images use `python:3.11-slim` and are pruned regularly; disk headroom
  is treated as a hard constraint, not an afterthought.
- SQLite over Postgres for now — zero daemon overhead on a RAM-constrained box.

## Quickstart

```bash
cp .env.example .env          # fill in any API keys you use (OpenRouter, etc.)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test                     # runs lint + pytest
make ingest RACE="2024 Monaco R"   # pull one session into local SQLite
make dashboard                # launch Streamlit locally
```

Docker (optional, for reproducibility / final packaging):

```bash
docker compose up --build
```

## Repo layout

```
f1-platform/
├── src/            # application code (config, db, schemas, ingestion, api)
├── tests/          # pytest suite
├── scripts/        # disk cleanup, one-off ops scripts
├── notebooks/       # Colab-ready training notebooks (LSTM autoencoder etc.)
├── docs/           # SECURITY.md, DECISIONS.md
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Process

- All work happens on feature branches, merged via PR once CI is green
  (lint + tests). No direct pushes to `main`.
- See `docs/SECURITY.md` for the CIA-triad threat model applied to this project.
