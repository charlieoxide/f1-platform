> **Note:** this guide was written at the start of the project. The system as built
> differs in places (the driver coach is deterministic signal processing on OpenF1
> telemetry, not an LSTM trained on Colab; there is no ONNX dependency). For what is
> current, see `README.md` and `docs/DECISIONS.md`.

# F1 Unified Platform — Study Guide

This explains *what* we're building, *why* each piece exists, and the
concepts behind it — so you can maintain and extend it yourself, not just
run the commands.

---

## 1. The Big Picture

One project, one shared data layer, five modules sitting on top of it.

```
FastF1 / OpenF1 → validated ingestion → SQLite → { 5 modules } → you / anyone
```

Instead of five disconnected scripts (which is what most "F1 AI project"
tutorials produce), everything reads from and writes to the same database.
That's the difference between a portfolio of demos and a *platform* — and
it's the difference a hiring manager at an F1 team actually notices.

---

## 2. The Data Layer

**FastF1** is a Python library that wraps F1's live timing feed and gives
you laps, telemetry, tyres, weather, and session results as pandas
DataFrames. **OpenF1** is a complementary open API for similar data, useful
as a cross-check or for data FastF1 doesn't cover.

**Ingestion** (`src/ingest.py`) is the *only* place data enters the system.
Why centralize it instead of pulling data wherever it's needed?

- One place to fix bugs in how data is cleaned
- One place to enforce validation (next point)
- One place to add caching/fallback logic

**Validation** (`src/schemas.py`) uses pydantic — a library that checks
data against a defined shape (types, ranges, required fields) and rejects
anything that doesn't match. Example: `lap_time_seconds` must be `>= 0`.
Real API data is messy (DNFs, missing sectors, safety-car laps with no
valid time) — validation is what stops garbage from quietly becoming part
of your dataset and corrupting downstream analysis or model training.

This maps directly to the **Integrity** leg of the CIA triad (section 5).

---

## 3. The Five Modules

### Module 1 — Race Dashboard
Pulls from the DB, shows pace deltas between drivers, tyre degradation per
stint, track evolution across a session. Currently a minimal Streamlit page
(`src/dashboard.py`) reading raw lap times — the next step is adding tyre
compound coloring and a track-evolution line.

### Module 2 — AI Driver Coach
Compares a lap's telemetry (braking point, throttle trace, racing line)
against a reference lap and flags where time is lost. This reuses the
anomaly-detection approach from your existing Telemetry Anomaly Detector
(Isolation Forest + LSTM Autoencoder) — the LSTM piece is trained on Colab
(see section 6) and does inference locally via ONNX Runtime.

### Module 3 — Race Summariser
An LLM call (via OpenRouter, same integration as your Telegram bot) over
the *structured* tables already in the DB — fastest laps, pit windows,
race-control messages — to produce a written report. Feeding it structured
data instead of asking it to "summarize F1" from general knowledge is what
keeps it factually grounded.

### Module 4 — Outcome Predictor
A gradient-boosting model (scikit-learn — already installed, no GPU
needed) trained on historical qualifying position, recent pace, and
circuit-specific performance, predicting finishing position. Framed
explicitly as "a prediction system to evaluate," not "a forecast that will
be right" — that framing itself is a signal of ML maturity.

### Module 5 — Knowledge Chatbot
Text-to-SQL or RAG over the same SQLite DB. The critical design rule: **it
must query the actual database, never invent an answer from the LLM's own
training data.** This is what separates it from "another generic chatbot"
— it's grounded in your specific dataset.

---

## 4. DevOps Concepts, Explained

**Version control workflow**: work happens on a branch per module/feature,
merged to `main` only via a Pull Request, and only once CI passes. This
isn't bureaucracy for its own sake — it means `main` is *always* in a
working state, and your commit history tells a story of incremental,
reviewed progress (visible to anyone looking at the repo).

**CI (Continuous Integration)** — `.github/workflows/ci.yml`: every push
automatically runs linting (`ruff`), formatting checks (`black`), and tests
(`pytest`). It also validates that the Docker image builds. This catches
mistakes *before* they reach `main`, without you having to remember to run
checks manually every time.

**Containerization (Docker)**: packages the app plus its exact dependencies
into a single reproducible unit. `docker compose up` should get anyone —
including a recruiter trying your project — from zero to running app.
Given your disk constraints, Docker here is a *packaging/validation step*,
not where you do daily development (see section 6).

---

## 5. Security: The CIA Triad, Concretely

The CIA triad is a standard security framework: **Confidentiality**
(only the right people/systems see data), **Integrity** (data isn't
corrupted or tampered with), **Availability** (the system stays usable).
Full detail lives in `docs/SECURITY.md` — the short version:

| Principle | Where it lives in this repo |
|---|---|
| Confidentiality | `.env` for secrets (never committed), API-key check + rate limiter in `src/api.py`, log redaction in `src/logging_conf.py` |
| Integrity | pydantic validation in `src/schemas.py`, transactional writes in `src/ingest.py` |
| Availability | `/health` endpoint, cache fallback on failed API pulls, `restart: unless-stopped` in `docker-compose.yml`, `scripts/backup.sh` |

Most student/portfolio projects have none of this. Having it — and being
able to explain *why* each piece exists — is what signals "this person
thinks like an engineer, not just someone who can call an API."

---

## 6. Hardware Constraints → Engineering Decisions

Your machine: 4 CPU threads, ~5.7GB RAM (single SODIMM, second slot empty),
~11GB free disk, no dedicated GPU. Rather than ignore this, we designed
around it — full reasoning in `docs/DECISIONS.md`:

- **LSTM training happens on Google Colab's free GPU tier**, not locally.
  You export the trained model (a small file) and only run *inference*
  on your laptop, via `onnxruntime` instead of the full `torch`/`tensorflow`
  packages (which alone can eat several GB of disk).
- **SQLite, not Postgres** — zero background daemon, minimal RAM footprint.
- **Docker as a packaging step**, not the primary dev loop — avoids
  running a Docker daemon and your dev tools simultaneously on a tight
  RAM budget, and avoids image bloat on a near-full disk.

This is a legitimate constraint-driven design, and documenting it (rather
than hiding it) turns a limitation into a demonstrated skill.

---

## 7. Repo Walkthrough

```
f1-platform/
├── src/
│   ├── config.py        # loads settings from .env, single source of truth
│   ├── db.py             # SQLAlchemy models + engine (SQLite now, swappable later)
│   ├── schemas.py        # pydantic validation — the integrity boundary
│   ├── ingest.py         # the ONLY path data takes into the DB
│   ├── logging_conf.py   # logging with secret redaction
│   ├── api.py            # FastAPI service: /health, /chat (rate-limited, API-key gated)
│   └── dashboard.py       # Streamlit UI (module 1, extend for 2-4)
├── tests/test_smoke.py   # pytest: schema validation + API health check
├── scripts/
│   ├── cleanup_disk.sh   # run before big installs — your disk is the tight constraint
│   └── backup.sh         # daily DB backup to a second local path
├── docs/
│   ├── SECURITY.md       # full CIA-triad threat model
│   └── DECISIONS.md      # engineering decisions log with reasoning
├── .github/workflows/ci.yml   # lint + test + docker-build on every push/PR
├── Dockerfile / docker-compose.yml
├── requirements.txt      # kept deliberately lean — no torch/tensorflow locally
└── Makefile              # make test / make ingest / make dashboard
```

---

## 8. What to Study / Build Next

1. Run `make test` — confirm the skeleton passes CI locally before touching
   anything else.
2. Run one real ingestion: `make ingest RACE="2024 Monaco R"`, then
   `make dashboard` to see it render.
3. Read `src/ingest.py` and `src/schemas.py` side by side — this is the
   pattern (pull → validate → transactional write) you'll repeat for every
   new data source.
4. Move your existing anomaly-detection notebook into `notebooks/`, adapt
   it for Colab, export to ONNX — that's Module 2's foundation.
5. Once 1 and 5 (dashboard + chatbot) are working end-to-end, you have a
   demoable platform even before modules 2-4 are finished.
