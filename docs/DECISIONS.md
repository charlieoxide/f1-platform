# Engineering Decisions

Short log of decisions and why, so the reasoning survives past the commit
that made it.

## 2026-09 — Train deep learning off-device, serve locally

**Context:** Local machine has ~5.7GB RAM (single SODIMM, second slot empty),
4 CPU threads, no dedicated GPU, and only ~11GB free disk.

**Decision:** The LSTM autoencoder (driver-coach / anomaly detection module)
is trained in a Google Colab notebook (free T4 GPU tier). The trained model
is exported (ONNX or a small `.h5`/`.pt` state dict, ideally <50MB) and
committed to the repo under `models/`. Only inference runs on the local
machine, via a lightweight runtime (no local `tensorflow`/`torch` install
required if ONNX Runtime is used instead).

**Consequence:** Retraining requires re-opening the Colab notebook rather than
a local `make train`. Documented explicitly so it isn't mistaken for an
oversight.

## 2026-09 — SQLite over Postgres

**Context:** Single developer, single machine, RAM-constrained.

**Decision:** SQLite as the primary store, accessed through SQLAlchemy so a
future move to Postgres (if the app goes multi-user / cloud) is a config
change, not a rewrite.

## 2026-09 — Docker as packaging step, not primary dev loop

**Context:** ~11GB free disk; Docker images and layer cache can consume this
quickly.

**Decision:** Day-to-day development happens in a local `venv`. Docker
(`python:3.11-slim` base, `.dockerignore` covering `data/`, `.venv/`,
`__pycache__/`) is used to validate reproducibility periodically and for any
eventual deployment, not run continuously alongside training/dev work.

## 2026-09 — Branch protection + CI gate on `main`

**Decision:** All changes go through a PR; GitHub Actions must pass (lint +
pytest) before merge. No direct pushes to `main`.

## 2026-10 — OpenF1 as the telemetry source for the driver coach

**Context:** FastF1 loads telemetry from F1's static timing server. By Oct
2026 that server answered with HTTP 401, so FastF1 reported "Car telemetry
data is unavailable" for every session tried (2023, 2024, 2026), and could
not fetch any session it had not already cached.

**Decision:** The coach reads telemetry from OpenF1 (free API) instead. The
FastF1 path stays behind `--source fastf1` for if access returns. The coach
maths (`coach.py`) is source-agnostic, so only a loader changed.

**Consequences:** OpenF1 has no distance channel (distance is the integral
of speed), ~3.8 samples/s (braking points good to ~10-20 m), no corner map
(zones are labelled "Braking zone N"), and does not flag deleted laps.
Session and driver names must match clearly, otherwise the call fails and
lists the options - the lesson of the Austria/"Great Britain" mislabel.
