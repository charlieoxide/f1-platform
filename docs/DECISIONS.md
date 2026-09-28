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
