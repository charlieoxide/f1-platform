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

## 2026-10 — How the driver coach is validated (and what it won't claim)

**Context:** The first run on real Monaco 2024 qualifying telemetry failed
its own sanity checks: the lap gap had the wrong sign versus the official
lap times, "braking 150 m earlier" appeared repeatedly (an artefact of a
search-window edge), and the running delta swung by up to 0.6 s at slow
corners and recovered within ~150 m, which no driver can do.

**Decisions:**
- Time comes from the measured timestamps, not from integrating speed.
- Braking zones are matched within 75 m; otherwise the output says
  "already braking" / "no braking onset" instead of a number.
- Differences below the telemetry's sample spacing (~12 m median) are not
  reported.
- Accuracy is checked against the official sector splits (fixed points on
  the track), and the UI warns when the coach disagrees by more than 0.05 s.

**Rejected:** aligning laps at their slowest points. Speed barely changes
near a corner's minimum, so that position is poorly defined and forcing it
to match would add error exactly where timing is most sensitive.

**Open:** the cause of the phantom swings is not yet established; the
`--trace-at` diagnostic exists to find it.

## 2026-10 — Telemetry clock offset between drivers (OpenF1)

**Finding:** On real 2024 Monaco qualifying data (NOR vs VER) the sector-split
check failed by ~0.1 s at both sector lines, and the running delta showed
0.5-0.65 s swings at slow corners that recovered within ~150 m. The aligned
traces showed every event in one lap (brake on, brake off, minimum speed)
shifted by a similar 10-20 m. A simulation reproduces this exactly: delaying
one car's telemetry clock by 0.13 s drops the coach's delta by ~0.22 s at
both sector lines. Cause (hypothesis, consistent with the evidence): OpenF1
stamps telemetry and lap timing from different systems, so one car's data
can run ~0.1 s early/late relative to the other's.

**Decision:** Estimate the relative offset per pair of laps by trying shifts
of +/-0.4 s and keeping the one where the speed traces line up best; apply
it only if it is strictly inside the search range and improves the fit by
>= 3%. On simulated laps this recovers planted offsets (+0.13, +0.26,
-0.15 s) to within ~0.03 s and brings the sector-check error from 0.22-0.39 s
to ~0.04 s.

**Limits:** Only the *relative* offset can be found, which is all a
two-lap comparison needs. The fit also absorbs a little of any consistent
real braking difference (~0.02 s bias in simulation). The sector-split check
remains the independent test of whether the result can be trusted, and the
UI warns when it disagrees by more than 0.05 s.

## 2026-11 — Ingestion moves to OpenF1 (free); FastF1 becomes optional

**Finding:** FastF1 3.9 reads F1's timing server through an F1 account and its own
error says an active F1TV subscription is required; without it the server answers
HTTP 401. A paid subscription breaks this project's "free tools" rule and means a
fresh clone could never get data.

**Decision:** `python -m src.ingest` defaults to OpenF1 (`--source fastf1` keeps the old
path). Probed against real 2024 Monaco data first: OpenF1 returns the same 1237 race
laps FastF1 gave, plus official points. Mapping choices: stints are assigned by
`lap_start` only, because OpenF1 returns `lap_end: 0` for some stints; a car a lap down is
"+N Lap", not a retirement; and `starting_grid` was empty for the race session, so grid
positions fall back to the qualifying classification (penalties not reflected) and the
ingest output says which source was used. Stored results are never replaced by an
empty answer.

**Also fixed on the way:** the summariser counted "+1 Lap" finishers as retirements, and
two API tests failed on any machine whose local `.env` set `F1_API_KEY` (tests now ignore
the developer's `.env`; the API-key and rate-limit paths now have their own tests).
