# F1 Unified Platform(Under Devlopment If Help Me To Build Contact kolidarshan890@gmail.com as Subject Help F1)

One platform, one shared database, five F1 data/AI modules. Runs on Windows,
macOS and Linux (CI tests all three).

| Module | What it does | Status |
|---|---|---|
| 1. Race dashboard | Pace comparison, tyre-degradation per stint | Working, on real 2024 data |
| 2. Driver coach | Compares two drivers' fastest laps from telemetry: braking points, apex speed, throttle return, time gained/lost | Working; reports its own accuracy (see below) |
| 3. Race summariser | Short race report written by an LLM from database facts only | Working; output is validated and retried |
| 4. Outcome predictor | Gradient-boosting model for finishing position, leakage-tested, chronological holdout | Working; honest MAE (~3.3 positions on 2024 data) |
| 5. Knowledge chatbot | Natural-language question -> SQL -> real rows | Working; read-only database, validated SQL |

## Run it

Needs Python 3.11-3.14; **3.12 is recommended** (it is what CI and the tests use). Python 3.15 is
too new for now: FastF1 requires pandas 2, which has no 3.15 builds. No `make` required.

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python --version                  # should say 3.12.x
python -m pip install -r requirements.txt
copy .env.example .env            # then add your OpenRouter key (optional, for modules 3 and 5)
python tasks.py test
python tasks.py dashboard
```

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python tasks.py test
python tasks.py dashboard
```

| Command | Does |
|---|---|
| `python tasks.py test` | lint + format check + all tests |
| `python tasks.py dashboard` | open the app in your browser |
| `python tasks.py coach --session "2024 Monaco Q" --driver NOR --reference VER` | driver coach in the terminal |
| `python tasks.py train` | train and evaluate the predictor |
| `python tasks.py sql "SELECT driver, COUNT(*) FROM laps GROUP BY driver"` | read-only query (no sqlite3 CLI needed) |
| `python tasks.py backup` | consistent snapshot of the database |
| `python tasks.py ingest --session "2024 Monaco R"` | load a session (see Data) |

`make <task>` still works on Linux/macOS and just calls the same commands.

## Data

- **Laps and results** are stored in a local SQLite file (`data/f1.db`), which is
  **not in git**: a fresh clone starts empty. Fill it for free with
  `python tasks.py ingest-season 2024` (about a minute; no account or key).
- **Ingestion uses OpenF1 (free, no account).** Starting grid comes from OpenF1's
  `starting_grid` when it has data, otherwise from the qualifying classification (which
  ignores grid penalties); the ingest command prints which one was used. FastF1 is still
  available with `--source fastf1`, but F1 now requires a paid F1TV subscription for it
  (`python -m fastf1 auth --authenticate`).
- **The driver coach uses OpenF1** (free API, no key) for telemetry, so it works
  on a fresh clone with no database at all.

## How it is built

- **Validation at every boundary.** External data passes pydantic schemas before it
  reaches the database; bad rows are skipped and logged, never stored.
- **LLM output is untrusted.** The chatbot only produces SQL, which is checked and run on
  a read-only connection; the summariser only sees facts from the database, and its
  reply is validated and retried (a free model once answered "User Safety: safe").
- **The coach checks itself.** It compares its time deltas against the official sector
  splits and warns when they disagree by more than 0.05 s. It also estimates and removes
  the clock offset between two cars' telemetry, which otherwise creates phantom
  half-second swings at slow corners. Differences below the telemetry's sample spacing
  (~11 m) are not reported.
- **Failures are structured.** A bad key, a network error or a rate limit shows a message;
  it never takes the app down.

Design decisions and the bugs behind them are in `docs/DECISIONS.md`; the security model is
in `docs/SECURITY.md`.

## Known limitations

- OpenF1 has no retirement reasons (statuses are Finished / +N Lap / DNF / DNS / DSQ) and its
  grid data can be missing, in which case grid positions are approximate.
- Telemetry is ~4 samples/second, so braking points are only good to ~10-20 m.
- The coach cannot name corners (OpenF1 has no corner map); zones are numbered along the lap.
- Free LLM models vary in quality; set `OPENROUTER_MODEL` in `.env` to pin one.
- The predictor is trained on a handful of 2024 races; treat its error figure as the
  honest result, not a forecast.
