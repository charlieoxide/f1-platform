"""Module 1 — Race Dashboard. Reads from local SQLite, renders pace
comparison and tyre-degradation views. Data shaping lives in analysis.py
(tested separately); this file only wires that logic to the UI."""

import sys
from pathlib import Path

# Let `streamlit run src/dashboard.py` find the `src` package from any working
# directory on any OS, without needing PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from sqlalchemy import select

from src.analysis import pace_comparison, tyre_stints  # noqa: E402
from src.chatbot import answer_question  # noqa: E402
from src.config import settings  # noqa: E402
from src.db import Lap, SessionLocal, init_db  # noqa: E402
from src.summarizer import summarize_race  # noqa: E402
from src.telemetry import (  # noqa: E402
    coach_report,
    describe_clock_offset,
    describe_sector_check,
)

st.set_page_config(page_title="F1 Unified Platform", layout="wide")
st.title("F1 Unified Platform")

init_db()

with SessionLocal() as db:
    rows = db.execute(select(Lap)).scalars().all()

if not rows:
    st.info('No data yet. Run `make ingest RACE="2024 Monaco R"` first.')
    st.stop()

df = pd.DataFrame(
    [
        {
            "session": r.session_key,
            "driver": r.driver,
            "lap": r.lap_number,
            "lap_time_s": r.lap_time_seconds,
            "compound": r.compound,
            "stint": r.stint,
        }
        for r in rows
    ]
)

sessions = sorted(df["session"].unique())
session = st.selectbox("Session", sessions, index=len(sessions) - 1)
df = df[df["session"] == session]
drivers = sorted(df["driver"].unique())

st.subheader("Pace comparison")
col1, col2 = st.columns(2)
driver_a = col1.selectbox("Driver A", drivers, index=0)
driver_b = col2.selectbox("Driver B", drivers, index=min(1, len(drivers) - 1))

pace = pace_comparison(df, driver_a, driver_b)
if pace.empty:
    st.info("No overlapping timed laps for these two drivers.")
else:
    st.line_chart(pace)

st.subheader("Tyre degradation")
driver = st.selectbox("Driver", drivers, key="tyre_driver")
stints = tyre_stints(df, driver)
if stints.empty:
    st.info("No stint/tyre data for this driver in this session.")
else:
    st.line_chart(stints)

with st.expander("Raw laps"):
    st.dataframe(df.sort_values(["driver", "lap"]))

st.subheader("Race summary")
if not settings.openrouter_api_key:
    st.info("Set OPENROUTER_API_KEY in .env to enable the race summary.")
elif st.button("Generate summary for this session"):
    with st.spinner("Summarizing..."):
        summary_result = summarize_race(session, api_key=settings.openrouter_api_key)
    if summary_result["error"]:
        st.error(summary_result["error"])
    else:
        st.write(summary_result["summary"])
        with st.expander("Facts this was grounded in"):
            st.json(summary_result["facts"])

st.subheader("Driver coach")
st.caption(
    "Compares two drivers' fastest laps using telemetry from OpenF1 (free API; "
    "~4 samples/s, so braking points are good to roughly 10-20 m). Results are "
    "cached after the first run."
)
c1, c2, c3 = st.columns(3)
coach_session = c1.text_input("Session", "2024 Monaco Q")
coach_driver = c2.text_input("Driver", "NOR")
coach_ref = c3.text_input("Reference", "VER")
if st.button("Run coach"):
    try:
        with st.spinner("Loading telemetry..."):
            coach = coach_report(coach_session, coach_driver, coach_ref)
    except Exception as e:  # noqa: BLE001 - show a message, never a traceback
        st.error(f"Could not run the coach: {e}")
    else:
        st.write(
            f"**{coach_driver.upper()} vs {coach_ref.upper()}: "
            f"{coach['total_delta_s']:+.3f} s** over the lap "
            "(positive = behind the reference)"
        )
        for corner in coach["corners"][:5]:
            st.write("- " + corner.message)
        times = coach.get("lap_times_s")
        if times:
            d, r = coach_driver.upper().strip(), coach_ref.upper().strip()
            st.caption(
                f"Official lap times: {d} {times[d]:.3f} s, {r} {times[r]:.3f} s "
                f"(gap {times[d] - times[r]:+.3f} s)"
            )
        offset_line = describe_clock_offset(coach.get("clock_offset"))
        if offset_line:
            st.caption(offset_line)
        checks = coach.get("sector_check")
        if checks:
            text, worst = describe_sector_check(checks)
            st.caption(f"Check against official sector splits - {text}")
            if worst > 0.05:
                st.warning(
                    f"The coach's timing is off by up to {worst:.2f} s against "
                    "the official sector splits, so treat per-corner times as rough."
                )
        st.caption("Running time delta along the lap (rising = losing time)")
        st.line_chart(coach["delta"].set_index("Distance")["delta_s"])

st.subheader("Ask the data")
if not settings.openrouter_api_key:
    st.info("Set OPENROUTER_API_KEY in .env to enable the chatbot.")
else:
    question = st.text_input(
        "e.g. 'Who had the five fastest laps and what were their times?'"
    )
    if question:
        with st.spinner("Querying..."):
            result = answer_question(question, api_key=settings.openrouter_api_key)
        st.code(result["sql"], language="sql")
        if result["error"]:
            st.error(result["error"])
        else:
            st.dataframe(result["rows"])
