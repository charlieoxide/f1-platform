"""Module 1 — Race Dashboard. Reads from local SQLite, renders pace
comparison and tyre-degradation views. Data shaping lives in analysis.py
(tested separately); this file only wires that logic to the UI."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from src.analysis import pace_comparison, tyre_stints
from src.db import Lap, SessionLocal, init_db

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
