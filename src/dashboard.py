"""Module 1 — Race Dashboard. Minimal starting point: reads from the local
SQLite DB, shows lap times per driver. Extend with tyre degradation / track
evolution views next."""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from src.db import Lap, SessionLocal, init_db

st.set_page_config(page_title="F1 Unified Platform", layout="wide")
st.title("F1 Unified Platform")

init_db()

with SessionLocal() as db:
    rows = db.execute(select(Lap)).scalars().all()

if not rows:
    st.info('No data yet. Run `make ingest RACE="2024 Monaco R"` first.')
else:
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
    driver = st.selectbox("Driver", sorted(df["driver"].unique()))
    st.line_chart(df[df["driver"] == driver].set_index("lap")["lap_time_s"])
    st.dataframe(df[df["driver"] == driver])
