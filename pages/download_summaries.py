# pages/download_summaries.py
import streamlit as st
from pages.analysis_shared import require_plates, ui_growth_summaries, ui_export

st.title("Download summaries")

plates = require_plates()

st.subheader("Summaries")
ui_growth_summaries(plates)

st.divider()

st.subheader("Export")
ui_export(plates)
