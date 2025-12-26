# pages/download_summaries.py
import streamlit as st
from functions.download_summaries import require_plates, ui_growth_summaries, ui_export

st.title("Download Processed Datasets and Plots")

plates = require_plates()

st.subheader("Export Processed Data")
ui_export(plates)

st.divider()

st.subheader("Plot Processed Data")
ui_growth_summaries(plates)
